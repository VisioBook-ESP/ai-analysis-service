import asyncio

import nats
from nats.js.api import ConsumerConfig, AckPolicy
import json
import logging
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class NatsClient:
    def __init__(self, url: str, user: str = "", password: str = ""):
        self.url = url
        self.user = user or None
        self.password = password or None
        self.nc = None
        self.js = None
        self._subscriptions = []

    async def connect(self):
        try:
            self.nc = await nats.connect(
                self.url,
                user=self.user,
                password=self.password,
                reconnect_time_wait=2,
                max_reconnect_attempts=10,
            )
            self.js = self.nc.jetstream()
            logger.info("Connected to NATS at %s", self.url)
        except Exception as e:
            logger.error("Failed to connect to NATS: %s", e)
            raise

    async def publish(self, subject: str, payload: dict):
        if not self.js:
            logger.warning("NATS not connected, cannot publish to %s", subject)
            return
        data = json.dumps(payload).encode()
        ack = await self.js.publish(subject, data)
        logger.info("Published to %s (stream=%s, seq=%d)", subject, ack.stream, ack.seq)

    async def subscribe(
        self,
        stream: str,
        subject: str,
        durable: str,
        callback: Callable[[dict], Awaitable[None]],
        max_retries: int = 10,
        retry_delay: float = 3.0,
    ):
        if not self.js:
            logger.warning("NATS not connected, cannot subscribe to %s", subject)
            return

        sub = None
        for attempt in range(1, max_retries + 1):
            try:
                sub = await self.js.subscribe(
                    subject,
                    durable=durable,
                    stream=stream,
                    config=ConsumerConfig(ack_policy=AckPolicy.EXPLICIT),
                )
                break
            except Exception as e:
                if "already bound" not in str(e):
                    raise
                if attempt <= 3:
                    logger.warning(
                        "Consumer %s already bound (attempt %d/%d), "
                        "waiting for old pod to drain...",
                        durable,
                        attempt,
                        max_retries,
                    )
                    await asyncio.sleep(retry_delay)
                elif attempt == 4:
                    logger.warning(
                        "Consumer %s still bound after %d retries, "
                        "deleting stale consumer and recreating...",
                        durable,
                        attempt - 1,
                    )
                    try:
                        await self.js.delete_consumer(stream, durable)
                        logger.info("Deleted stale consumer %s", durable)
                    except Exception as del_err:
                        logger.error(
                            "Failed to delete consumer %s: %s", durable, del_err
                        )
                    await asyncio.sleep(1)
                else:
                    if attempt < max_retries:
                        logger.warning(
                            "Consumer %s still bound after delete (attempt %d/%d), retrying...",
                            durable,
                            attempt,
                            max_retries,
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        raise

        if sub is None:
            raise RuntimeError(
                f"Failed to subscribe to {subject} after {max_retries} attempts"
            )

        self._subscriptions.append(sub)
        logger.info("Subscribed to %s (durable=%s)", subject, durable)

        async for msg in sub.messages:
            try:
                data = json.loads(msg.data.decode())
                await callback(data)
                await msg.ack()
            except Exception as e:
                logger.error("Error processing message from %s: %s", subject, e)
                await msg.nak()

    async def close(self):
        for sub in self._subscriptions:
            try:
                await sub.unsubscribe()
            except Exception:
                pass
        if self.nc:
            await self.nc.drain()
            logger.info("NATS connection drained")
