import nats
from nats.js.api import ConsumerConfig, AckPolicy
import json
import logging
import asyncio
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
    ):
        if not self.js:
            logger.warning("NATS not connected, cannot subscribe to %s", subject)
            return

        sub = await self.js.subscribe(
            subject,
            durable=durable,
            stream=stream,
            config=ConsumerConfig(ack_policy=AckPolicy.EXPLICIT),
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
