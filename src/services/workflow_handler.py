import logging
import time

from src.clients.database_client import DatabaseClient
from src.services.analysis.analyzer import Analyzer
from src.services.nats_client import NatsClient

logger = logging.getLogger(__name__)

# NATS subjects
SUBJECT_WORKFLOW_STARTED = "visiobook.project.workflow.started"
SUBJECT_ANALYSIS_COMPLETED = "visiobook.ai.analysis.completed"
SUBJECT_ANALYSIS_FAILED = "visiobook.ai.analysis.failed"
SUBJECT_PROGRESS = "visiobook.ai.progress"


class WorkflowHandler:
    def __init__(
        self,
        nats_client: NatsClient,
        analyzer: Analyzer,
        db_client: DatabaseClient | None = None,
    ):
        self.nats = nats_client
        self.analyzer = analyzer
        self.db = db_client or DatabaseClient()

    async def handle_workflow_started(self, data: dict):
        """Handle visiobook.project.workflow.started event."""
        project_id = data.get("projectId", "")
        version_id = data.get("versionId", "")
        execution_id = data.get("executionId", "")
        user_id = data.get("userId", "")
        correlation_id = data.get("correlationId", "")
        content_text = data.get("contentText", "")
        config = data.get("config", {})

        logger.info(
            "Received workflow.started: projectId=%s, executionId=%s",
            project_id,
            execution_id,
        )

        if not content_text:
            error_msg = "No content text provided"
            await self._persist_failure(
                project_id, version_id, execution_id, user_id, correlation_id, error_msg
            )
            await self._publish_failed(
                project_id, version_id, execution_id, user_id, correlation_id, error_msg
            )
            return

        # Publish progress: starting (0%)
        await self._publish_progress(
            project_id, version_id, execution_id, correlation_id, 0
        )

        start_time = time.monotonic()

        try:
            # Determine language from config
            language = config.get("language", "auto")

            # Publish progress: preprocessing (10%)
            await self._publish_progress(
                project_id, version_id, execution_id, correlation_id, 10
            )

            # Run the analysis using existing Analyzer
            result = await self.analyzer.analyze(content_text, language=language)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Publish progress: parsing (80%)
            await self._publish_progress(
                project_id, version_id, execution_id, correlation_id, 80
            )

            # Map scenes to core-project-service format
            scenes = self._map_scenes(result.get("scenes", []))

            # Map characters to core-project-service format
            characters = self._map_characters(result.get("characters", []))

            # Persist full analysis result to database
            await self.db.save_analysis(
                project_id=project_id,
                version_id=version_id,
                execution_id=execution_id,
                user_id=user_id,
                status="completed",
                scenes=scenes,
                characters=characters,
                narrative=result.get("narrative"),
                sentiment=result.get("sentiment"),
                summary=result.get("summary"),
                text_stats=result.get("text_stats"),
                language=result.get("language", language),
                processing_time_ms=elapsed_ms,
                correlation_id=correlation_id,
            )

            # Publish analysis completed
            await self.nats.publish(
                SUBJECT_ANALYSIS_COMPLETED,
                {
                    "projectId": project_id,
                    "versionId": version_id,
                    "executionId": execution_id,
                    "userId": user_id,
                    "scenes": scenes,
                    "characters": characters,
                    "correlationId": correlation_id,
                },
            )

            # Publish progress: done (100%)
            await self._publish_progress(
                project_id, version_id, execution_id, correlation_id, 100
            )

            logger.info(
                "Analysis completed: projectId=%s, scenes=%d, characters=%d, elapsed=%.0fms",
                project_id,
                len(scenes),
                len(characters),
                elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Analysis failed for projectId=%s: %s",
                project_id,
                str(e),
                exc_info=True,
            )
            await self._persist_failure(
                project_id,
                version_id,
                execution_id,
                user_id,
                correlation_id,
                str(e),
                elapsed_ms,
            )
            await self._publish_failed(
                project_id, version_id, execution_id, user_id, correlation_id, str(e)
            )

    def _map_scenes(self, raw_scenes: list) -> list:
        """Map ai-analysis-service scene format to core-project-service format."""
        scenes = []
        for i, scene in enumerate(raw_scenes):
            # Build image prompt from atmosphere and setting
            image_prompt = self._build_image_prompt(scene)

            # Estimate duration from text length (~5s per 100 words)
            text = scene.get("text_excerpt", "")
            word_count = len(text.split()) if text else 0
            duration = max(3, min(30, int(word_count / 20)))  # 3-30 seconds

            scenes.append(
                {
                    "order": (
                        scene.get("scene_id", i)
                        if isinstance(scene.get("scene_id"), int)
                        else i
                    ),
                    "text": text,
                    "description": scene.get("title", ""),
                    "imagePrompt": image_prompt,
                    "duration": duration,
                    "sentiment": (
                        scene.get("atmosphere", {}).get("mood", "neutral")
                        if isinstance(scene.get("atmosphere"), dict)
                        else "neutral"
                    ),
                }
            )
        return scenes

    def _map_characters(self, raw_characters: list) -> list:
        """Map ai-analysis-service character format to core-project-service format."""
        characters = []
        for char in raw_characters:
            role = char.get("role", "")
            physical = char.get("physical_description", "")
            description = f"{role}. {physical}".strip(". ") if role or physical else ""

            characters.append(
                {
                    "name": char.get("name", ""),
                    "description": description,
                    "aliases": [],
                    "traits": char.get("personality_traits", []),
                }
            )
        return characters

    def _build_image_prompt(self, scene: dict) -> str:
        """Build a detailed image generation prompt from scene data."""
        parts = []

        # Title/description
        title = scene.get("title", "")
        if title:
            parts.append(title)

        # Setting
        setting = scene.get("setting", {})
        if isinstance(setting, dict):
            location = setting.get("location", "")
            time_of_day = setting.get("time_of_day", "")
            if location:
                parts.append(f"Location: {location}")
            if time_of_day:
                parts.append(f"Time: {time_of_day}")

        # Atmosphere
        atmosphere = scene.get("atmosphere", {})
        if isinstance(atmosphere, dict):
            mood = atmosphere.get("mood", "")
            lighting = atmosphere.get("lighting", "")
            colors = atmosphere.get("colors", [])
            if mood:
                parts.append(f"Mood: {mood}")
            if lighting:
                parts.append(f"Lighting: {lighting}")
            if colors and isinstance(colors, list):
                parts.append(f"Colors: {', '.join(colors)}")

        # Characters present
        chars = scene.get("characters_present", [])
        if chars and isinstance(chars, list):
            parts.append(f"Characters: {', '.join(chars)}")

        return ". ".join(parts) if parts else "A scene from the story"

    async def _publish_progress(
        self,
        project_id: str,
        version_id: str,
        execution_id: str,
        correlation_id: str,
        progress: int,
    ):
        await self.nats.publish(
            SUBJECT_PROGRESS,
            {
                "projectId": project_id,
                "versionId": version_id,
                "executionId": execution_id,
                "step": "analysis",
                "progress": progress,
                "correlationId": correlation_id,
            },
        )

    async def _persist_failure(
        self,
        project_id: str,
        version_id: str,
        execution_id: str,
        user_id: str,
        correlation_id: str,
        error: str,
        elapsed_ms: float = 0,
    ):
        try:
            await self.db.save_analysis(
                project_id=project_id,
                version_id=version_id,
                execution_id=execution_id,
                user_id=user_id,
                status="failed",
                error=error,
                processing_time_ms=elapsed_ms,
                correlation_id=correlation_id,
            )
        except Exception as e:
            logger.error("Failed to persist failure record: %s", e)

    async def _publish_failed(
        self,
        project_id: str,
        version_id: str,
        execution_id: str,
        user_id: str,
        correlation_id: str,
        error: str,
    ):
        await self.nats.publish(
            SUBJECT_ANALYSIS_FAILED,
            {
                "projectId": project_id,
                "versionId": version_id,
                "executionId": execution_id,
                "userId": user_id,
                "error": error,
                "correlationId": correlation_id,
            },
        )
