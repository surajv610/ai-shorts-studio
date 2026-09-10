"""Video Agent — animates each storyboard scene's LOCKED image into an MP4 clip.

For every scene that does not yet have a completed current clip, the agent:

1. Validates the scene has a locked selected image (the visual anchor).
2. Reads the idea + Project Bible + complete storyboard + current/previous/next
   scene context + the locked image + its stored image prompt.
3. Asks the LLM (structured output) for an animation prompt that explicitly
   separates GLOBAL CONTINUITY from SCENE-SPECIFIC ACTION and focuses on WHAT
   CHANGES rather than re-describing the image.
4. Submits image-to-video generation through the generic provider abstraction
   (async), polls the operation to completion (bounded timeout), downloads the
   result into the project's own ``videos/`` directory, and records everything.

The agent is resume-safe: scenes with a completed current clip are never
regenerated; failed/pending scenes are retried; a restart keeps completed clips,
generation ids, prompts, and failure state, resuming only from incomplete
scenes. Regeneration creates a NEW attempt while preserving the previous
generation record (history), and only promotes the new clip to the scene's
current clip after it SUCCEEDs.

Google-specific behavior stays inside the provider adapter — this module only
talks to the generic async VideoProvider interface.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from backend.config import Settings, get_settings
from backend.models import Project, Scene, VideoGeneration
from backend.providers.base import (
    AsyncJob,
    AsyncJobStatus,
    GenerationOptions,
    GenerationStatus,
    ProviderError,
    VideoGenerationResult,
)
from backend.providers.registry import get_video_provider
from backend.storage import project_asset_dir, project_root, save_project
from agents.llm import call_structured
from agents.video_schemas import (
    VIDEO_PROMPT_SCHEMA,
    VideoPromptError,
    VideoPromptOutput,
    validate_video_prompt_output,
)

# Status strings we persist on a VideoGeneration record.
STATUS_SUBMITTED = "SUBMITTED"
STATUS_PROCESSING = "PROCESSING"

SYSTEM_PROMPT = (
    "You are the Video Agent for an AI Shorts production studio. You write "
    "image-to-video animation prompts for vertical (9:16) YouTube Shorts, one "
    "per storyboard scene. You explicitly separate GLOBAL CONTINUITY (what must "
    "stay pixel-identical across the whole video: subject identity, plant/"
    "species, environment, soil, composition, camera perspective, visual style, "
    "lighting, and other important visual features) from SCENE-SPECIFIC ACTION "
    "(the ONE progression step happening in this scene). Always: focus on WHAT "
    "CHANGES instead of re-describing the source image; keep the camera locked "
    "unless the storyboard explicitly calls for movement (never allow arbitrary "
    "camera drift); describe realistic motion and environmental behavior; add "
    "timelapse behavior when the content type is a timelapse (slow chronological "
    "growth, nothing else changing); include clear negative constraints. Always "
    "return the exact JSON schema provided with every field populated."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _story_scenes(project: Project) -> list:
    return (project.storyboard or {}).get("scenes") or []


def _bible_lines(bible: dict) -> str:
    if not bible:
        return "  (none provided)"
    lines = []
    for key, value in bible.items():
        if value is None:
            continue
        label = str(key).replace("_", " ")
        lines.append(f"  {label}: {value}")
    return "\n".join(lines)


def _scene_line(prefix: str, story_scene: Optional[dict]) -> str:
    if not story_scene:
        return ""
    return (
        f"{prefix}:\n"
        f"  scene number: {story_scene.get('scene_number')}\n"
        f"  title: {story_scene.get('title', '')}\n"
        f"  visual description: {story_scene.get('visual_description', '')}\n"
        f"  action: {story_scene.get('action', '')}\n"
        f"  continuity notes: {story_scene.get('continuity_notes', '')}\n"
        f"  duration: {story_scene.get('estimated_duration')} seconds"
    )


def _build_user_message(
    idea: str,
    bible: dict,
    story_scene: Optional[dict],
    previous_scene: Optional[dict],
    next_scene: Optional[dict],
    aspect_ratio: str,
    source_image_id: str,
    source_image_prompt: Optional[str],
) -> str:
    msg = (
        f"PROJECT IDEA: {idea}\n"
        f"ASPECT RATIO: {aspect_ratio}\n"
        f"LOCKED SOURCE IMAGE: candidate {source_image_id or 'n/a'}\n"
        "Generate an image-to-video animation prompt for a vertical 9:16 "
        "composition. The selected image is the visual anchor — preserve its "
        "identity, composition, environment, style, camera, and lighting.\n\n"
        "PROJECT BIBLE (global continuity):\n"
        f"{_bible_lines(bible)}\n\n"
    )
    if source_image_prompt:
        msg += (
            "SOURCE IMAGE PROMPT (what the image was generated from; reuse its "
            "continuity):\n"
            f"{source_image_prompt}\n\n"
        )
    msg += _scene_line("CURRENT SCENE", story_scene) + "\n"
    if previous_scene:
        msg += _scene_line("PREVIOUS SCENE (already happened)", previous_scene) + "\n"
    if next_scene:
        msg += _scene_line("NEXT SCENE (will happen after)", next_scene) + "\n"
    msg += (
        "Return structured output. The sharing of prompts across scenes is what "
        "keeps the project visually coherent; describe only what changes in this "
        "scene. The final `prompt` field is the full text sent to the video model."
    )
    return msg


class VideoAgent:
    """Orchestrates image-to-video generation for a project's scenes."""

    def __init__(
        self,
        *,
        llm=call_structured,
        video_provider=None,
        settings: Optional[Settings] = None,
    ):
        self._llm = llm
        self._video_provider = video_provider
        self._settings = settings or get_settings()

    # -- provider resolution ---------------------------------------------

    def _provider(self):
        if self._video_provider is not None:
            return self._video_provider
        return get_video_provider(self._settings)

    # -- public flow -----------------------------------------------------

    def run(self, project: Project, *, model: Optional[str] = None) -> Dict[str, int]:
        """Generate clips for every scene that still needs one.

        Completed scenes (a SUCCEEDED current clip) are never regenerated.
        Scenes without a locked image are counted as needing selection and are
        skipped. Failed scenes are retried. Saves after each scene so a restart
        can resume without repeating work.

        Returns a summary dict: scenes_ready / scenes_generated / scenes_failed /
        scenes_pending / scenes_needs_selection.
        """
        summary = {
            "scenes_ready": 0,
            "scenes_generated": 0,
            "scenes_failed": 0,
            "scenes_pending": 0,
            "scenes_needs_selection": 0,
        }
        for index, scene in enumerate(project.scenes):
            if self._is_complete(scene):
                summary["scenes_ready"] += 1
                continue
            try:
                self._locked_image(project, scene)
            except ValueError:
                summary["scenes_needs_selection"] += 1
                continue
            try:
                self.generate_scene(project, scene, index=index, model=model)
            except Exception as e:  # per-scene isolation; never block other scenes
                self._mark_scene_failed(project, scene, e)
            save_project(project)
            current = self.current_generation(scene)
            if current is not None and current.status == "SUCCEEDED":
                summary["scenes_ready"] += 1
                summary["scenes_generated"] += 1
            elif current is not None and current.status in (
                STATUS_SUBMITTED,
                STATUS_PROCESSING,
            ):
                summary["scenes_pending"] += 1
            else:
                summary["scenes_failed"] += 1
        return summary

    def generate_scene(
        self,
        project: Project,
        scene: Scene,
        *,
        index: int = 0,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Scene:
        """Submit + poll + download a clip for one scene.

        ``prompt`` may be supplied to reuse an existing/retried prompt;
        otherwise a fresh structured animation prompt is generated.
        """
        source_image, source_image_id = self._locked_image(project, scene)

        story_scene = _story_scenes(project)
        prompt_text = prompt
        if prompt_text is None:
            output = self._prompt_for_scene(
                project,
                scene,
                story_scene,
                index=index,
                model=model,
                source_image_id=source_image_id,
            )
            prompt_text = output.prompt
            scene.video_prompt = prompt_text

        options = self._options(project, scene)
        generation = VideoGeneration(
            project_id=project.id,
            scene_id=scene.id,
            provider="",
            prompt=prompt_text,
            source_image_id=source_image_id or "",
            source_image=source_image,
            duration_seconds=max(1, int(scene.duration_seconds or 0)),
        )
        scene.video_generations.append(generation)
        save_project(project)

        try:
            provider = self._provider()
            job = provider.submit_async(source_image, prompt_text, options=options)
        except ProviderError as e:
            generation.status = "FAILED"
            generation.error = self._sanitize_error(e)
            generation.completed_at = _now()
            save_project(project)
            raise
        except Exception as e:  # unexpected non-provider error still isolated
            generation.status = "FAILED"
            generation.error = self._sanitize_error(f"Provider submission error: {e}")
            generation.completed_at = _now()
            save_project(project)
            raise

        generation.provider = getattr(provider, "name", "")
        generation.job_id = getattr(job, "job_id", "") or ""
        generation.model = getattr(provider, "model", "") or options.model or ""
        generation.status = STATUS_SUBMITTED
        generation.submitted_at = _now()
        generation.raw = dict(getattr(job, "raw", {}) or {})
        save_project(project)

        job = self._poll(provider, job, options)
        self._apply_job_state(generation, job, options)

        if generation.status == "SUCCEEDED":
            scene.video_clip = generation.output_path
            scene.current_video_id = generation.attempt_id
        save_project(project)
        return scene

    def generate_scene_from_existing(
        self,
        project: Project,
        scene_id: str,
        *,
        model: Optional[str] = None,
    ) -> str:
        """Generate ONE scene's clip, reusing its prompt or creating one.

        Mirrors ``run()`` for a single scene (used by the API). Returns the
        status of the latest attempt after the sync loop finishes.
        """
        scene = self._find_scene(project, scene_id)
        self._locked_image(project, scene)
        index = project.scenes.index(scene)
        prompt = scene.video_prompt or (
            self.latest_generation(scene).prompt if scene.video_generations else None
        )
        if prompt:
            self.generate_scene(project, scene, index=index, model=model, prompt=prompt)
        else:
            self.generate_scene(project, scene, index=index, model=model)
        save_project(project)
        gen = self.latest_generation(scene)
        return gen.status if gen else "PENDING"

    def regenerate_clip(
        self,
        project: Project,
        scene_id: str,
        *,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Scene:
        """Regenerate ONE scene's clip, reusing an existing animation prompt.

        The locked source image, storyboard, and Project Bible stay untouched.
        A previous successful clip is preserved in history until the new attempt
        succeeds. If ``prompt`` is omitted, the scene's current video prompt (or
        the latest generation's prompt) is reused.
        """
        scene = self._find_scene(project, scene_id)
        if prompt is None:
            prompt = scene.video_prompt or (
                self.latest_generation(scene).prompt
                if scene.video_generations else None
            )
        if not prompt:
            raise VideoPromptError(
                f"Scene {scene_id} has no video prompt to reuse — use "
                "regenerate-prompt+clip to create one."
            )
        index = project.scenes.index(scene)
        return self.generate_scene(project, scene, index=index, model=model, prompt=prompt)

    def regenerate_prompt_clip(
        self,
        project: Project,
        scene_id: str,
        *,
        model: Optional[str] = None,
    ) -> Scene:
        """Generate a NEW animation prompt, then regenerate the scene's clip.

        The locked source image, storyboard, and Project Bible are preserved;
        only the animation prompt is regenerated.
        """
        scene = self._find_scene(project, scene_id)
        index = project.scenes.index(scene)
        # Force a fresh prompt by not passing one.
        return self.generate_scene(project, scene, index=index, model=model)

    def retry_failed(
        self,
        project: Project,
        scene_id: str,
        *,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Scene:
        """Retry the most recent failed generation for a scene (same prompt)."""
        scene = self._find_scene(project, scene_id)
        if prompt is None:
            gen = scene.video_generations[-1] if scene.video_generations else None
            prompt = gen.prompt if gen else (scene.video_prompt or None)
        # Fall back to reuse only from a known prompt or regenerate the prompt.
        if not prompt:
            return self.regenerate_prompt_clip(project, scene_id, model=model)
        return self.regenerate_clip(project, scene_id, model=model, prompt=prompt)

    def generation_status(self, project: Project, scene_id: str) -> Optional[dict]:
        scene = self._find_scene(project, scene_id)
        gen = self.latest_generation(scene)
        return gen.model_dump() if gen else None

    def _apply_job_state(
        self,
        generation: VideoGeneration,
        job: AsyncJob,
        options: GenerationOptions,
    ) -> None:
        """Translate the polled provider job into the persisted generation record."""
        if generation.status == "SUCCEEDED":
            return
        status = getattr(job.status, "value", str(job.status))
        if status == AsyncJobStatus.SUCCEEDED.value:
            generation.status = "SUCCEEDED"
            result = self._download(job, options)
            generation.output_path = result.video or ""
            generation.model = generation.model or result.model
            generation.raw["video_uri"] = (job.raw or {}).get("video_uri", "")
            if not generation.output_path:
                generation.status = "FAILED"
                generation.error = "Generation succeeded but the provider returned no clip."
        elif status == AsyncJobStatus.FAILED.value:
            generation.status = "FAILED"
            generation.error = self._sanitize_error(
                job.error or "Video generation failed (provider error)."
            )
        elif status == AsyncJobStatus.CANCELLED.value:
            generation.status = "CANCELLED"
            generation.error = self._sanitize_error(job.error or "Generation cancelled.")
        else:
            # Poll loop timed out without a terminal state.
            generation.status = "FAILED"
            generation.error = (
                f"Video generation timed out after {options.params.get('poll_timeout')}s. "
                f"Job {generation.job_id or ''} may still be running; retry to re-check."
            )
        generation.completed_at = _now()

    def _poll(
        self,
        provider,
        job: AsyncJob,
        options: GenerationOptions,
    ) -> AsyncJob:
        poll_timeout = float(
            options.params.get("poll_timeout") or self._settings.video_poll_timeout
        )
        poll_interval = float(
            options.params.get("poll_interval") or self._settings.video_poll_interval
        )
        started = time.monotonic()
        poll_async = getattr(provider, "poll_async", None)
        if poll_async is None:
            raise ProviderError("Provider does not support async polling.")
        while time.monotonic() - started < poll_timeout:
            job = poll_async(job, options=options)
            if job.is_terminal:
                return job
            time.sleep(poll_interval)
        job.status = AsyncJobStatus.FAILED
        job.error = "Video generation timed out waiting for the provider."
        return job

    def _download(self, job: AsyncJob, options: GenerationOptions) -> VideoGenerationResult:
        provider = self._provider()
        download_result = getattr(provider, "download_result", None)
        if download_result is not None:
            return download_result(job, options=options)
        # Providers that return the clip in the job (e.g. the mock) need no download.
        video = (job.raw or {}).get("video") or (job.raw or {}).get("output_path")
        if not video:
            raise ProviderError("Provider returned no video path to store.")
        return VideoGenerationResult(
            provider=getattr(provider, "name", ""),
            model=(job.raw or {}).get("model") or "",
            prompt=(job.raw or {}).get("prompt") or "",
            status=GenerationStatus.SUCCEEDED,
            video=video,
            job_id=job.job_id,
            generation_id=(job.raw or {}).get("generation_id") or job.job_id,
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _locked_image(project: Project, scene: Scene):
        """Return (absolute_path, source_image_id) for the scene's locked image."""
        selected = scene.selected_image
        if not selected:
            raise ValueError(f"Scene {scene.id} has no selected (locked) image")
        path = Path(selected)
        if not path.is_absolute() and "storage/projects" not in selected:
            path = project_root(project.id) / selected
        if not path.is_file():
            raise ValueError(f"Scene {scene.id} locked image not found on disk: {path}")
        return str(path.resolve()), scene.selected_image_id or selected

    @staticmethod
    def _is_complete(scene: Scene) -> bool:
        gen = VideoAgent.current_generation(scene)
        return bool(gen and gen.status == "SUCCEEDED" and gen.output_path)

    @staticmethod
    def _find_scene(project: Project, scene_id: str) -> Scene:
        scene = next((s for s in project.scenes if s.id == scene_id), None)
        if scene is None:
            raise ValueError(f"Scene {scene_id} not found")
        return scene

    @staticmethod
    def current_generation(scene: Scene) -> Optional[VideoGeneration]:
        if not scene.current_video_id:
            return scene.video_generations[-1] if scene.video_generations else None
        for gen in reversed(scene.video_generations):
            if gen.attempt_id == scene.current_video_id:
                return gen
        return scene.video_generations[-1] if scene.video_generations else None

    @staticmethod
    def latest_generation(scene: Scene) -> Optional[VideoGeneration]:
        return scene.video_generations[-1] if scene.video_generations else None

    @staticmethod
    def _sanitize_error(error) -> str:
        message = str(error)
        # Keep error text compact and free of raw provider internals.
        return (message or "Unknown generation error").strip()[:500]

    def _options(self, project: Project, scene: Scene) -> GenerationOptions:
        # Normalized, provider-agnostic request. The concrete adapter translates
        # 9:16 + duration into its own parameters; never hard-code vendor config here.
        duration = max(1, int(scene.duration_seconds or 0))
        return GenerationOptions(
            aspect_ratio=project.settings.aspect_ratio or "9:16",
            duration_seconds=duration,
            params={"output_dir": str(project_asset_dir(project.id, "videos"))},
        )

    def _prompt_for_scene(
        self,
        project: Project,
        scene: Scene,
        stories: list,
        *,
        index: int,
        model: Optional[str],
        source_image_id: str,
    ) -> VideoPromptOutput:
        story_scene = stories[index] if index < len(stories) else None
        previous = stories[index - 1] if index > 0 else None
        following = stories[index + 1] if index + 1 < len(stories) else None
        try:
            raw = self._llm(
                system_prompt=SYSTEM_PROMPT,
                user_message=_build_user_message(
                    idea=project.idea,
                    bible=project.project_bible,
                    story_scene=story_scene,
                    previous_scene=previous,
                    next_scene=following,
                    aspect_ratio=project.settings.aspect_ratio or "9:16",
                    source_image_id=source_image_id,
                    source_image_prompt=scene.image_prompt,
                ),
                json_schema=VIDEO_PROMPT_SCHEMA,
                model=model,
            )
        except ProviderError as e:
            raise VideoPromptError(f"Video prompt provider error: {e}")
        output = VideoPromptOutput.model_validate(raw)
        issues = validate_video_prompt_output(output)
        if issues:
            raise VideoPromptError("; ".join(issues))
        return output

    def _mark_scene_failed(self, project: Project, scene: Scene, error) -> None:
        message = self._sanitize_error(error)
        gen = self.latest_generation(scene)
        if gen is None or gen.status in ("SUCCEEDED",):
            gen = VideoGeneration(
                project_id=project.id,
                scene_id=scene.id,
                status="FAILED",
                error=message,
                completed_at=_now(),
            )
            scene.video_generations.append(gen)
        elif gen.status not in ("SUCCEEDED", "FAILED", "CANCELLED"):
            gen.status = "FAILED"
            gen.error = (gen.error or "") + (f" [{message}]" if message else "")
            gen.completed_at = _now()
        save_project(project)