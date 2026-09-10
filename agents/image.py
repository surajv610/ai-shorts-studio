"""Image Agent — generates two candidate images per storyboard scene.

For every scene that does not yet have a locked selected image, the agent:

1. Reads the idea + Project Bible + complete storyboard + current/previous/next
   scene context.
2. Asks the LLM (structured output) for an image prompt that explicitly
   separates GLOBAL CONTINUITY from SCENE-SPECIFIC ACTION, producing two
   complete prompts: candidate A (primary composition) and candidate B
   (meaningfully different composition).
3. Generates TWO candidates through the provider abstraction using the
   project's normalized aspect ratio (9:16 by default). It never hard-codes
   provider-specific dimensions.
4. Persists detailed records for every candidate (project id, scene id,
   candidate id, provider, model, generation id, prompt, storage path, status,
   timestamp, error) and stores stable local file references in the project's
   own asset directory.

The agent is resume-safe: scenes with successful candidates (or a locked
selection) are never regenerated. Regeneration only ever touches the single
requested candidate and preserves the Project Bible, scene, continuity, and any
existing selection.
"""

from datetime import datetime, timezone
from typing import List, Optional

from backend.config import Settings, get_settings
from backend.models import ImageCandidate, Project, Scene
from backend.providers.base import GenerationOptions, ProviderError
from backend.providers.registry import get_image_provider
from backend.storage import project_asset_dir, save_project
from agents.image_schemas import (
    IMAGE_PROMPT_SCHEMA,
    ImagePromptError,
    ImagePromptOutput,
    validate_image_prompt_output,
)
from agents.llm import call_structured

SYSTEM_PROMPT = (
    "You are the Image Agent for an AI Shorts production studio. "
    "You turn a storyboard scene into a precise image-generation prompt for "
    "a vertical (9:16) YouTube Short. For every scene you clearly separate "
    "GLOBAL CONTINUITY (what must remain pixel-identical across the whole "
    "video: subject identity, species/character, environment, soil, camera, "
    "lighting, visual style, color) from SCENE-SPECIFIC ACTION (the one new "
    "physical/bio-logical progression step depicted in THIS scene). "
    "You always return two complete, ready-to-use prompts: prompt_a with the "
    "primary composition and prompt_b with a meaningfully different composition "
    "that still satisfies every continuity constraint. Never introduce props, "
    "text, watermarks, people, or random duplicates. Always return the exact "
    "JSON schema provided with every field populated."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _story_scenes(project: Project) -> list:
    return (project.storyboard or {}).get("scenes") or []


def _successful_candidates(scene: Scene) -> List[ImageCandidate]:
    return [c for c in scene.image_candidates if c.status == "SUCCEEDED"]


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
) -> str:
    msg = (
        f"PROJECT IDEA: {idea}\n"
        f"ASPECT RATIO: {aspect_ratio}\n"
        "Generate prompts for a vertical 9:16 composition.\n\n"
        "PROJECT BIBLE (global continuity):\n"
        f"{_bible_lines(bible)}\n\n"
    )
    msg += _scene_line("CURRENT SCENE", story_scene) + "\n"
    if previous_scene:
        msg += _scene_line("PREVIOUS SCENE (already happened)", previous_scene) + "\n"
    if next_scene:
        msg += _scene_line("NEXT SCENE (will happen after)", next_scene) + "\n"
    msg += (
        "Return structured prompts. prompt_a is the primary composition; "
        "prompt_b is a meaningfully different composition (different camera "
        "angle, distance, or framing) but with identical subject, environment, "
        "lighting, style, and continuity. Each prompt is the full text sent to "
        "an image model."
    )
    return msg


class ImageAgent:
    """Orchestrates candidate image generation for a project's scenes."""

    def __init__(
        self,
        *,
        llm=call_structured,
        image_provider=None,
        settings: Optional[Settings] = None,
    ):
        self._llm = llm
        self._image_provider = image_provider
        self._settings = settings or get_settings()

    # -- provider resolution ---------------------------------------------

    def _provider(self):
        if self._image_provider is not None:
            return self._image_provider
        return get_image_provider(self._settings)

    # -- public flow -----------------------------------------------------

    def run(self, project: Project, *, model: Optional[str] = None) -> dict:
        """Generate candidates for every scene that still needs them.

        Scenes with a locked selection or at least one successful candidate are
        NOT regenerated. Failed scenes are retried. Saves after each scene so a
        restart can resume without repeating work.

        Returns a summary dict: scenes_ready / scenes_generated / scenes_failed.
        """
        summary = {"scenes_ready": 0, "scenes_generated": 0, "scenes_failed": 0}
        for index, scene in enumerate(project.scenes):
            if self._is_locked(scene):
                continue
            if _successful_candidates(scene):
                summary["scenes_ready"] += 1
                continue
            try:
                self.generate_scene(project, scene, index=index, model=model)
            except Exception as e:  # per-scene isolation; never block other scenes
                self._mark_scene_failed(project, scene, error=e)
            save_project(project)
            if _successful_candidates(scene):
                summary["scenes_ready"] += 1
                summary["scenes_generated"] += 1
            else:
                summary["scenes_failed"] += 1
        return summary

    def generate_scene(
        self,
        project: Project,
        scene: Scene,
        *,
        index: int = 0,
        candidates: tuple = ("A", "B"),
        model: Optional[str] = None,
    ) -> Scene:
        """Generate (or regenerate) the requested candidates for one scene."""
        stories = _story_scenes(project)
        story_scene = stories[index] if index < len(stories) else None
        previous = stories[index - 1] if index > 0 else None
        following = stories[index + 1] if index + 1 < len(stories) else None

        prompts = self._prompt_for_scene(
            project,
            scene,
            story_scene,
            previous,
            following,
            model=model,
        )
        if "A" in candidates:
            scene.image_prompt = prompts.prompt_a

        options = self._options(project)
        for candidate_id in candidates:
            prompt = prompts.prompt_a if candidate_id == "A" else prompts.prompt_b
            candidate = self._generate_candidate(project, scene, candidate_id, prompt, options)
            self._upsert_candidate(scene, candidate)
            if candidate_id == "A":
                scene.image_a = candidate.storage_path or None
            else:
                scene.image_b = candidate.storage_path or None
        return scene

    def regenerate_candidate(
        self,
        project: Project,
        scene_id: str,
        candidate_id: str,
        *,
        model: Optional[str] = None,
    ) -> Scene:
        """Regenerate a single candidate (A or B) for a scene.

        The Project Bible, scene, continuity, and any existing selection are
        preserved; only the requested candidate is replaced.
        """
        if candidate_id not in ("A", "B"):
            raise ImagePromptError(f"candidate_id must be 'A' or 'B', got {candidate_id!r}")
        scene = next((s for s in project.scenes if s.id == scene_id), None)
        if scene is None:
            raise ValueError(f"Scene {scene_id} not found")
        index = project.scenes.index(scene)
        self.generate_scene(
            project,
            scene,
            index=index,
            candidates=(candidate_id,),
            model=model,
        )
        save_project(project)
        return scene

    # -- internals --------------------------------------------------------

    def _options(self, project: Project) -> GenerationOptions:
        # Normalized, provider-agnostic request. The concrete adapter translates
        # 9:16 into its own parameters; never hard-code provider dims here.
        return GenerationOptions(
            aspect_ratio=project.settings.aspect_ratio or "9:16",
            params={"output_dir": str(project_asset_dir(project.id, "images"))},
        )

    def _prompt_for_scene(
        self,
        project: Project,
        scene: Scene,
        story_scene: Optional[dict],
        previous_scene: Optional[dict],
        next_scene: Optional[dict],
        *,
        model: Optional[str],
    ) -> ImagePromptOutput:
        try:
            raw = self._llm(
                system_prompt=SYSTEM_PROMPT,
                user_message=_build_user_message(
                    idea=project.idea,
                    bible=project.project_bible,
                    story_scene=story_scene,
                    previous_scene=previous_scene,
                    next_scene=next_scene,
                    aspect_ratio=project.settings.aspect_ratio or "9:16",
                ),
                json_schema=IMAGE_PROMPT_SCHEMA,
                model=model,
            )
        except ProviderError as e:
            raise ImagePromptError(f"Image prompt provider error: {e}")
        output = ImagePromptOutput.model_validate(raw)
        issues = validate_image_prompt_output(output)
        if issues:
            raise ImagePromptError("; ".join(issues))
        return output

    def _generate_candidate(
        self,
        project: Project,
        scene: Scene,
        candidate_id: str,
        prompt: str,
        options: GenerationOptions,
    ) -> ImageCandidate:
        provider = self._provider()
        provider_name = getattr(provider, "name", "")
        try:
            result = provider.generate_images(prompt, options=options, count=1)
            path = (result.images or [])[0] if result.images else ""
            if not path:
                raise ImagePromptError("provider returned no image path")
            return ImageCandidate(
                candidate_id=candidate_id,
                project_id=project.id,
                scene_id=scene.id,
                provider=result.provider,
                model=result.model,
                generation_id=result.generation_id,
                prompt=prompt,
                storage_path=path,
                status="SUCCEEDED",
                created_at=_now(),
            )
        except ProviderError as e:
            # Auth/quota/safety/invalid/unavailable/generation errors all map to
            # a FAILED candidate record so the UI can offer a retry.
            return ImageCandidate(
                candidate_id=candidate_id,
                project_id=project.id,
                scene_id=scene.id,
                provider=provider_name,
                prompt=prompt,
                status="FAILED",
                error=str(e),
                created_at=_now(),
            )

    @staticmethod
    def _upsert_candidate(scene: Scene, candidate: ImageCandidate) -> None:
        scene.image_candidates = [
            c for c in scene.image_candidates if c.candidate_id != candidate.candidate_id
        ] + [candidate]

    @staticmethod
    def _mark_scene_failed(project: Project, scene: Scene, error) -> None:
        message = str(error)
        for candidate_id in ("A", "B"):
            existing = next(
                (c for c in scene.image_candidates if c.candidate_id == candidate_id),
                None,
            )
            if existing and existing.status == "SUCCEEDED":
                continue
            scene.image_candidates = [
                c for c in scene.image_candidates if c.candidate_id != candidate_id
            ]
            scene.image_candidates.append(
                ImageCandidate(
                    candidate_id=candidate_id,
                    project_id=project.id,
                    scene_id=scene.id,
                    status="FAILED",
                    error=message,
                    created_at=_now(),
                )
            )

    @staticmethod
    def _is_locked(scene: Scene) -> bool:
        return bool(scene.selected_image_id or scene.selected_image)