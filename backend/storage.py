import json
import os
from pathlib import Path
from typing import Optional

from backend.models import Project

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage" / "projects"

# Asset subdirs created per project under storage/projects/<project_id>/.
ASSET_DIRS = (
    "storyboard",
    "images",
    "videos",
    "final",
    "metadata",
    "qc",
)


def _project_path(project_id: str) -> Path:
    return STORAGE_DIR / f"{project_id}.json"


def project_root(project_id: str) -> Path:
    """The per-project asset directory (storage/projects/<id>/)."""
    return STORAGE_DIR / project_id


def ensure_project_dirs(project_id: str) -> Path:
    """Create the nested asset directory tree for a project and return its root."""
    root = project_root(project_id)
    for name in ASSET_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def project_asset_dir(project_id: str, kind: str) -> Path:
    """Return (creating if needed) the directory for a given asset kind."""
    if kind not in ASSET_DIRS:
        raise ValueError(f"Unknown asset kind {kind!r}")
    path = project_root(project_id) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_project(project: Project) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = _project_path(project.id)
    # Idempotently set up the asset directory tree for this project.
    ensure_project_dirs(project.id)
    path.write_text(project.model_dump_json(indent=2))


def load_project(project_id: str) -> Optional[Project]:
    path = _project_path(project_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return Project.model_validate(data)


def list_projects() -> list[dict]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    projects = []
    for f in sorted(STORAGE_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        projects.append({
            "id": data["id"],
            "name": data.get("name", ""),
            "idea": data["idea"],
            "status": data["status"],
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "settings": data.get("settings", {}),
            "storyboard": data.get("storyboard"),
        })
    return projects


def delete_project(project_id: str) -> bool:
    path = _project_path(project_id)
    deleted = False
    if path.exists():
        path.unlink()
        deleted = True
    # Remove the per-project asset directory tree if it exists.
    root = project_root(project_id)
    if root.is_dir():
        import shutil

        shutil.rmtree(root, ignore_errors=True)
        deleted = True
    return deleted
