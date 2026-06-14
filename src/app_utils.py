
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        Path("/content/drive/MyDrive/Deep/vietnamese-toxic-comment-classification"),
        Path.cwd(),
    ])
    for c in candidates:
        if (c / "data").exists() and (c / "outputs").exists():
            return c.resolve()
    return candidates[0].resolve() if candidates else Path.cwd().resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
