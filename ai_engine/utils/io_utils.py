import hashlib
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_id(*parts: str, length: int = 12) -> str:
    """Deterministic short id from arbitrary string parts (e.g. source path + label).
    Deterministic (vs. uuid4) so re-running the pipeline on the same raw data
    produces the same processed filenames instead of duplicating work."""
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h[:length]


def iter_files_with_extensions(root: Path, extensions: tuple[str, ...]):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path
