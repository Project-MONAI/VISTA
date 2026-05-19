import os
import shutil
from pathlib import Path
from typing import Optional


def _is_rank_zero() -> bool:
    for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value not in (None, "", "0"):
            return False
    return True


def touch_huggingface_download_counter(
    repo_id: str,
    filename: str,
    revision: str = "main",
    rank_zero_only: bool = True,
) -> Optional[str]:
    """Force a tiny Hugging Face file request without re-downloading model weights."""

    if rank_zero_only and not _is_rank_zero():
        return None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[vista3d] warning: huggingface_hub is not installed; skipping Hugging Face download counter touch.")
        return None

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            force_download=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[vista3d] warning: could not touch Hugging Face download counter for {repo_id}/{filename}: {exc}")
        return None

    print(f"[vista3d] touched Hugging Face download counter for {repo_id}/{filename}")
    return path


def prepare_huggingface_checkpoint(
    repo_id: str,
    checkpoint_filename: str,
    local_checkpoint_path: str,
    counter_filename: str,
    revision: str = "main",
    rank_zero_only: bool = True,
) -> str:
    """Ensure the local VISTA3D checkpoint path exists and touch HF stats for this inference."""

    local_path = Path(local_checkpoint_path)

    if rank_zero_only and not _is_rank_zero():
        return str(local_path)

    touch_huggingface_download_counter(repo_id, counter_filename, revision, rank_zero_only=False)
    if local_path.exists():
        return str(local_path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(f"{local_path} does not exist and huggingface_hub is not installed; cannot download {repo_id}.") from exc

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_filename,
        repo_type="model",
        revision=revision,
    )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        local_path.symlink_to(checkpoint_path)
    except OSError:
        shutil.copy2(checkpoint_path, local_path)

    print(f"[vista3d] prepared checkpoint at {local_path}")
    return str(local_path)
