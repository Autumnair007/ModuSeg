#!/usr/bin/env python3
"""
Launch the full ModuSeg pipeline (build + inference) inside a tmux session.

Usage:
    python tools/run_pipeline.py voc              # VOC dataset, CUDA device 1
    python tools/run_pipeline.py coco             # COCO dataset, CUDA device 1
    python tools/run_pipeline.py ade20k           # ADE20K dataset, CUDA device 1
    python tools/run_pipeline.py cityscapes       # CityScapes dataset, CUDA device 1
    python tools/run_pipeline.py voc --cuda 0     # VOC dataset, CUDA device 0
    python tools/run_pipeline.py coco --cuda 3    # COCO dataset, CUDA device 3

The full stdout/stderr of both scripts is tee-d into:
    feature_bank_voc/logs/run_<timestamp>.log     (VOC)
    feature_bank_coco/logs/run_<timestamp>.log    (COCO)
    feature_bank_ade20k/logs/run_<timestamp>.log  (ADE20K)
    feature_bank_cityscapes/logs/run_<timestamp>.log  (CityScapes)
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default CUDA device per dataset type
_DEFAULT_CUDA = {"voc": "1", "coco": "1", "ade20k": "1", "cityscapes": "1"}

# Output root directories (must match configs/config.py defaults)
_FEATURE_BANK_ROOT = {
    "voc": "feature_bank_voc",
    "coco": "feature_bank_coco",
    "ade20k": "feature_bank_ade20k",
    "cityscapes": "feature_bank_cityscapes",
}

# tmux session name template
_SESSION_NAME = {
    "voc": "moduseg_voc",
    "coco": "moduseg_coco",
    "ade20k": "moduseg_ade20k",
    "cityscapes": "moduseg_cityscapes",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_conda_base() -> str:
    """Return the conda base prefix path."""
    result = subprocess.run(
        ["conda", "info", "--base"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def tmux_session_exists(session_name: str) -> bool:
    """Return True if a tmux session with *session_name* already exists."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
    )
    return result.returncode == 0


def kill_existing_session(session_name: str) -> None:
    """Kill an existing tmux session."""
    subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
    print(f"[tmux] Killed existing session '{session_name}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ModuSeg build + inference pipeline inside a tmux session."
    )
    parser.add_argument(
        "dataset",
        choices=["voc", "coco", "ade20k", "cityscapes"],
        help="Dataset type: 'voc', 'coco', 'ade20k', or 'cityscapes'",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=None,
        help="CUDA_VISIBLE_DEVICES value (default: 1)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Override the tmux session name",
    )
    parser.add_argument(
        "--kill-existing",
        action="store_true",
        help="Kill an existing tmux session with the same name before launching",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset: str = args.dataset

    # Resolve CUDA device
    cuda_device: str = str(args.cuda) if args.cuda is not None else _DEFAULT_CUDA[dataset]

    # Paths
    workspace_root = Path(__file__).resolve().parent.parent
    fb_root_name: str = _FEATURE_BANK_ROOT[dataset]
    log_dir: Path = workspace_root / fb_root_name / "logs"
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file: Path = log_dir / f"run_{timestamp}.log"

    # tmux session name
    session_name: str = args.session if args.session else _SESSION_NAME[dataset]

    # Handle existing session
    if tmux_session_exists(session_name):
        if args.kill_existing:
            kill_existing_session(session_name)
        else:
            print(f"[Error] tmux session '{session_name}' already exists.")
            print(f"  To kill it : tmux kill-session -t {session_name}")
            print(f"  Or re-run  : python tools/run_pipeline.py {dataset} --kill-existing")
            sys.exit(1)

    # Build conda init command (works in non-interactive shells)
    try:
        conda_base = get_conda_base()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[Error] Cannot determine conda base: {e}")
        sys.exit(1)
    conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")

    # Write a standalone bash script to avoid all quoting/escaping issues
    # when passing commands to tmux.  The script is placed next to the log file.
    log_dir.mkdir(parents=True, exist_ok=True)
    run_sh: Path = log_dir / f"run_{timestamp}.sh"

    sep = "=" * 60
    script_lines = [
        "#!/usr/bin/env bash",
        "set -eo pipefail",
        "",
        f"# Auto-generated by tools/run_pipeline.py  ({timestamp})",
        f"# dataset={dataset}  cuda={cuda_device}",
        "",
        f"LOG={log_file}",
        "",
        f"cd {workspace_root}",
        "",
        "# Initialize conda for non-interactive shell",
        f"source {conda_sh}",
        "conda activate CorrCLIP",
        "",
        "# ----------------------------------------------------------------",
        "# BUILD STAGE",
        "# ----------------------------------------------------------------",
        f'echo "{sep}" | tee -a "$LOG"',
        f'echo "[Pipeline] BUILD STAGE | dataset={dataset} cuda={cuda_device} start=$(date)" | tee -a "$LOG"',
        f'echo "{sep}" | tee -a "$LOG"',
        f"CUDA_VISIBLE_DEVICES={cuda_device} OVERRIDE_DATASET_TYPE={dataset} \\",
        f"    python run_build_feature_bank.py 2>&1 | tee -a \"$LOG\"",
        "",
        "# ----------------------------------------------------------------",
        "# INFERENCE STAGE",
        "# ----------------------------------------------------------------",
        f'echo "{sep}" | tee -a "$LOG"',
        f'echo "[Pipeline] INFERENCE STAGE | dataset={dataset} cuda={cuda_device} start=$(date)" | tee -a "$LOG"',
        f'echo "{sep}" | tee -a "$LOG"',
        f"CUDA_VISIBLE_DEVICES={cuda_device} OVERRIDE_DATASET_TYPE={dataset} \\",
        f"    python run_inference.py 2>&1 | tee -a \"$LOG\"",
        "",
        f'echo "[Pipeline] ALL DONE | exit_code=$? | end=$(date)" | tee -a "$LOG"',
    ]

    run_sh.write_text("\n".join(script_lines) + "\n")
    run_sh.chmod(0o755)

    # tmux: create detached session that runs the script; exec bash keeps the
    # pane open so the user can inspect output after the pipeline finishes.
    tmux_launch = [
        "tmux", "new-session", "-d", "-s", session_name,
        f"bash {run_sh} ; exec bash",
    ]

    # Print launch summary
    print("=" * 60)
    print(f"[Launch] Dataset       : {dataset.upper()}")
    print(f"[Launch] CUDA device   : {cuda_device}")
    print(f"[Launch] Feature bank  : {fb_root_name}/")
    print(f"[Launch] Run script    : {run_sh}")
    print(f"[Launch] Log file      : {log_file}")
    print(f"[Launch] tmux session  : {session_name}")
    print("=" * 60)

    try:
        subprocess.run(tmux_launch, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to start tmux session: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("[Error] 'tmux' not found. Please install tmux.")
        sys.exit(1)

    print(f"\n[OK] Pipeline launched in tmux session '{session_name}'.")
    print(f"  Attach to session : tmux attach -t {session_name}")
    print(f"  Follow log        : tail -f {log_file}")
    print(f"  List sessions     : tmux ls")


if __name__ == "__main__":
    main()
