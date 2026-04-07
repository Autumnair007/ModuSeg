"""Random seed utilities for reproducibility."""
import os
import random

_SEED_SET = False


def set_global_seed(seed: int, verbose: bool = True) -> None:
    """Set global random seed for Python, NumPy, PyTorch, and FAISS."""
    global _SEED_SET
    
    if verbose and not _SEED_SET:
        print(f"[Seed] Setting global seed: {seed}")
    
    # Python random
    random.seed(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # cuDNN deterministic settings
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass

        # Strict deterministic mode (may increase VRAM usage)
        strict_deterministic = os.environ.get("OVERRIDE_STRICT_DETERMINISTIC", "false").lower() == "true"

        if strict_deterministic:
            if verbose:
                print("[Seed] Warning: strict deterministic mode enabled, may increase VRAM usage")
            
            # CUBLAS deterministic
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

            # Disable TF32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass

            # Require deterministic algorithms
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass

    # FAISS seed
    try:
        import faiss
        # faiss API varies across versions
        if hasattr(faiss, "set_random_seed"):
            faiss.set_random_seed(seed)
        elif hasattr(faiss, "rand") and hasattr(faiss.rand, "seed"):
            faiss.rand.seed(seed)
        elif hasattr(faiss, "cvar") and hasattr(faiss.cvar, "rand_seed"):
            faiss.cvar.rand_seed = int(seed)
    except ImportError:
        pass
    
    _SEED_SET = True


def seed_worker(worker_id: int) -> None:
    """Set deterministic seed for DataLoader worker processes."""
    try:
        import torch
        worker_seed = torch.initial_seed() % 2**32
    except Exception:
        worker_seed = (os.getpid() + worker_id) % 2**32

    random.seed(worker_seed)
    
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass


__all__ = [
    'set_global_seed',
    'seed_worker',
]
