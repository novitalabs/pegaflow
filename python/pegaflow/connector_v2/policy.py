"""Choices inside what the semantics allow, made once per model.

- Copies: a pure-attention model (full and MLA groups only) keeps one copy
  per slice, as v1 does. A hybrid model (any window or checkpoint group)
  keeps one per CPU socket, so a load reads a copy on its own socket.
- Transfer backend: as v1 ``resolve_transfer_backend``, the kernel for MLA
  models (many small slots, one launch beats one copy per slot) and direct
  copies for everything else; ``pegaflow.transfer_backend`` overrides.
- ``accept`` can only narrow a hit.

Pure Python, so the default unit gate tests it without vLLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import semantics as sem

TRANSFER_BACKENDS = ("direct", "kernel")


@dataclass(frozen=True)
class Policy:
    copy_per_socket: bool = False
    transfer_backend: str = "direct"

    @classmethod
    def of(cls, groups: Sequence[sem.Group], is_mla: bool, override: str | None = None) -> Policy:
        """``is_mla``: v1 ``detect_mla`` (the model config has ``kv_lora_rank``)."""
        backend = "kernel" if is_mla else "direct"
        if override is not None:
            backend = override.strip().lower()
            if backend not in TRANSFER_BACKENDS:
                raise ValueError(
                    f"Unsupported pegaflow.transfer_backend {override!r}; "
                    f"expected one of: {', '.join(TRANSFER_BACKENDS)}"
                )
        hybrid = any(g.kind != sem.FULL for g in groups)
        return cls(copy_per_socket=hybrid, transfer_backend=backend)

    def accept(self, p: int) -> int:
        return p
