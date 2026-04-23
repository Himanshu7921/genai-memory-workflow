from __future__ import annotations


class MemoryError(RuntimeError):
    """Base class for memory subsystem failures."""


class StorageError(MemoryError):
    """Raised when durable storage operations fail."""


class ContradictionResolutionError(MemoryError):
    """Raised when facts cannot be normalized or reconciled."""
