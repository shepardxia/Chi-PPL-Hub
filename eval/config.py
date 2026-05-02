"""Shared eval configuration constants and helpers."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_TIMEOUT = 60
DEFAULT_SEED = 42
DEFAULT_N_MC = 200
DEFAULT_MC_WORKERS = 8


@dataclass(frozen=True)
class EvalConfig:
    timeout: int = DEFAULT_TIMEOUT
    seed: int = DEFAULT_SEED
    n_mc: int = DEFAULT_N_MC
    mc_workers: int = DEFAULT_MC_WORKERS
