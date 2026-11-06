"""Compatibility API for the preserved Platform allocation.

The common engine also supports Brand and Sub-Segment. Platform keeps its
original configuration, exhaustive small-population search and output schema.
Platform is excluded from MiniLM training; this module only allocates records.
"""
import pandas as pd  # Retain the original file-access testing seam.
from scipy.optimize import milp

from .features import sha256
from .target_development_allocation import (
    TargetAllocation as PlatformAllocation,
    TargetAllocationConfig as PlatformAllocationConfig,
    allocate_target_development,
    write_target_proposal,
)
from .target_development_preparation import PreparedTargetDevelopment

VERSION = "platform-development-allocation-v1"


def allocate_platform_development(prepared, config=None):
    """Preserve the Platform-only public API and deterministic allocation."""
    if not isinstance(prepared, PreparedTargetDevelopment) or prepared.target_column != "Platform":
        raise ValueError("This allocator supports prepared Platform development only")
    return allocate_target_development(prepared, config, _solver=milp)


def write_platform_proposal(split_directory, output_directory, config=None):
    """Write a new Platform proposal through the shared allocation engine."""
    return write_target_proposal(split_directory, output_directory, "Platform", config,
                                 _allocator=allocate_platform_development)
