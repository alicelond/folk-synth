"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_rate():
    """Standard sample rate fixture."""
    return 44100


@pytest.fixture
def valid_frequency():
    """Valid test frequency."""
    return 440.0


@pytest.fixture
def short_duration():
    """Short duration for quick tests."""
    return 0.1


@pytest.fixture
def numpy_seed():
    """Seed for reproducible random tests."""
    return 42


@pytest.fixture(autouse=True)
def reset_numpy_random(numpy_seed):
    """Reset numpy random state before each test."""
    np.random.seed(numpy_seed)
    yield
    # Cleanup if needed
