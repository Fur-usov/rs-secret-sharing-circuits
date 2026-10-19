"""Reproducible random source, recreated independently for each test."""

from random import Random

import pytest


@pytest.fixture
def rng():
    return Random(20261019)
