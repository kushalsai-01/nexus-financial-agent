from __future__ import annotations

import os

import pytest

os.environ["NEXUS_ENV"] = "test"
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
