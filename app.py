"""Root entry point for HuggingFace Spaces (Streamlit SDK).

This thin wrapper sets demo mode, then launches the NEXUS dashboard.
It is also the file referenced by the HF Space YAML header in README.md.
"""
from __future__ import annotations

import os

# Force demo mode so the dashboard renders without real API keys / databases
os.environ.setdefault("USE_DEMO_DATA", "true")
os.environ.setdefault("NEXUS_ENV", "production")

from nexus.ui.dashboard.app import main  # noqa: E402

if __name__ == "__main__":
    main()
