"""Entry point for `python -m nexus`."""
from nexus.core.logging import setup_logging

setup_logging()

from nexus.cli.main import main  # noqa: E402

main()
