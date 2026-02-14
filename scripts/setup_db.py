from __future__ import annotations

import sys

from nexus.core.config import get_config
from nexus.core.logging import get_logger, setup_logging
from nexus.data.storage.postgres import PostgresStorage

logger = get_logger("scripts.setup_db")


def main() -> None:
    setup_logging()
    config = get_config()

    logger.info(f"Setting up database at {config.storage.postgres.host}:{config.storage.postgres.port}")
    logger.info(f"Database: {config.storage.postgres.database}")

    try:
        storage = PostgresStorage()
        storage.create_tables()
        logger.info("All database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to set up database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
