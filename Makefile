.PHONY: install install-dev test test-cov lint format typecheck docker-up docker-down setup-db seed clean all

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v -m "not integration and not slow"

test-cov:
	pytest tests/ -v --cov=nexus --cov-report=html --cov-report=term -m "not integration and not slow"

test-all:
	pytest tests/ -v --cov=nexus

lint:
	ruff check nexus/ tests/

format:
	black nexus/ tests/
	ruff check --fix nexus/ tests/

typecheck:
	mypy nexus/ --ignore-missing-imports

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f

setup-db:
	python scripts/setup_db.py

seed:
	python scripts/seed_data.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null; true
	rm -rf htmlcov/ .coverage coverage.xml

all: lint typecheck test
