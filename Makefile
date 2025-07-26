.PHONY: help install dev test clean docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  install     Install production dependencies"
	@echo "  dev         Install development dependencies"
	@echo "  test        Run tests with coverage"
	@echo "  lint        Run code quality checks"
	@echo "  format      Format code with black and isort"
	@echo "  docker-up   Start Docker services"
	@echo "  docker-down Stop Docker services"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest backend/tests/ --cov=backend/app --cov-report=html

lint:
	black --check backend/ frontend/
	isort --check backend/ frontend/
	flake8 backend/ frontend/
	mypy backend/app/

format:
	black backend/ frontend/
	isort backend/ frontend/

docker-up:
	docker-compose up -d
	python scripts/pull_models.py
	python scripts/init_databases.py

docker-down:
	docker-compose down