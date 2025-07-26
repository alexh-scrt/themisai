# =============================================================================
# Legal AI Chatbot - Development Automation Makefile
# =============================================================================
#
# This Makefile provides comprehensive automation for the Legal AI Chatbot
# development workflow including environment setup, testing, deployment,
# and maintenance tasks.
#
# Requirements:
# - Python 3.13+
# - Docker & Docker Compose v2
# - NVIDIA Container Toolkit (for GPU support)
# - make utility
#
# Quick Start:
#   make help          # Show available commands
#   make setup         # Complete environment setup
#   make dev           # Start development environment
#   make test          # Run comprehensive tests
#
# =============================================================================

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Project configuration
PROJECT_NAME := patexia-legal-ai
PYTHON_VERSION := 3.13
DOCKER_COMPOSE := docker-compose

# Environment configuration
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Development settings
BACKEND_DIR := backend
FRONTEND_DIR := frontend
SCRIPTS_DIR := scripts
DOCS_DIR := docs
CONFIG_DIR := config
DATA_DIR := data
LOGS_DIR := logs

# Testing configuration
TEST_DIR := $(BACKEND_DIR)/tests
COVERAGE_DIR := htmlcov
COVERAGE_THRESHOLD := 80

# Docker configuration
DOCKER_NETWORK := legal_ai_network
COMPOSE_FILE := docker-compose.yml
COMPOSE_OVERRIDE := docker-compose.override.yml

# Service URLs (for health checks)
BACKEND_URL := http://localhost:8000
FRONTEND_URL := http://localhost:7860
WEAVIATE_URL := http://localhost:8080
MONGODB_URL := mongodb://localhost:27017
OLLAMA_URL := http://localhost:11434

# Color codes for output formatting
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m  # No Color

# =============================================================================
# PHONY TARGETS (targets that don't create files)
# =============================================================================

.PHONY: help setup clean install install-dev venv
.PHONY: format lint type-check test test-unit test-integration test-performance
.PHONY: docker-build docker-up docker-down docker-restart docker-logs docker-clean
.PHONY: models-pull models-status databases-init databases-backup databases-restore
.PHONY: dev-start dev-stop dev-restart dev-logs dev-shell
.PHONY: security-scan dependency-check vulnerability-check
.PHONY: docs-build docs-serve docs-clean
.PHONY: backup restore performance-test
.PHONY: validate-env health-check status
.PHONY: quick-setup full-setup production-setup

# =============================================================================
# DEFAULT TARGET
# =============================================================================

# Default target - show help
all: help

# =============================================================================
# HELP AND INFORMATION
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)Legal AI Chatbot - Development Automation$(NC)"
	@echo "=============================================="
	@echo ""
	@echo "$(GREEN)Quick Start Commands:$(NC)"
	@echo "  make setup         Complete environment setup"
	@echo "  make dev           Start development environment"
	@echo "  make test          Run comprehensive tests"
	@echo "  make clean         Clean all generated files"
	@echo ""
	@echo "$(GREEN)Available Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { \
		printf "  $(BLUE)%-18s$(NC) %s\n", $$1, $$2 \
	}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Service URLs:$(NC)"
	@echo "  Frontend (Gradio): $(FRONTEND_URL)"
	@echo "  Backend (FastAPI): $(BACKEND_URL)"
	@echo "  API Documentation: $(BACKEND_URL)/docs"
	@echo "  Weaviate Admin:    $(WEAVIATE_URL)"
	@echo ""
	@echo "$(GREEN)Usage Examples:$(NC)"
	@echo "  make setup && make dev     # Full setup and start development"
	@echo "  make test-unit             # Run only unit tests"
	@echo "  make docker-logs backend   # View backend service logs"
	@echo "  make performance-test      # Run performance benchmarks"

status: ## Show system and service status
	@echo "$(BLUE)System Status$(NC)"
	@echo "=============="
	@echo "Python Version: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@echo "Docker Version: $$(docker --version 2>/dev/null || echo 'Not found')"
	@echo "Docker Compose: $$(docker-compose --version 2>/dev/null || echo 'Not found')"
	@echo "NVIDIA GPU:     $$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not detected')"
	@echo ""
	@echo "$(BLUE)Service Status$(NC)"
	@echo "==============="
	@$(DOCKER_COMPOSE) ps 2>/dev/null || echo "Docker services not running"
	@echo ""
	@echo "$(BLUE)Environment Status$(NC)"
	@echo "==================="
	@echo "Virtual Environment: $$(if [ -d $(VENV_DIR) ]; then echo 'Created'; else echo 'Not found'; fi)"
	@echo "Dependencies: $$(if [ -f $(VENV_DIR)/pyvenv.cfg ]; then echo 'Installed'; else echo 'Not installed'; fi)"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup: validate-env venv install-dev docker-build models-pull databases-init ## Complete environment setup
	@echo "$(GREEN)✓ Environment setup completed successfully!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  make dev          # Start development environment"
	@echo "  make test         # Run tests to verify setup"
	@echo "  make health-check # Verify all services are healthy"

quick-setup: venv install docker-up ## Quick setup for development (no models/databases)
	@echo "$(GREEN)✓ Quick setup completed!$(NC)"
	@echo "$(YELLOW)Note: Run 'make models-pull' and 'make databases-init' for full functionality$(NC)"

full-setup: clean setup test ## Complete setup with validation
	@echo "$(GREEN)✓ Full setup with validation completed!$(NC)"

production-setup: validate-env install docker-build ## Production-ready setup
	@echo "$(GREEN)✓ Production setup completed!$(NC)"
	@echo "$(YELLOW)Remember to configure production environment variables$(NC)"

validate-env: ## Validate system requirements
	@echo "$(BLUE)Validating system requirements...$(NC)"
	@command -v python3 >/dev/null 2>&1 || { echo "$(RED)Error: Python 3 not found$(NC)"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)Error: Docker not found$(NC)"; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "$(RED)Error: Docker Compose not found$(NC)"; exit 1; }
	@python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)" || { echo "$(RED)Error: Python 3.13+ required$(NC)"; exit 1; }
	@echo "$(GREEN)✓ System requirements validated$(NC)"

# =============================================================================
# PYTHON ENVIRONMENT MANAGEMENT
# =============================================================================

venv: $(VENV_DIR)/bin/activate ## Create virtual environment

$(VENV_DIR)/bin/activate:
	@echo "$(BLUE)Creating Python virtual environment...$(NC)"
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	@echo "$(GREEN)✓ Virtual environment created$(NC)"

install: venv ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

install-dev: venv ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@if [ -f .pre-commit-config.yaml ]; then \
		$(VENV_DIR)/bin/pre-commit install; \
		echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"; \
	fi
	@echo "$(GREEN)✓ Development environment ready$(NC)"

upgrade-deps: venv ## Upgrade all dependencies
	@echo "$(BLUE)Upgrading dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt
	@echo "$(GREEN)✓ Dependencies upgraded$(NC)"

# =============================================================================
# CODE QUALITY AND FORMATTING
# =============================================================================

format: venv ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(VENV_DIR)/bin/black $(BACKEND_DIR)/ $(SCRIPTS_DIR)/
	$(VENV_DIR)/bin/isort $(BACKEND_DIR)/ $(SCRIPTS_DIR)/
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: venv ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(VENV_DIR)/bin/black --check $(BACKEND_DIR)/ $(SCRIPTS_DIR)/ || (echo "$(YELLOW)Run 'make format' to fix formatting$(NC)" && false)
	$(VENV_DIR)/bin/isort --check $(BACKEND_DIR)/ $(SCRIPTS_DIR)/ || (echo "$(YELLOW)Run 'make format' to fix imports$(NC)" && false)
	$(VENV_DIR)/bin/flake8 $(BACKEND_DIR)/ $(SCRIPTS_DIR)/
	@echo "$(GREEN)✓ Linting checks passed$(NC)"

type-check: venv ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	$(VENV_DIR)/bin/mypy $(BACKEND_DIR)/app/
	@echo "$(GREEN)✓ Type checks passed$(NC)"

quality-check: lint type-check ## Run all code quality checks
	@echo "$(GREEN)✓ All code quality checks passed$(NC)"

# =============================================================================
# TESTING
# =============================================================================

test: test-unit test-integration ## Run all tests
	@echo "$(GREEN)✓ All tests completed$(NC)"

test-unit: venv ## Run unit tests with coverage
	@echo "$(BLUE)Running unit tests...$(NC)"
	@mkdir -p $(COVERAGE_DIR)
	$(VENV_DIR)/bin/pytest $(TEST_DIR)/unit/ \
		--cov=$(BACKEND_DIR)/app \
		--cov-report=html:$(COVERAGE_DIR) \
		--cov-report=term \
		--cov-fail-under=$(COVERAGE_THRESHOLD) \
		-v
	@echo "$(GREEN)✓ Unit tests completed$(NC)"
	@echo "Coverage report: $(COVERAGE_DIR)/index.html"

test-integration: venv docker-up-test ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(VENV_DIR)/bin/pytest $(TEST_DIR)/integration/ -v
	@echo "$(GREEN)✓ Integration tests completed$(NC)"

test-api: venv ## Run API tests
	@echo "$(BLUE)Running API tests...$(NC)"
	$(VENV_DIR)/bin/pytest $(TEST_DIR)/api/ -v
	@echo "$(GREEN)✓ API tests completed$(NC)"

test-performance: venv docker-up ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/performance_test.py --verbose
	@echo "$(GREEN)✓ Performance tests completed$(NC)"

test-clean: ## Clean test artifacts
	@echo "$(BLUE)Cleaning test artifacts...$(NC)"
	rm -rf $(COVERAGE_DIR)
	rm -rf .pytest_cache
	rm -f .coverage
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "$(GREEN)✓ Test artifacts cleaned$(NC)"

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-up: ## Start all Docker services
	@echo "$(BLUE)Starting Docker services...$(NC)"
	@mkdir -p $(DATA_DIR)/{mongodb,weaviate,ollama,uploads,backups}
	@mkdir -p $(LOGS_DIR)
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Docker services started$(NC)"
	@echo "Waiting for services to be ready..."
	@sleep 10
	@$(MAKE) health-check

docker-up-test: ## Start Docker services for testing
	@echo "$(BLUE)Starting test Docker services...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f docker-compose.test.yml up -d
	@sleep 5

docker-down: ## Stop all Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Docker services stopped$(NC)"

docker-restart: docker-down docker-up ## Restart all Docker services

docker-logs: ## Show logs from all services (or specific service: make docker-logs SERVICE=backend)
	@if [ -n "$(SERVICE)" ]; then \
		echo "$(BLUE)Showing logs for $(SERVICE)...$(NC)"; \
		$(DOCKER_COMPOSE) logs -f $(SERVICE); \
	else \
		echo "$(BLUE)Showing logs for all services...$(NC)"; \
		$(DOCKER_COMPOSE) logs -f; \
	fi

docker-shell: ## Open shell in backend container
	@echo "$(BLUE)Opening shell in backend container...$(NC)"
	$(DOCKER_COMPOSE) exec backend /bin/bash

docker-clean: docker-down ## Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker system prune -f
	docker volume prune -f
	@echo "$(GREEN)✓ Docker resources cleaned$(NC)"

docker-clean-all: docker-down ## Clean all Docker resources including volumes
	@echo "$(YELLOW)WARNING: This will delete all data volumes!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		$(DOCKER_COMPOSE) down -v; \
		docker system prune -a -f; \
		echo "$(GREEN)✓ All Docker resources cleaned$(NC)"; \
	else \
		echo ""; \
		echo "$(BLUE)Operation cancelled$(NC)"; \
	fi

# =============================================================================
# AI MODELS MANAGEMENT
# =============================================================================

models-pull: venv docker-up ## Pull required AI models
	@echo "$(BLUE)Pulling AI models...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/pull_models.py --verbose
	@echo "$(GREEN)✓ AI models pulled$(NC)"

models-status: venv ## Check AI models status
	@echo "$(BLUE)Checking AI models status...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/pull_models.py --check-only

models-clean: ## Clean AI model cache
	@echo "$(BLUE)Cleaning AI model cache...$(NC)"
	docker exec legal_ai_ollama ollama rm --all 2>/dev/null || true
	@echo "$(GREEN)✓ AI model cache cleaned$(NC)"

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

databases-init: venv docker-up ## Initialize databases
	@echo "$(BLUE)Initializing databases...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/init_databases.py --verbose
	@echo "$(GREEN)✓ Databases initialized$(NC)"

databases-backup: venv ## Backup databases
	@echo "$(BLUE)Backing up databases...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/backup_data.py --verbose
	@echo "$(GREEN)✓ Databases backed up$(NC)"

databases-restore: venv ## Restore databases from backup
	@echo "$(BLUE)Restoring databases...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/backup_data.py --restore --verbose
	@echo "$(GREEN)✓ Databases restored$(NC)"

databases-reset: ## Reset all databases (destructive)
	@echo "$(YELLOW)WARNING: This will delete all database data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		$(DOCKER_COMPOSE) down; \
		docker volume rm legal_ai_mongodb_data legal_ai_weaviate_data 2>/dev/null || true; \
		$(MAKE) docker-up databases-init; \
		echo "$(GREEN)✓ Databases reset$(NC)"; \
	else \
		echo ""; \
		echo "$(BLUE)Operation cancelled$(NC)"; \
	fi

# =============================================================================
# DEVELOPMENT WORKFLOW
# =============================================================================

dev: dev-start ## Start development environment

dev-start: docker-up models-pull databases-init ## Start complete development environment
	@echo "$(GREEN)✓ Development environment started$(NC)"
	@echo ""
	@echo "$(BLUE)Available Services:$(NC)"
	@echo "  Frontend: $(FRONTEND_URL)"
	@echo "  Backend:  $(BACKEND_URL)"
	@echo "  API Docs: $(BACKEND_URL)/docs"
	@echo ""
	@echo "$(BLUE)Useful Commands:$(NC)"
	@echo "  make dev-logs     # View development logs"
	@echo "  make dev-shell    # Open development shell"
	@echo "  make health-check # Check service health"

dev-stop: docker-down ## Stop development environment

dev-restart: dev-stop dev-start ## Restart development environment

dev-logs: docker-logs ## Show development logs

dev-shell: docker-shell ## Open development shell

dev-reset: docker-clean-all quick-setup ## Reset development environment

# =============================================================================
# HEALTH CHECKS AND MONITORING
# =============================================================================

health-check: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@echo -n "Backend:  "; curl -s -o /dev/null -w "%{http_code}" $(BACKEND_URL)/health && echo " $(GREEN)✓$(NC)" || echo " $(RED)✗$(NC)"
	@echo -n "Frontend: "; curl -s -o /dev/null -w "%{http_code}" $(FRONTEND_URL) && echo " $(GREEN)✓$(NC)" || echo " $(RED)✗$(NC)"
	@echo -n "Weaviate: "; curl -s -o /dev/null -w "%{http_code}" $(WEAVIATE_URL)/v1/.well-known/ready && echo " $(GREEN)✓$(NC)" || echo " $(RED)✗$(NC)"
	@echo -n "Ollama:   "; curl -s -o /dev/null -w "%{http_code}" $(OLLAMA_URL)/api/version && echo " $(GREEN)✓$(NC)" || echo " $(RED)✗$(NC)"

monitor: ## Monitor system resources (requires htop)
	@command -v htop >/dev/null 2>&1 && htop || (echo "$(YELLOW)htop not found, using top$(NC)" && top)

gpu-status: ## Show GPU status
	@nvidia-smi 2>/dev/null || echo "$(YELLOW)NVIDIA GPU not detected$(NC)"

# =============================================================================
# SECURITY AND MAINTENANCE
# =============================================================================

security-scan: venv ## Run security scans
	@echo "$(BLUE)Running security scans...$(NC)"
	$(VENV_DIR)/bin/safety check --json || true
	$(VENV_DIR)/bin/bandit -r $(BACKEND_DIR)/ -f json || true
	@echo "$(GREEN)✓ Security scans completed$(NC)"

dependency-check: venv ## Check for dependency vulnerabilities
	@echo "$(BLUE)Checking dependencies...$(NC)"
	$(VENV_DIR)/bin/pip-audit
	@echo "$(GREEN)✓ Dependency check completed$(NC)"

vulnerability-check: security-scan dependency-check ## Run all vulnerability checks

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs-build: venv ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		$(VENV_DIR)/bin/mkdocs build; \
		echo "$(GREEN)✓ Documentation built$(NC)"; \
	else \
		echo "$(YELLOW)Documentation directory not found$(NC)"; \
	fi

docs-serve: venv ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8001$(NC)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		$(VENV_DIR)/bin/mkdocs serve -a localhost:8001; \
	else \
		echo "$(YELLOW)Documentation directory not found$(NC)"; \
	fi

docs-clean: ## Clean documentation artifacts
	@echo "$(BLUE)Cleaning documentation...$(NC)"
	rm -rf site/
	@echo "$(GREEN)✓ Documentation cleaned$(NC)"

# =============================================================================
# BACKUP AND RESTORE
# =============================================================================

backup: databases-backup ## Create full system backup
	@echo "$(BLUE)Creating full system backup...$(NC)"
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@echo "$(GREEN)✓ Full backup completed$(NC)"

restore: ## Restore from backup (interactive)
	@echo "$(BLUE)Available backups:$(NC)"
	@ls -la backups/ 2>/dev/null || echo "No backups found"
	@echo "Use 'make databases-restore' to restore database backup"

# =============================================================================
# PERFORMANCE AND OPTIMIZATION
# =============================================================================

performance-test: venv docker-up ## Run comprehensive performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/performance_test.py --test-suite all --duration 5 --users 3 --verbose
	@echo "$(GREEN)✓ Performance tests completed$(NC)"

benchmark: performance-test ## Alias for performance-test

profile: venv ## Profile application performance
	@echo "$(BLUE)Profiling application...$(NC)"
	@echo "$(YELLOW)Profiling tools need to be implemented$(NC)"

# =============================================================================
# CLEANUP OPERATIONS
# =============================================================================

clean: test-clean docs-clean ## Clean generated files
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete
	find . -name "*.egg-info" -type d -delete
	rm -rf build/
	rm -rf dist/
	rm -rf .mypy_cache/
	@echo "$(GREEN)✓ Generated files cleaned$(NC)"

clean-all: clean docker-clean-all ## Clean everything including Docker
	@echo "$(BLUE)Cleaning everything...$(NC)"
	rm -rf $(VENV_DIR)
	@echo "$(GREEN)✓ Everything cleaned$(NC)"

# =============================================================================
# UTILITY TARGETS
# =============================================================================

env-info: ## Show environment information
	@echo "$(BLUE)Environment Information$(NC)"
	@echo "======================="
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python:  $(PYTHON_VERSION)"
	@echo "Venv:    $(VENV_DIR)"
	@echo "Backend: $(BACKEND_DIR)"
	@echo "Frontend: $(FRONTEND_DIR)"
	@echo ""
	@echo "$(BLUE)Service URLs$(NC)"
	@echo "============"
	@echo "Backend:  $(BACKEND_URL)"
	@echo "Frontend: $(FRONTEND_URL)"
	@echo "Weaviate: $(WEAVIATE_URL)"
	@echo "MongoDB:  $(MONGODB_URL)"
	@echo "Ollama:   $(OLLAMA_URL)"

# =============================================================================
# SPECIAL TARGETS
# =============================================================================

# Check if running on a system with NVIDIA GPU
check-gpu:
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "$(GREEN)✓ NVIDIA GPU detected$(NC)"; \
	else \
		echo "$(YELLOW)⚠ NVIDIA GPU not detected - some features may not work$(NC)"; \
	fi

# Initialize git hooks
init-hooks: venv
	@if [ -f .pre-commit-config.yaml ]; then \
		$(VENV_DIR)/bin/pre-commit install; \
		echo "$(GREEN)✓ Git hooks initialized$(NC)"; \
	fi

# Quick development cycle: format, lint, test
dev-cycle: format lint test-unit
	@echo "$(GREEN)✓ Development cycle completed$(NC)"

# =============================================================================
# MAKEFILE MAINTENANCE
# =============================================================================

# Show Makefile configuration
show-config:
	@echo "$(BLUE)Makefile Configuration$(NC)"
	@echo "======================"
	@echo "PROJECT_NAME: $(PROJECT_NAME)"
	@echo "PYTHON_VERSION: $(PYTHON_VERSION)"
	@echo "VENV_DIR: $(VENV_DIR)"
	@echo "COVERAGE_THRESHOLD: $(COVERAGE_THRESHOLD)"
	@echo "DOCKER_COMPOSE: $(DOCKER_COMPOSE)"

# Validate Makefile syntax
validate-makefile:
	@echo "$(BLUE)Validating Makefile...$(NC)"
	@make -n help >/dev/null && echo "$(GREEN)✓ Makefile syntax valid$(NC)" || echo "$(RED)✗ Makefile syntax error$(NC)"

# =============================================================================
# END OF MAKEFILE
# =============================================================================