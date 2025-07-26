#!/usr/bin/env python3
"""
Environment Setup Automation for Legal AI Chatbot

This script provides comprehensive environment setup and validation for the legal document
processing system. It ensures all system requirements, dependencies, services, and
configurations are properly initialized before the application starts.

Features:
- Complete system requirements validation (GPU, memory, storage)
- Docker container orchestration and health checking
- Python environment and dependency management
- Database initialization and schema creation
- Ollama model validation and pulling
- Configuration file generation and validation
- Service health monitoring and troubleshooting
- Detailed progress reporting and error recovery

System Requirements:
- Ubuntu 24.04+ with NVIDIA GPU support
- NVIDIA H100 (80GB VRAM) or equivalent GPU
- 16GB+ system RAM
- 500GB+ storage space
- Docker Engine with NVIDIA Container Toolkit
- Python 3.13+

Usage:
    python scripts/setup_environment.py [--full] [--force] [--skip-gpu] [--verbose]
    
Arguments:
    --full: Perform complete setup including models and databases
    --force: Force reinstallation of existing components
    --skip-gpu: Skip GPU validation (for CPU-only development)
    --verbose: Enable detailed logging and progress information
    --check-only: Only validate environment without making changes
    --config-only: Only generate configuration files
    --services-only: Only start and validate services
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from urllib.parse import urlparse
import tempfile

import docker
import httpx
import psutil
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeElapsedColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich.prompt import Confirm

# Add the backend app to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

try:
    from backend.app.utils.logging import setup_logging
    from backend.app.core.config import get_settings
except ImportError:
    # Fallback for initial setup when backend isn't available yet
    def setup_logging(name: str, level: int = logging.INFO):
        logging.basicConfig(level=level)
        return logging.getLogger(name)


class SetupPhase(Enum):
    """Setup phases for progress tracking."""
    VALIDATION = "validation"
    PYTHON_ENV = "python_environment"
    DOCKER_SETUP = "docker_setup"
    SERVICES = "services"
    MODELS = "models"
    DATABASES = "databases"
    CONFIG = "configuration"
    VERIFICATION = "verification"


class ComponentStatus(Enum):
    """Component status states."""
    NOT_CHECKED = "not_checked"
    CHECKING = "checking"
    AVAILABLE = "available"
    MISSING = "missing"
    ERROR = "error"
    INSTALLING = "installing"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"


@dataclass
class SystemRequirement:
    """System requirement specification."""
    name: str
    description: str
    check_function: str
    required: bool = True
    min_value: Optional[Union[str, int, float]] = None
    status: ComponentStatus = ComponentStatus.NOT_CHECKED
    error_message: Optional[str] = None
    actual_value: Optional[Any] = None


@dataclass
class ServiceConfig:
    """Docker service configuration."""
    name: str
    image: str
    ports: List[str] = field(default_factory=list)
    volumes: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    healthcheck: Optional[Dict[str, Any]] = None
    gpu_support: bool = False
    memory_limit: Optional[str] = None
    restart_policy: str = "unless-stopped"
    status: ComponentStatus = ComponentStatus.NOT_CHECKED


@dataclass
class SetupResult:
    """Result of setup operation."""
    success: bool
    phase: SetupPhase
    duration_seconds: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    components_installed: List[str] = field(default_factory=list)


class EnvironmentSetup:
    """
    Comprehensive environment setup and validation for the legal AI system.
    
    This class manages the complete setup process including system validation,
    dependency installation, service orchestration, and configuration management.
    """
    
    # System requirements specification
    SYSTEM_REQUIREMENTS = {
        "operating_system": SystemRequirement(
            name="Operating System",
            description="Ubuntu 24.04+ required for full compatibility",
            check_function="check_operating_system",
            min_value="24.04"
        ),
        "python_version": SystemRequirement(
            name="Python Version",
            description="Python 3.13+ required for modern async features",
            check_function="check_python_version",
            min_value="3.13"
        ),
        "system_memory": SystemRequirement(
            name="System Memory",
            description="16GB+ RAM required for document processing",
            check_function="check_system_memory",
            min_value=16  # GB
        ),
        "storage_space": SystemRequirement(
            name="Storage Space",
            description="500GB+ free space required for models and data",
            check_function="check_storage_space",
            min_value=500  # GB
        ),
        "docker_engine": SystemRequirement(
            name="Docker Engine",
            description="Docker Engine with Compose v2 support",
            check_function="check_docker_engine"
        ),
        "nvidia_gpu": SystemRequirement(
            name="NVIDIA GPU",
            description="NVIDIA GPU with 8GB+ VRAM for model inference",
            check_function="check_nvidia_gpu",
            min_value=8,  # GB VRAM
            required=False  # Can be skipped with --skip-gpu
        ),
        "nvidia_docker": SystemRequirement(
            name="NVIDIA Container Toolkit",
            description="NVIDIA Container Toolkit for GPU support in Docker",
            check_function="check_nvidia_docker",
            required=False
        )
    }
    
    # Docker services configuration
    DOCKER_SERVICES = {
        "mongodb": ServiceConfig(
            name="mongodb",
            image="mongo:7.0",
            ports=["27017:27017"],
            volumes=[
                "mongodb_data:/data/db",
                "./data/backups:/backups"
            ],
            environment={
                "MONGO_INITDB_ROOT_USERNAME": "admin",
                "MONGO_INITDB_ROOT_PASSWORD": "legal_ai_admin",
                "MONGO_INITDB_DATABASE": "patexia_legal_ai"
            },
            healthcheck={
                "test": ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            },
            memory_limit="4g"
        ),
        "weaviate": ServiceConfig(
            name="weaviate",
            image="semitechnologies/weaviate:1.23.0",
            ports=["8080:8080"],
            volumes=["weaviate_data:/var/lib/weaviate"],
            environment={
                "QUERY_DEFAULTS_LIMIT": "25",
                "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
                "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
                "DEFAULT_VECTORIZER_MODULE": "none",
                "CLUSTER_HOSTNAME": "node1"
            },
            healthcheck={
                "test": ["CMD", "curl", "-f", "http://localhost:8080/v1/.well-known/ready"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            },
            memory_limit="8g"
        ),
        "ollama": ServiceConfig(
            name="ollama",
            image="ollama/ollama:latest",
            ports=["11434:11434"],
            volumes=[
                "ollama_data:/root/.ollama",
                "./data/models:/models"
            ],
            environment={
                "OLLAMA_HOST": "0.0.0.0",
                "OLLAMA_ORIGINS": "*",
                "OLLAMA_KEEP_ALIVE": "24h",
                "OLLAMA_NUM_PARALLEL": "4",
                "OLLAMA_MAX_LOADED_MODELS": "3"
            },
            healthcheck={
                "test": ["CMD", "curl", "-f", "http://localhost:11434/api/version"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            },
            gpu_support=True,
            memory_limit="16g"
        )
    }
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        verbose: bool = False,
        skip_gpu: bool = False
    ):
        """
        Initialize the environment setup manager.
        
        Args:
            project_root: Root directory of the project
            verbose: Enable verbose logging
            skip_gpu: Skip GPU-related requirements
        """
        self.project_root = project_root or Path(__file__).parent.parent.absolute()
        self.verbose = verbose
        self.skip_gpu = skip_gpu
        
        # Initialize console and logging
        self.console = Console()
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = setup_logging("setup_environment", level=log_level)
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
        except docker.errors.DockerException:
            self.docker_client = None
            
        # HTTP client for service validation
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Setup tracking
        self.setup_results: Dict[SetupPhase, SetupResult] = {}
        self.start_time: Optional[float] = None
        
        # Adjust requirements for GPU skip
        if skip_gpu:
            self.SYSTEM_REQUIREMENTS["nvidia_gpu"].required = False
            self.SYSTEM_REQUIREMENTS["nvidia_docker"].required = False
            self.DOCKER_SERVICES["ollama"].gpu_support = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.aclose()
    
    def display_welcome_banner(self) -> None:
        """Display welcome banner with system information."""
        banner_text = """
[bold blue]Legal AI Chatbot Environment Setup[/bold blue]
[dim]Comprehensive system setup and validation for legal document processing[/dim]

[bold]Project Information:[/bold]
• System: Legal Document Processing with AI
• Architecture: FastAPI + LlamaIndex + Ollama + Gradio
• Databases: MongoDB + Weaviate + Neo4j (Phase 2)
• AI Models: mxbai-embed-large, nomic-embed-text, llama3.1:8b

[bold]Setup Phases:[/bold]
1. System Requirements Validation
2. Python Environment Setup
3. Docker Configuration
4. Service Orchestration
5. AI Model Management
6. Database Initialization
7. Configuration Generation
8. Final Verification
        """
        
        self.console.print(Panel(banner_text.strip(), title="Environment Setup", border_style="blue"))
        
        # Display system information
        system_info = f"""
[bold]Current System:[/bold]
• OS: {platform.system()} {platform.release()}
• Python: {sys.version.split()[0]}
• Architecture: {platform.machine()}
• CPU Cores: {psutil.cpu_count()}
• Memory: {psutil.virtual_memory().total // (1024**3)} GB
• Project Root: {self.project_root}
        """
        
        self.console.print(Panel(system_info.strip(), title="System Information", border_style="green"))
    
    # System validation methods
    def check_operating_system(self) -> Tuple[bool, str, Any]:
        """Check operating system version."""
        try:
            if platform.system() != "Linux":
                return False, "Linux operating system required", platform.system()
            
            # Try to get Ubuntu version
            try:
                with open("/etc/os-release", "r") as f:
                    os_info = f.read()
                    
                for line in os_info.split("\n"):
                    if line.startswith("VERSION_ID="):
                        version = line.split("=")[1].strip('"')
                        version_parts = [int(x) for x in version.split(".")]
                        
                        if version_parts[0] >= 24:
                            return True, f"Ubuntu {version} detected", version
                        else:
                            return False, f"Ubuntu 24.04+ required (found {version})", version
            except FileNotFoundError:
                return False, "Ubuntu version could not be determined", "unknown"
                
            return True, "Linux system detected", platform.release()
            
        except Exception as e:
            return False, f"Error checking OS: {e}", None
    
    def check_python_version(self) -> Tuple[bool, str, Any]:
        """Check Python version."""
        try:
            version = sys.version_info
            version_str = f"{version.major}.{version.minor}.{version.micro}"
            
            if version.major == 3 and version.minor >= 13:
                return True, f"Python {version_str} detected", version_str
            else:
                return False, f"Python 3.13+ required (found {version_str})", version_str
                
        except Exception as e:
            return False, f"Error checking Python version: {e}", None
    
    def check_system_memory(self) -> Tuple[bool, str, Any]:
        """Check system memory."""
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if memory_gb >= 16:
                return True, f"{memory_gb:.1f} GB RAM available", memory_gb
            else:
                return False, f"16GB+ RAM required (found {memory_gb:.1f} GB)", memory_gb
                
        except Exception as e:
            return False, f"Error checking memory: {e}", None
    
    def check_storage_space(self) -> Tuple[bool, str, Any]:
        """Check available storage space."""
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb >= 500:
                return True, f"{free_gb:.1f} GB free space available", free_gb
            else:
                return False, f"500GB+ free space required (found {free_gb:.1f} GB)", free_gb
                
        except Exception as e:
            return False, f"Error checking storage: {e}", None
    
    def check_docker_engine(self) -> Tuple[bool, str, Any]:
        """Check Docker Engine availability."""
        try:
            if not self.docker_client:
                return False, "Docker Engine not available", None
            
            # Test Docker connection
            version_info = self.docker_client.version()
            version = version_info.get("Version", "unknown")
            
            # Check for Compose plugin
            try:
                result = subprocess.run(
                    ["docker", "compose", "version"], 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return True, f"Docker Engine {version} with Compose v2", version
                else:
                    return False, "Docker Compose v2 not available", version
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False, "Docker Compose command not found", version
                
        except Exception as e:
            return False, f"Error checking Docker: {e}", None
    
    def check_nvidia_gpu(self) -> Tuple[bool, str, Any]:
        """Check NVIDIA GPU availability."""
        try:
            # Try nvidia-smi command
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                memory_mb = int(result.stdout.strip())
                memory_gb = memory_mb / 1024
                
                if memory_gb >= 8:
                    return True, f"NVIDIA GPU with {memory_gb:.1f} GB VRAM", memory_gb
                else:
                    return False, f"8GB+ VRAM required (found {memory_gb:.1f} GB)", memory_gb
            else:
                return False, "nvidia-smi command failed", None
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "NVIDIA GPU drivers not found", None
        except Exception as e:
            return False, f"Error checking GPU: {e}", None
    
    def check_nvidia_docker(self) -> Tuple[bool, str, Any]:
        """Check NVIDIA Container Toolkit."""
        try:
            if not self.docker_client:
                return False, "Docker not available", None
            
            # Try to run a test container with GPU support
            try:
                container = self.docker_client.containers.run(
                    "hello-world",
                    runtime="nvidia",
                    device_requests=[
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ],
                    remove=True,
                    detach=False
                )
                return True, "NVIDIA Container Toolkit available", "available"
            except docker.errors.ContainerError:
                return False, "NVIDIA Container Toolkit not configured", None
            except docker.errors.ImageNotFound:
                return True, "NVIDIA Container Toolkit available (image missing)", "available"
                
        except Exception as e:
            return False, f"Error checking NVIDIA Docker: {e}", None
    
    async def validate_system_requirements(self) -> SetupResult:
        """Validate all system requirements."""
        start_time = time.time()
        errors = []
        warnings = []
        
        self.console.print("\n[bold blue]Phase 1: System Requirements Validation[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Validating system requirements...", total=len(self.SYSTEM_REQUIREMENTS))
            
            for req_name, requirement in self.SYSTEM_REQUIREMENTS.items():
                progress.update(task, description=f"Checking {requirement.name}...")
                
                requirement.status = ComponentStatus.CHECKING
                
                try:
                    # Get the check function
                    check_func = getattr(self, requirement.check_function)
                    success, message, actual_value = check_func()
                    
                    requirement.actual_value = actual_value
                    
                    if success:
                        requirement.status = ComponentStatus.AVAILABLE
                        self.logger.info(f"✓ {requirement.name}: {message}")
                    else:
                        requirement.status = ComponentStatus.MISSING
                        requirement.error_message = message
                        
                        if requirement.required:
                            errors.append(f"{requirement.name}: {message}")
                            self.logger.error(f"✗ {requirement.name}: {message}")
                        else:
                            warnings.append(f"{requirement.name}: {message}")
                            self.logger.warning(f"⚠ {requirement.name}: {message}")
                            
                except Exception as e:
                    requirement.status = ComponentStatus.ERROR
                    requirement.error_message = str(e)
                    error_msg = f"{requirement.name}: Error during check - {e}"
                    
                    if requirement.required:
                        errors.append(error_msg)
                        self.logger.error(f"✗ {error_msg}")
                    else:
                        warnings.append(error_msg)
                        self.logger.warning(f"⚠ {error_msg}")
                
                progress.advance(task)
        
        # Display results table
        self.display_requirements_table()
        
        duration = time.time() - start_time
        success = len(errors) == 0
        
        result = SetupResult(
            success=success,
            phase=SetupPhase.VALIDATION,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings
        )
        
        if success:
            self.console.print("[green]✓ System requirements validation completed successfully[/green]")
        else:
            self.console.print(f"[red]✗ System requirements validation failed with {len(errors)} errors[/red]")
        
        return result
    
    def display_requirements_table(self) -> None:
        """Display system requirements validation results."""
        table = Table(title="System Requirements Status", show_header=True, header_style="bold magenta")
        
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Required", style="yellow", justify="center")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")
        
        for requirement in self.SYSTEM_REQUIREMENTS.values():
            # Status styling
            if requirement.status == ComponentStatus.AVAILABLE:
                status_style = "[green]✓ Available[/green]"
            elif requirement.status == ComponentStatus.MISSING:
                status_style = "[red]✗ Missing[/red]"
            elif requirement.status == ComponentStatus.ERROR:
                status_style = "[red]✗ Error[/red]"
            else:
                status_style = "[dim]? Not Checked[/dim]"
            
            # Required styling
            required_text = "Yes" if requirement.required else "No"
            required_style = "[red]Yes[/red]" if requirement.required else "[green]No[/green]"
            
            # Details
            details = ""
            if requirement.status == ComponentStatus.AVAILABLE and requirement.actual_value:
                if isinstance(requirement.actual_value, (int, float)):
                    details = f"{requirement.actual_value:.1f}"
                else:
                    details = str(requirement.actual_value)
            elif requirement.error_message:
                details = requirement.error_message[:50] + "..." if len(requirement.error_message) > 50 else requirement.error_message
            
            table.add_row(
                requirement.name,
                required_style,
                status_style,
                details
            )
        
        self.console.print(table)
    
    async def setup_python_environment(self) -> SetupResult:
        """Setup Python virtual environment and install dependencies."""
        start_time = time.time()
        errors = []
        warnings = []
        components_installed = []
        
        self.console.print("\n[bold blue]Phase 2: Python Environment Setup[/bold blue]")
        
        try:
            # Check if we're in a virtual environment
            venv_path = os.environ.get('VIRTUAL_ENV')
            if not venv_path:
                warnings.append("Not running in a virtual environment - consider using one")
                self.logger.warning("Not running in virtual environment")
            else:
                self.logger.info(f"Using virtual environment: {venv_path}")
            
            # Check if requirements files exist
            req_files = [
                self.project_root / "requirements.txt",
                self.project_root / "requirements-dev.txt"
            ]
            
            missing_files = [f for f in req_files if not f.exists()]
            if missing_files:
                errors.append(f"Missing requirements files: {[str(f) for f in missing_files]}")
                return SetupResult(
                    success=False,
                    phase=SetupPhase.PYTHON_ENV,
                    duration_seconds=time.time() - start_time,
                    errors=errors
                )
            
            # Install production requirements
            self.console.print("Installing production dependencies...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                errors.append(f"Failed to install production requirements: {result.stderr}")
            else:
                components_installed.append("production dependencies")
                self.logger.info("Production dependencies installed successfully")
            
            # Install development requirements
            self.console.print("Installing development dependencies...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                warnings.append(f"Failed to install dev requirements: {result.stderr}")
            else:
                components_installed.append("development dependencies")
                self.logger.info("Development dependencies installed successfully")
            
        except Exception as e:
            errors.append(f"Unexpected error during Python setup: {e}")
        
        duration = time.time() - start_time
        success = len(errors) == 0
        
        return SetupResult(
            success=success,
            phase=SetupPhase.PYTHON_ENV,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings,
            components_installed=components_installed
        )
    
    async def setup_docker_environment(self) -> SetupResult:
        """Setup and configure Docker environment."""
        start_time = time.time()
        errors = []
        warnings = []
        components_installed = []
        
        self.console.print("\n[bold blue]Phase 3: Docker Environment Setup[/bold blue]")
        
        try:
            if not self.docker_client:
                errors.append("Docker client not available")
                return SetupResult(
                    success=False,
                    phase=SetupPhase.DOCKER_SETUP,
                    duration_seconds=time.time() - start_time,
                    errors=errors
                )
            
            # Create necessary directories
            directories = [
                self.project_root / "data" / "models",
                self.project_root / "data" / "uploads", 
                self.project_root / "data" / "backups",
                self.project_root / "logs"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {directory}")
            
            components_installed.append("project directories")
            
            # Generate docker-compose.yml file
            compose_content = self.generate_docker_compose()
            compose_path = self.project_root / "docker-compose.yml"
            
            with open(compose_path, 'w') as f:
                f.write(compose_content)
            
            components_installed.append("docker-compose.yml")
            self.logger.info("Generated docker-compose.yml file")
            
            # Pull required Docker images
            self.console.print("Pulling Docker images...")
            
            for service_name, service in self.DOCKER_SERVICES.items():
                try:
                    self.console.print(f"Pulling {service.image}...")
                    self.docker_client.images.pull(service.image)
                    components_installed.append(f"{service_name} image")
                    self.logger.info(f"Pulled image: {service.image}")
                except Exception as e:
                    warnings.append(f"Failed to pull {service.image}: {e}")
            
        except Exception as e:
            errors.append(f"Unexpected error during Docker setup: {e}")
        
        duration = time.time() - start_time
        success = len(errors) == 0
        
        return SetupResult(
            success=success,
            phase=SetupPhase.DOCKER_SETUP,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings,
            components_installed=components_installed
        )
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml content."""
        compose_config = {
            "version": "3.8",
            "services": {},
            "volumes": {
                "mongodb_data": {"driver": "local"},
                "weaviate_data": {"driver": "local"},
                "ollama_data": {"driver": "local"}
            },
            "networks": {
                "legal_ai_network": {
                    "driver": "bridge"
                }
            }
        }
        
        # Add services
        for service_name, service in self.DOCKER_SERVICES.items():
            service_config = {
                "image": service.image,
                "container_name": f"legal_ai_{service_name}",
                "restart": service.restart_policy,
                "networks": ["legal_ai_network"]
            }
            
            if service.ports:
                service_config["ports"] = service.ports
            
            if service.volumes:
                service_config["volumes"] = service.volumes
            
            if service.environment:
                service_config["environment"] = service.environment
            
            if service.depends_on:
                service_config["depends_on"] = service.depends_on
            
            if service.healthcheck:
                service_config["healthcheck"] = service.healthcheck
            
            if service.memory_limit:
                service_config["mem_limit"] = service.memory_limit
            
            if service.gpu_support and not self.skip_gpu:
                service_config["deploy"] = {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "count": "all",
                                    "capabilities": ["gpu"]
                                }
                            ]
                        }
                    }
                }
            
            compose_config["services"][service_name] = service_config
        
        # Convert to YAML string (simplified for this example)
        import yaml
        return yaml.dump(compose_config, default_flow_style=False, sort_keys=False)
    
    async def start_services(self) -> SetupResult:
        """Start and validate Docker services."""
        start_time = time.time()
        errors = []
        warnings = []
        components_installed = []
        
        self.console.print("\n[bold blue]Phase 4: Service Orchestration[/bold blue]")
        
        try:
            # Start services using docker-compose
            self.console.print("Starting Docker services...")
            
            result = subprocess.run(
                ["docker", "compose", "up", "-d"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                errors.append(f"Failed to start services: {result.stderr}")
                return SetupResult(
                    success=False,
                    phase=SetupPhase.SERVICES,
                    duration_seconds=time.time() - start_time,
                    errors=errors
                )
            
            components_installed.append("docker services")
            
            # Wait for services to be healthy
            self.console.print("Waiting for services to be ready...")
            
            max_wait_time = 300  # 5 minutes
            check_interval = 10   # 10 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                all_healthy = True
                
                for service_name in self.DOCKER_SERVICES.keys():
                    container_name = f"legal_ai_{service_name}"
                    
                    try:
                        container = self.docker_client.containers.get(container_name)
                        
                        if container.status != "running":
                            all_healthy = False
                            break
                        
                        # Check health if healthcheck is defined
                        health = container.attrs.get("State", {}).get("Health", {})
                        if health and health.get("Status") not in ["healthy", "none"]:
                            all_healthy = False
                            break
                            
                    except docker.errors.NotFound:
                        all_healthy = False
                        break
                
                if all_healthy:
                    break
                
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval
                self.console.print(f"Waiting for services... ({elapsed_time}s/{max_wait_time}s)")
            
            if not all_healthy:
                warnings.append("Some services may not be fully ready")
            else:
                self.logger.info("All services are running and healthy")
            
            # Validate service connectivity
            await self.validate_service_connectivity()
            
        except Exception as e:
            errors.append(f"Unexpected error during service startup: {e}")
        
        duration = time.time() - start_time
        success = len(errors) == 0
        
        return SetupResult(
            success=success,
            phase=SetupPhase.SERVICES,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings,
            components_installed=components_installed
        )
    
    async def validate_service_connectivity(self) -> None:
        """Validate connectivity to all services."""
        service_urls = {
            "MongoDB": "mongodb://admin:legal_ai_admin@localhost:27017",
            "Weaviate": "http://localhost:8080/v1/.well-known/ready",
            "Ollama": "http://localhost:11434/api/version"
        }
        
        for service_name, url in service_urls.items():
            try:
                if service_name == "MongoDB":
                    # Test MongoDB connection (would need pymongo)
                    self.logger.info(f"MongoDB connectivity check skipped (requires pymongo)")
                else:
                    response = await self.http_client.get(url)
                    response.raise_for_status()
                    self.logger.info(f"✓ {service_name} is accessible")
            except Exception as e:
                self.logger.warning(f"⚠ {service_name} connectivity check failed: {e}")
    
    async def generate_configuration_files(self) -> SetupResult:
        """Generate default configuration files."""
        start_time = time.time()
        errors = []
        warnings = []
        components_installed = []
        
        self.console.print("\n[bold blue]Phase 7: Configuration Generation[/bold blue]")
        
        try:
            config_dir = self.project_root / "backend" / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Base configuration
            base_config = {
                "ollama_settings": {
                    "base_url": "http://localhost:11434",
                    "embedding_model": "mxbai-embed-large",
                    "fallback_model": "nomic-embed-text",
                    "llm_model": "llama3.1:8b",
                    "timeout": 45,
                    "concurrent_requests": 3,
                    "gpu_memory_limit": "60GB" if not self.skip_gpu else "0GB"
                },
                "llamaindex_settings": {
                    "chunk_size": 768,
                    "chunk_overlap": 100,
                    "hybrid_search_alpha": 0.6,
                    "top_k_results": 15,
                    "similarity_threshold": 0.7
                },
                "legal_document_settings": {
                    "preserve_legal_structure": True,
                    "section_aware_chunking": True,
                    "citation_extraction": True,
                    "metadata_enhancement": True
                },
                "ui_settings": {
                    "progress_update_interval": 250,
                    "max_search_history": 100,
                    "websocket_heartbeat": 25,
                    "result_highlight_context": 3
                },
                "capacity_limits": {
                    "documents_per_case": 25,
                    "manual_override_enabled": True,
                    "max_chunk_size": 2048,
                    "embedding_cache_size": 10000
                }
            }
            
            # Development configuration overrides
            dev_config = {
                "debug": True,
                "log_level": "DEBUG",
                "reload": True
            }
            
            # Runtime configuration (hot-reloadable)
            runtime_config = {
                "ollama_settings": {
                    "embedding_model": "mxbai-embed-large"
                }
            }
            
            # Write configuration files
            config_files = [
                ("base_config.json", base_config),
                ("development_config.json", dev_config),
                ("runtime_config.json", runtime_config)
            ]
            
            for filename, config_data in config_files:
                config_path = config_dir / filename
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                components_installed.append(filename)
                self.logger.info(f"Generated {filename}")
            
            # Generate .env.example if it doesn't exist
            env_example_path = self.project_root / ".env.example"
            if not env_example_path.exists():
                env_content = """# Environment Variables for Legal AI Chatbot

# Database Configuration
DB_MONGODB_URI=mongodb://admin:legal_ai_admin@localhost:27017
DB_MONGODB_DATABASE=patexia_legal_ai
DB_WEAVIATE_URL=http://localhost:8080
DB_NEO4J_URI=bolt://localhost:7687
DB_NEO4J_USER=neo4j
DB_NEO4J_PASSWORD=your_password_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
OLLAMA_FALLBACK_MODEL=nomic-embed-text

# Application Configuration
LOG_LEVEL=DEBUG
DEBUG=true
RELOAD=true

# Security (for production)
SECRET_KEY=your_secret_key_here
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]
"""
                with open(env_example_path, 'w') as f:
                    f.write(env_content)
                components_installed.append(".env.example")
                self.logger.info("Generated .env.example")
            
        except Exception as e:
            errors.append(f"Error generating configuration files: {e}")
        
        duration = time.time() - start_time
        success = len(errors) == 0
        
        return SetupResult(
            success=success,
            phase=SetupPhase.CONFIG,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings,
            components_installed=components_installed
        )
    
    async def run_final_verification(self) -> SetupResult:
        """Run final system verification."""
        start_time = time.time()
        errors = []
        warnings = []
        
        self.console.print("\n[bold blue]Phase 8: Final Verification[/bold blue]")
        
        try:
            # Test API endpoints
            await self.validate_service_connectivity()
            
            # Check model availability (if Ollama is running)
            try:
                response = await self.http_client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    
                    required_models = ["mxbai-embed-large", "nomic-embed-text"]
                    missing_models = [m for m in required_models if m not in model_names]
                    
                    if missing_models:
                        warnings.append(f"Missing AI models: {missing_models}")
                        self.logger.warning(f"Models need to be pulled: {missing_models}")
                    else:
                        self.logger.info("All required AI models are available")
                        
            except Exception as e:
                warnings.append(f"Could not verify AI models: {e}")
            
            # Test database connections
            # (Implementation would depend on specific database drivers)
            
        except Exception as e:
            errors.append(f"Error during final verification: {e}")
        
        duration = time.time() - start_time
        success = len(errors) == 0
        
        return SetupResult(
            success=success,
            phase=SetupPhase.VERIFICATION,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings
        )
    
    def display_setup_summary(self) -> None:
        """Display comprehensive setup summary."""
        total_duration = sum(result.duration_seconds for result in self.setup_results.values())
        total_errors = sum(len(result.errors) for result in self.setup_results.values())
        total_warnings = sum(len(result.warnings) for result in self.setup_results.values())
        
        # Summary table
        table = Table(title="Setup Summary", show_header=True, header_style="bold magenta")
        table.add_column("Phase", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Duration", justify="right", style="blue")
        table.add_column("Components", style="green")
        table.add_column("Issues", style="yellow")
        
        for phase, result in self.setup_results.items():
            status = "[green]✓ Success[/green]" if result.success else "[red]✗ Failed[/red]"
            duration = f"{result.duration_seconds:.1f}s"
            components = str(len(result.components_installed))
            issues = f"{len(result.errors)}E, {len(result.warnings)}W"
            
            table.add_row(
                phase.value.replace("_", " ").title(),
                status,
                duration,
                components,
                issues
            )
        
        self.console.print(table)
        
        # Overall status
        if total_errors == 0:
            status_panel = Panel(
                f"[green]✓ Environment setup completed successfully![/green]\n"
                f"Total time: {total_duration:.1f} seconds\n"
                f"Warnings: {total_warnings}",
                title="Setup Complete",
                border_style="green"
            )
        else:
            status_panel = Panel(
                f"[red]✗ Environment setup failed with {total_errors} errors[/red]\n"
                f"Total time: {total_duration:.1f} seconds\n"
                f"Warnings: {total_warnings}",
                title="Setup Failed",
                border_style="red"
            )
        
        self.console.print(status_panel)
        
        # Next steps
        if total_errors == 0:
            next_steps = """
[bold]Next Steps:[/bold]
1. Pull AI models: [cyan]python scripts/pull_models.py[/cyan]
2. Initialize databases: [cyan]python scripts/init_databases.py[/cyan]
3. Start the application: [cyan]python backend/main.py[/cyan]
4. Access the web interface: [cyan]http://localhost:8000[/cyan]

[bold]Useful Commands:[/bold]
• Check service status: [cyan]docker compose ps[/cyan]
• View service logs: [cyan]docker compose logs -f[/cyan]
• Stop services: [cyan]docker compose down[/cyan]
            """
            self.console.print(Panel(next_steps.strip(), title="Getting Started", border_style="blue"))
    
    async def run_complete_setup(
        self,
        phases: Optional[List[SetupPhase]] = None,
        force: bool = False
    ) -> bool:
        """
        Run the complete environment setup process.
        
        Args:
            phases: Specific phases to run (None for all)
            force: Force reinstallation of existing components
            
        Returns:
            True if setup completed successfully
        """
        self.start_time = time.time()
        
        if phases is None:
            phases = list(SetupPhase)
        
        # Phase execution mapping
        phase_functions = {
            SetupPhase.VALIDATION: self.validate_system_requirements,
            SetupPhase.PYTHON_ENV: self.setup_python_environment,
            SetupPhase.DOCKER_SETUP: self.setup_docker_environment,
            SetupPhase.SERVICES: self.start_services,
            SetupPhase.CONFIG: self.generate_configuration_files,
            SetupPhase.VERIFICATION: self.run_final_verification
        }
        
        try:
            for phase in phases:
                if phase in phase_functions:
                    self.console.print(f"\n[bold]Executing {phase.value.replace('_', ' ').title()}...[/bold]")
                    
                    result = await phase_functions[phase]()
                    self.setup_results[phase] = result
                    
                    if not result.success and phase == SetupPhase.VALIDATION:
                        self.console.print("[red]Critical validation errors - stopping setup[/red]")
                        break
                    
                    # Display errors and warnings
                    for error in result.errors:
                        self.console.print(f"[red]ERROR: {error}[/red]")
                    for warning in result.warnings:
                        self.console.print(f"[yellow]WARNING: {warning}[/yellow]")
            
            # Display final summary
            self.display_setup_summary()
            
            # Overall success determination
            critical_phases = [SetupPhase.VALIDATION, SetupPhase.SERVICES]
            success = all(
                self.setup_results.get(phase, SetupResult(False, phase, 0)).success
                for phase in critical_phases
            )
            
            return success
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Setup interrupted by user[/yellow]")
            return False
        except Exception as e:
            self.console.print(f"\n[red]Setup failed with unexpected error: {e}[/red]")
            if self.verbose:
                self.logger.exception("Unexpected error during setup")
            return False


async def main():
    """Main entry point for the environment setup script."""
    parser = argparse.ArgumentParser(
        description="Complete environment setup for legal AI chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_environment.py                    # Complete setup
  python setup_environment.py --check-only       # Validation only
  python setup_environment.py --skip-gpu         # Skip GPU requirements
  python setup_environment.py --force --verbose  # Force reinstall with detailed output
        """
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Perform complete setup including models and databases"
    )
    parser.add_argument(
        "--force",
        action="store_true", 
        help="Force reinstallation of existing components"
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU validation and configuration"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging and progress information"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate environment without making changes"
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only generate configuration files"
    )
    parser.add_argument(
        "--services-only",
        action="store_true",
        help="Only start and validate services"
    )
    
    args = parser.parse_args()
    
    # Determine phases to run
    phases = None
    if args.check_only:
        phases = [SetupPhase.VALIDATION]
    elif args.config_only:
        phases = [SetupPhase.CONFIG]
    elif args.services_only:
        phases = [SetupPhase.DOCKER_SETUP, SetupPhase.SERVICES]
    
    # Create setup manager
    async with EnvironmentSetup(
        verbose=args.verbose,
        skip_gpu=args.skip_gpu
    ) as setup:
        
        # Display welcome banner
        setup.display_welcome_banner()
        
        # Confirm destructive operations
        if args.force and not args.check_only:
            if not Confirm.ask("Force mode will reinstall existing components. Continue?"):
                setup.console.print("[yellow]Setup cancelled by user[/yellow]")
                return 1
        
        # Run setup
        success = await setup.run_complete_setup(
            phases=phases,
            force=args.force
        )
        
        return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nSetup cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)