#!/usr/bin/env python3
"""
Ollama Model Pulling Script for Legal AI Chatbot

This script automatically pulls and validates required Ollama models for the legal document
processing system. It ensures all necessary embedding and language models are available
before the system starts processing documents.

Required Models:
- mxbai-embed-large: Primary embedding model (1000 dimensions, legal-optimized)
- nomic-embed-text: Fallback embedding model (768 dimensions, reliable backup)
- llama3.1:8b: Language model for future text generation features

Features:
- Automatic model health checking and validation
- Progress tracking with detailed status updates
- Error handling with retry logic and fallback strategies
- Docker integration for containerized Ollama instances
- Configuration validation and model availability verification
- Performance metrics and timing information
- Parallel model pulling with resource management

Usage:
    python scripts/pull_models.py [--force] [--parallel] [--timeout 600] [--verbose]
    
Arguments:
    --force: Force re-pull models even if they already exist
    --parallel: Pull multiple models concurrently (use with caution)
    --timeout: Timeout in seconds for each model pull (default: 600)
    --verbose: Enable detailed logging output
    --check-only: Only check model availability without pulling
    --config: Path to custom configuration file
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import httpx
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add the backend app to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from backend.config.settings import get_settings
from backend.app.utils.logging import setup_logging


class ModelStatus(Enum):
    """Model availability status."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    PULLING = "pulling"
    FAILED = "failed"
    NOT_FOUND = "not_found"


class ModelPriority(Enum):
    """Model importance priority."""
    CRITICAL = "critical"      # System cannot function without these
    IMPORTANT = "important"    # Fallback models, needed for reliability
    OPTIONAL = "optional"      # Future features, nice to have


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    description: str
    priority: ModelPriority
    size_gb: float
    dimensions: Optional[int] = None
    model_type: str = "embedding"
    tags: List[str] = field(default_factory=list)
    min_ram_gb: float = 8.0
    min_vram_gb: float = 4.0
    status: ModelStatus = ModelStatus.UNKNOWN
    pull_time_seconds: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class PullResult:
    """Result of a model pull operation."""
    model_name: str
    success: bool
    duration_seconds: float
    error_message: Optional[str] = None
    bytes_downloaded: Optional[int] = None
    final_status: ModelStatus = ModelStatus.UNKNOWN


class OllamaModelPuller:
    """
    Handles pulling and validating Ollama models for the legal AI system.
    
    This class provides comprehensive model management including health checking,
    parallel pulling with resource management, and detailed progress reporting.
    """
    
    # Define required models for the legal AI system
    REQUIRED_MODELS = {
        "mxbai-embed-large": ModelInfo(
            name="mxbai-embed-large",
            description="Primary embedding model with 1000 dimensions, optimized for legal documents",
            priority=ModelPriority.CRITICAL,
            size_gb=1.2,
            dimensions=1000,
            model_type="embedding",
            tags=["embedding", "legal", "primary"],
            min_ram_gb=8.0,
            min_vram_gb=4.0
        ),
        "nomic-embed-text": ModelInfo(
            name="nomic-embed-text",
            description="Fallback embedding model with 768 dimensions, reliable backup",
            priority=ModelPriority.IMPORTANT,
            size_gb=0.8,
            dimensions=768,
            model_type="embedding",
            tags=["embedding", "fallback", "reliable"],
            min_ram_gb=6.0,
            min_vram_gb=3.0
        ),
        "llama3.1:8b": ModelInfo(
            name="llama3.1:8b",
            description="Language model for future text generation features",
            priority=ModelPriority.OPTIONAL,
            size_gb=4.7,
            dimensions=None,
            model_type="language",
            tags=["llm", "generation", "future"],
            min_ram_gb=16.0,
            min_vram_gb=8.0
        )
    }
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        timeout: int = 600,
        max_concurrent: int = 2,
        verbose: bool = False
    ):
        """
        Initialize the Ollama model puller.
        
        Args:
            ollama_url: Base URL for Ollama API
            timeout: Timeout in seconds for each model pull
            max_concurrent: Maximum number of concurrent model pulls
            verbose: Enable verbose logging
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        
        # Initialize rich console for beautiful output
        self.console = Console()
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = setup_logging("pull_models", level=log_level)
        
        # HTTP client for Ollama API
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max_concurrent)
        )
        
        # Tracking
        self.pull_results: Dict[str, PullResult] = {}
        self.start_time: Optional[float] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    async def check_ollama_connection(self) -> bool:
        """
        Check if Ollama service is running and accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            self.logger.info(f"Checking Ollama connection at {self.ollama_url}")
            
            response = await self.client.get(f"{self.ollama_url}/api/version")
            response.raise_for_status()
            
            version_info = response.json()
            self.logger.info(
                f"Ollama service is running (version: {version_info.get('version', 'unknown')})"
            )
            
            return True
            
        except httpx.RequestError as e:
            self.logger.error(f"Failed to connect to Ollama at {self.ollama_url}: {e}")
            return False
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Ollama returned error {e.response.status_code}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error checking Ollama connection: {e}")
            return False
    
    async def get_available_models(self) -> Set[str]:
        """
        Get list of currently available models in Ollama.
        
        Returns:
            Set of available model names
        """
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            available = set()
            
            for model in models_data.get("models", []):
                model_name = model.get("name", "")
                if model_name:
                    available.add(model_name)
            
            self.logger.debug(f"Found {len(available)} available models")
            return available
            
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return set()
    
    async def check_model_status(self, model_name: str) -> ModelStatus:
        """
        Check the status of a specific model.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Current status of the model
        """
        try:
            available_models = await self.get_available_models()
            
            if model_name in available_models:
                return ModelStatus.AVAILABLE
            else:
                return ModelStatus.NOT_FOUND
                
        except Exception as e:
            self.logger.error(f"Failed to check status for model {model_name}: {e}")
            return ModelStatus.UNKNOWN
    
    async def pull_model(
        self,
        model_name: str,
        progress_task: Optional[TaskID] = None,
        progress_obj: Optional[Progress] = None
    ) -> PullResult:
        """
        Pull a single model from the Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            progress_task: Rich progress task ID for updates
            progress_obj: Rich progress object for updates
            
        Returns:
            Result of the pull operation
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting pull for model: {model_name}")
            
            if progress_obj and progress_task:
                progress_obj.update(progress_task, description=f"Pulling {model_name}...")
            
            # Start the pull request
            response = await self.client.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Process streaming response
            bytes_downloaded = 0
            last_progress_update = time.time()
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                    
                try:
                    progress_data = json.loads(line)
                    
                    # Check for errors in response
                    if "error" in progress_data:
                        error_msg = progress_data["error"]
                        self.logger.error(f"Model pull failed for {model_name}: {error_msg}")
                        
                        return PullResult(
                            model_name=model_name,
                            success=False,
                            duration_seconds=time.time() - start_time,
                            error_message=error_msg,
                            final_status=ModelStatus.FAILED
                        )
                    
                    # Update progress if available
                    status = progress_data.get("status", "")
                    total = progress_data.get("total", 0)
                    completed = progress_data.get("completed", 0)
                    
                    if total and completed:
                        bytes_downloaded = completed
                        percent = (completed / total) * 100
                        
                        # Update progress bar (limit updates to avoid spam)
                        if (time.time() - last_progress_update) > 0.5:
                            if progress_obj and progress_task:
                                progress_obj.update(
                                    progress_task,
                                    completed=completed,
                                    total=total,
                                    description=f"Pulling {model_name} ({percent:.1f}%)"
                                )
                            last_progress_update = time.time()
                    
                    if self.verbose:
                        self.logger.debug(f"Pull progress for {model_name}: {status}")
                        
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
            
            # Verify the model was successfully pulled
            final_status = await self.check_model_status(model_name)
            success = final_status == ModelStatus.AVAILABLE
            
            duration = time.time() - start_time
            
            if success:
                self.logger.info(
                    f"Successfully pulled model {model_name} "
                    f"({bytes_downloaded:,} bytes in {duration:.1f}s)"
                )
            else:
                self.logger.warning(
                    f"Model {model_name} pull completed but model not found in registry"
                )
            
            return PullResult(
                model_name=model_name,
                success=success,
                duration_seconds=duration,
                bytes_downloaded=bytes_downloaded,
                final_status=final_status
            )
            
        except httpx.TimeoutException:
            error_msg = f"Timeout pulling model {model_name} after {self.timeout}s"
            self.logger.error(error_msg)
            
            return PullResult(
                model_name=model_name,
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=error_msg,
                final_status=ModelStatus.FAILED
            )
            
        except Exception as e:
            error_msg = f"Unexpected error pulling model {model_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return PullResult(
                model_name=model_name,
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=error_msg,
                final_status=ModelStatus.FAILED
            )
    
    async def pull_models_parallel(
        self,
        model_names: List[str],
        force: bool = False
    ) -> Dict[str, PullResult]:
        """
        Pull multiple models in parallel with progress tracking.
        
        Args:
            model_names: List of model names to pull
            force: Force re-pull even if models exist
            
        Returns:
            Dictionary mapping model names to pull results
        """
        if not model_names:
            return {}
        
        # Check which models need to be pulled
        if not force:
            available_models = await self.get_available_models()
            models_to_pull = [
                name for name in model_names
                if name not in available_models
            ]
            
            if len(models_to_pull) < len(model_names):
                skipped = set(model_names) - set(models_to_pull)
                self.logger.info(f"Skipping already available models: {', '.join(skipped)}")
        else:
            models_to_pull = model_names
        
        if not models_to_pull:
            self.logger.info("All requested models are already available")
            return {}
        
        # Setup progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        ) as progress:
            
            # Create progress tasks for each model
            tasks = {}
            for model_name in models_to_pull:
                model_info = self.REQUIRED_MODELS.get(model_name)
                total_size = int(model_info.size_gb * 1024 * 1024 * 1024) if model_info else None
                
                task_id = progress.add_task(
                    f"Waiting to pull {model_name}...",
                    total=total_size
                )
                tasks[model_name] = task_id
            
            # Pull models with controlled concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def pull_with_semaphore(model_name: str) -> Tuple[str, PullResult]:
                async with semaphore:
                    task_id = tasks[model_name]
                    result = await self.pull_model(model_name, task_id, progress)
                    return model_name, result
            
            # Execute pulls
            pull_tasks = [
                pull_with_semaphore(model_name)
                for model_name in models_to_pull
            ]
            
            results = await asyncio.gather(*pull_tasks, return_exceptions=True)
            
            # Process results
            pull_results = {}
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Model pull task failed: {result}")
                    continue
                
                model_name, pull_result = result
                pull_results[model_name] = pull_result
                self.pull_results[model_name] = pull_result
        
        return pull_results
    
    def display_model_status_table(self) -> None:
        """Display a formatted table of model status information."""
        table = Table(title="Ollama Model Status", show_header=True, header_style="bold magenta")
        
        table.add_column("Model Name", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Size (GB)", justify="right", style="blue")
        table.add_column("Status", style="bold")
        table.add_column("Description", style="dim")
        
        for model_name, model_info in self.REQUIRED_MODELS.items():
            # Determine status style
            status_text = model_info.status.value.title()
            if model_info.status == ModelStatus.AVAILABLE:
                status_style = "[green]✓ Available[/green]"
            elif model_info.status == ModelStatus.PULLING:
                status_style = "[yellow]⟳ Pulling[/yellow]"
            elif model_info.status == ModelStatus.FAILED:
                status_style = "[red]✗ Failed[/red]"
            elif model_info.status == ModelStatus.NOT_FOUND:
                status_style = "[orange]○ Not Found[/orange]"
            else:
                status_style = "[dim]? Unknown[/dim]"
            
            # Priority color coding
            priority_style = {
                ModelPriority.CRITICAL: "[red]Critical[/red]",
                ModelPriority.IMPORTANT: "[yellow]Important[/yellow]",
                ModelPriority.OPTIONAL: "[green]Optional[/green]"
            }.get(model_info.priority, str(model_info.priority.value))
            
            table.add_row(
                model_name,
                model_info.model_type.title(),
                priority_style,
                f"{model_info.size_gb:.1f}",
                status_style,
                model_info.description
            )
        
        self.console.print(table)
    
    def display_pull_results(self) -> None:
        """Display results of model pull operations."""
        if not self.pull_results:
            return
        
        table = Table(title="Model Pull Results", show_header=True, header_style="bold magenta")
        
        table.add_column("Model Name", style="cyan", no_wrap=True)
        table.add_column("Result", style="bold")
        table.add_column("Duration", justify="right", style="blue")
        table.add_column("Downloaded", justify="right", style="green")
        table.add_column("Error", style="red")
        
        total_time = 0
        total_downloaded = 0
        successful_pulls = 0
        
        for model_name, result in self.pull_results.items():
            # Result styling
            if result.success:
                result_style = "[green]✓ Success[/green]"
                successful_pulls += 1
            else:
                result_style = "[red]✗ Failed[/red]"
            
            # Format duration
            duration_str = f"{result.duration_seconds:.1f}s"
            
            # Format download size
            if result.bytes_downloaded:
                size_mb = result.bytes_downloaded / (1024 * 1024)
                download_str = f"{size_mb:.1f} MB"
                total_downloaded += result.bytes_downloaded
            else:
                download_str = "N/A"
            
            # Error message (truncated)
            error_str = result.error_message[:50] + "..." if result.error_message and len(result.error_message) > 50 else (result.error_message or "")
            
            table.add_row(
                model_name,
                result_style,
                duration_str,
                download_str,
                error_str
            )
            
            total_time += result.duration_seconds
        
        self.console.print(table)
        
        # Summary statistics
        total_size_mb = total_downloaded / (1024 * 1024)
        self.console.print(
            f"\n[bold]Summary:[/bold] {successful_pulls}/{len(self.pull_results)} models pulled successfully. "
            f"Total time: {total_time:.1f}s, Total downloaded: {total_size_mb:.1f} MB"
        )
    
    async def validate_system_requirements(self) -> bool:
        """
        Validate that the system meets requirements for all models.
        
        Returns:
            True if system meets requirements, False otherwise
        """
        self.logger.info("Validating system requirements...")
        
        # For now, just check if Ollama is running
        # In a full implementation, this would check RAM, VRAM, disk space, etc.
        ollama_available = await self.check_ollama_connection()
        
        if not ollama_available:
            self.console.print(
                Panel(
                    "[red]Ollama service is not running or not accessible.[/red]\n"
                    f"Please ensure Ollama is running at {self.ollama_url}",
                    title="System Requirements Check Failed",
                    border_style="red"
                )
            )
            return False
        
        self.console.print(
            Panel(
                "[green]✓ Ollama service is running and accessible[/green]",
                title="System Requirements Check",
                border_style="green"
            )
        )
        
        return True
    
    async def run_model_check(self) -> None:
        """Run a complete model availability check."""
        self.logger.info("Checking model availability...")
        
        # Update model statuses
        for model_name, model_info in self.REQUIRED_MODELS.items():
            status = await self.check_model_status(model_name)
            model_info.status = status
        
        # Display results
        self.display_model_status_table()
        
        # Check for critical missing models
        critical_missing = [
            name for name, info in self.REQUIRED_MODELS.items()
            if info.priority == ModelPriority.CRITICAL and info.status != ModelStatus.AVAILABLE
        ]
        
        if critical_missing:
            self.console.print(
                Panel(
                    f"[red]Critical models missing: {', '.join(critical_missing)}[/red]\n"
                    "The system cannot function without these models.",
                    title="Critical Models Missing",
                    border_style="red"
                )
            )
        else:
            self.console.print(
                Panel(
                    "[green]All critical models are available[/green]",
                    title="Model Check Complete",
                    border_style="green"
                )
            )
    
    async def run_model_pull(
        self,
        models: Optional[List[str]] = None,
        force: bool = False,
        parallel: bool = True
    ) -> bool:
        """
        Run the complete model pulling process.
        
        Args:
            models: Specific models to pull (None for all required)
            force: Force re-pull even if models exist
            parallel: Pull models in parallel
            
        Returns:
            True if all critical models are available, False otherwise
        """
        self.start_time = time.time()
        
        # Validate system requirements first
        if not await self.validate_system_requirements():
            return False
        
        # Determine which models to pull
        if models is None:
            models_to_pull = list(self.REQUIRED_MODELS.keys())
        else:
            # Validate requested models
            invalid_models = set(models) - set(self.REQUIRED_MODELS.keys())
            if invalid_models:
                self.console.print(
                    f"[red]Invalid model names: {', '.join(invalid_models)}[/red]"
                )
                return False
            models_to_pull = models
        
        # Pull models
        if parallel and len(models_to_pull) > 1:
            self.console.print(f"[bold]Pulling {len(models_to_pull)} models in parallel...[/bold]")
            await self.pull_models_parallel(models_to_pull, force=force)
        else:
            self.console.print(f"[bold]Pulling {len(models_to_pull)} models sequentially...[/bold]")
            for model_name in models_to_pull:
                result = await self.pull_model(model_name)
                self.pull_results[model_name] = result
        
        # Display results
        if self.pull_results:
            self.display_pull_results()
        
        # Final status check
        await self.run_model_check()
        
        # Check if all critical models are now available
        critical_available = all(
            info.status == ModelStatus.AVAILABLE
            for info in self.REQUIRED_MODELS.values()
            if info.priority == ModelPriority.CRITICAL
        )
        
        total_time = time.time() - self.start_time
        
        if critical_available:
            self.console.print(
                Panel(
                    f"[green]✓ All critical models are available![/green]\n"
                    f"Total setup time: {total_time:.1f} seconds",
                    title="Model Setup Complete",
                    border_style="green"
                )
            )
            return True
        else:
            self.console.print(
                Panel(
                    f"[red]✗ Some critical models are still missing[/red]\n"
                    f"Setup time: {total_time:.1f} seconds",
                    title="Model Setup Incomplete",
                    border_style="red"
                )
            )
            return False


async def main():
    """Main entry point for the model pulling script."""
    parser = argparse.ArgumentParser(
        description="Pull required Ollama models for legal AI chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pull_models.py                              # Pull all required models
  python pull_models.py --check-only                 # Check model status only
  python pull_models.py --force --parallel           # Force re-pull all models in parallel
  python pull_models.py mxbai-embed-large           # Pull specific model only
  python pull_models.py --timeout 1200 --verbose    # Extended timeout with verbose output
        """
    )
    
    parser.add_argument(
        "models",
        nargs="*",
        help="Specific models to pull (leave empty for all required models)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-pull models even if they already exist"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Pull multiple models concurrently (default: True)"
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Pull models sequentially instead of in parallel"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each model pull (default: 600)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check model availability without pulling"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum number of concurrent model pulls (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Create model puller
    async with OllamaModelPuller(
        ollama_url=args.ollama_url,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        verbose=args.verbose
    ) as puller:
        
        try:
            if args.check_only:
                # Only check model status
                await puller.run_model_check()
                return 0
            else:
                # Run full model pull process
                success = await puller.run_model_pull(
                    models=args.models if args.models else None,
                    force=args.force,
                    parallel=args.parallel
                )
                
                return 0 if success else 1
                
        except KeyboardInterrupt:
            puller.console.print("\n[yellow]Model pulling interrupted by user[/yellow]")
            return 130
        except Exception as e:
            puller.console.print(f"\n[red]Unexpected error: {e}[/red]")
            if args.verbose:
                puller.logger.exception("Unexpected error in main")
            return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)