#!/usr/bin/env python3
"""
Performance Testing Suite for Legal AI Chatbot

This script provides comprehensive performance testing and benchmarking for the legal document
processing system. It validates system performance against POC success metrics and provides
detailed analysis of resource utilization, response times, and system behavior under load.

Performance Targets (POC Success Metrics):
- Document Processing: <30 seconds per document, >95% success rate
- Search Response: <3 seconds (95th percentile)
- WebSocket Stability: Reliable connections during multi-user testing
- Resource Utilization: GPU memory <80%, system RAM <16GB
- Multi-user Concurrency: 5+ simultaneous users without degradation
- System Uptime: >99% during testing sessions

Test Categories:
1. Document Processing Performance Tests
2. Search Query Performance Tests
3. Resource Utilization Tests
4. Concurrent User Load Tests
5. WebSocket Connection Stability Tests
6. Database Performance Tests
7. AI Model Performance Tests
8. End-to-End Workflow Tests

Features:
- Realistic legal document simulation and processing
- Multi-user concurrent testing with WebSocket validation
- GPU memory and system resource monitoring
- Performance regression detection
- Detailed statistical analysis and reporting
- Automated test data generation
- Performance bottleneck identification
- Load testing with gradual ramp-up

Usage:
    python scripts/performance_test.py [--test-suite] [--duration] [--users] [--verbose]
    
Arguments:
    --test-suite: Specific test suite to run (document|search|resource|load|all)
    --duration: Test duration in minutes (default: 10)
    --users: Number of concurrent users for load testing (default: 5)
    --verbose: Enable detailed logging and real-time monitoring
    --report-only: Generate report from existing test data
    --baseline: Create performance baseline for regression testing
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import random
import statistics
import string
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
import uuid

import aiofiles
import httpx
import numpy as np
import pandas as pd
import psutil
import websockets
from faker import Faker
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout

# Add the backend app to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

try:
    from backend.app.utils.logging import setup_logging
    from backend.config.settings import get_settings
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    def setup_logging(name: str, level: int = logging.INFO):
        logging.basicConfig(level=level)
        return logging.getLogger(name)

try:
    import GPUtil
    import pynvml
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False


class TestCategory(Enum):
    """Performance test categories."""
    DOCUMENT_PROCESSING = "document_processing"
    SEARCH_PERFORMANCE = "search_performance"
    RESOURCE_UTILIZATION = "resource_utilization"
    LOAD_TESTING = "load_testing"
    WEBSOCKET_STABILITY = "websocket_stability"
    DATABASE_PERFORMANCE = "database_performance"
    MODEL_PERFORMANCE = "model_performance"
    END_TO_END = "end_to_end"


class TestResult(Enum):
    """Test result status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    name: str
    value: float
    unit: str
    target: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TestSuiteResult:
    """Results from a performance test suite."""
    suite_name: str
    test_category: TestCategory
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    error_tests: int
    metrics: List[PerformanceMetric] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SystemSnapshot:
    """System resource snapshot for monitoring."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_percent: List[float] = field(default_factory=list)
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0


@dataclass
class LoadTestUser:
    """Simulated user for load testing."""
    user_id: str
    session_id: str
    websocket_url: str
    api_base_url: str
    actions_completed: int = 0
    errors_encountered: int = 0
    total_response_time: float = 0.0
    active: bool = True
    last_action_time: Optional[datetime] = None


class LegalDocumentGenerator:
    """Generate realistic legal documents for testing."""
    
    def __init__(self):
        """Initialize legal document generator."""
        self.faker = Faker()
        
        # Legal document templates and content
        self.legal_terms = [
            "whereas", "heretofore", "hereinafter", "pursuant to", "notwithstanding",
            "in consideration of", "subject to", "without prejudice", "force majeure",
            "intellectual property", "trade secret", "non-disclosure", "liability",
            "indemnification", "jurisdiction", "arbitration", "breach of contract"
        ]
        
        self.patent_terms = [
            "WiFi6", "IEEE 802.11ax", "OFDMA", "MU-MIMO", "BSS coloring",
            "target wake time", "spatial reuse", "wireless standard", "throughput",
            "latency", "spectrum efficiency", "modulation", "beamforming"
        ]
        
        self.contract_sections = [
            "RECITALS", "DEFINITIONS", "SCOPE OF WORK", "PAYMENT TERMS",
            "INTELLECTUAL PROPERTY", "CONFIDENTIALITY", "LIABILITY",
            "TERMINATION", "GOVERNING LAW", "MISCELLANEOUS"
        ]
    
    def generate_patent_document(self, complexity: str = "medium") -> str:
        """Generate a realistic patent application document."""
        if complexity == "simple":
            paragraphs = 5
        elif complexity == "medium":
            paragraphs = 15
        else:  # complex
            paragraphs = 30
        
        title = f"System and Method for {self.faker.catch_phrase()} in {random.choice(self.patent_terms)} Technology"
        
        content = [
            f"PATENT APPLICATION\n\nTitle: {title}\n",
            f"Inventor: {self.faker.name()}",
            f"Application Number: {random.randint(10000000, 99999999)}",
            f"Filing Date: {self.faker.date_between(start_date='-2y', end_date='today')}\n",
            "BACKGROUND OF THE INVENTION\n"
        ]
        
        for i in range(paragraphs):
            paragraph = self._generate_legal_paragraph(include_patent_terms=True)
            content.append(f"[{i+1:04d}] {paragraph}\n")
        
        return "\n".join(content)
    
    def generate_contract_document(self, complexity: str = "medium") -> str:
        """Generate a realistic contract document."""
        if complexity == "simple":
            sections = 3
        elif complexity == "medium":
            sections = 6
        else:  # complex
            sections = 10
        
        party1 = self.faker.company()
        party2 = self.faker.company()
        
        content = [
            f"SOFTWARE LICENSING AGREEMENT\n",
            f"This Agreement is entered into between {party1} ('Licensor') and {party2} ('Licensee').\n",
            f"Effective Date: {self.faker.date_between(start_date='-1y', end_date='today')}\n"
        ]
        
        selected_sections = random.sample(self.contract_sections, min(sections, len(self.contract_sections)))
        
        for section in selected_sections:
            content.append(f"\n{section}\n")
            for _ in range(2, 5):
                paragraph = self._generate_legal_paragraph(include_patent_terms=False)
                content.append(f"{paragraph}\n")
        
        return "\n".join(content)
    
    def _generate_legal_paragraph(self, include_patent_terms: bool = False) -> str:
        """Generate a realistic legal paragraph."""
        sentences = []
        
        for _ in range(random.randint(2, 4)):
            sentence_parts = []
            
            # Add legal terms
            if random.random() < 0.4:
                sentence_parts.append(random.choice(self.legal_terms))
            
            # Add technical terms if patent document
            if include_patent_terms and random.random() < 0.6:
                sentence_parts.append(random.choice(self.patent_terms))
            
            # Add regular content
            sentence_parts.extend([
                self.faker.catch_phrase(),
                "for the purpose of",
                self.faker.bs(),
                "in accordance with applicable law"
            ])
            
            sentence = " ".join(sentence_parts).capitalize() + "."
            sentences.append(sentence)
        
        return " ".join(sentences)
    
    def generate_test_documents(self, count: int, complexity_mix: bool = True) -> List[Tuple[str, str]]:
        """Generate a set of test documents for performance testing."""
        documents = []
        
        for i in range(count):
            if complexity_mix:
                complexity = random.choice(["simple", "medium", "complex"])
            else:
                complexity = "medium"
            
            doc_type = random.choice(["patent", "contract"])
            
            if doc_type == "patent":
                content = self.generate_patent_document(complexity)
                filename = f"patent_application_{i+1:03d}_{complexity}.txt"
            else:
                content = self.generate_contract_document(complexity)
                filename = f"contract_agreement_{i+1:03d}_{complexity}.txt"
            
            documents.append((filename, content))
        
        return documents


class PerformanceMonitor:
    """Real-time performance monitoring during tests."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize performance monitor."""
        self.monitoring_interval = monitoring_interval
        self.snapshots: List[SystemSnapshot] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        if GPU_MONITORING_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = True
            except Exception:
                self.gpu_count = 0
        else:
            self.gpu_count = 0
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> SystemSnapshot:
        """Stop monitoring and return final snapshot."""
        if not self.is_monitoring:
            return self._get_current_snapshot()
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        return self._get_current_snapshot()
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                snapshot = self._get_current_snapshot()
                self.snapshots.append(snapshot)
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.warning(f"Error in monitoring loop: {e}")
    
    def _get_current_snapshot(self) -> SystemSnapshot:
        """Get current system resource snapshot."""
        timestamp = datetime.now(timezone.utc)
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = (memory.total - memory.available) / (1024**3)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0.0
        disk_io_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0.0
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = network_io.bytes_sent / (1024**2) if network_io else 0.0
        network_recv_mb = network_io.bytes_recv / (1024**2) if network_io else 0.0
        
        # GPU metrics
        gpu_utilization = []
        gpu_memory_percent = []
        
        if self.gpu_available:
            try:
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization.append(utilization.gpu)
                    
                    # GPU memory
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_percent.append((memory_info.used / memory_info.total) * 100)
                    
            except Exception as e:
                logging.warning(f"Error reading GPU metrics: {e}")
        
        return SystemSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            gpu_utilization=gpu_utilization,
            gpu_memory_percent=gpu_memory_percent,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from monitoring data."""
        if not self.snapshots:
            return {}
        
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_percent for s in self.snapshots]
        memory_gb_values = [s.memory_used_gb for s in self.snapshots]
        
        stats = {
            "duration_seconds": len(self.snapshots) * self.monitoring_interval,
            "sample_count": len(self.snapshots),
            "cpu": {
                "avg": statistics.mean(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "p95": np.percentile(cpu_values, 95)
            },
            "memory": {
                "avg_percent": statistics.mean(memory_values),
                "max_percent": max(memory_values),
                "avg_gb": statistics.mean(memory_gb_values),
                "max_gb": max(memory_gb_values)
            }
        }
        
        # GPU statistics
        if self.gpu_available and self.snapshots[0].gpu_utilization:
            gpu_util_all = []
            gpu_mem_all = []
            
            for snapshot in self.snapshots:
                gpu_util_all.extend(snapshot.gpu_utilization)
                gpu_mem_all.extend(snapshot.gpu_memory_percent)
            
            if gpu_util_all:
                stats["gpu"] = {
                    "utilization": {
                        "avg": statistics.mean(gpu_util_all),
                        "max": max(gpu_util_all),
                        "p95": np.percentile(gpu_util_all, 95)
                    },
                    "memory": {
                        "avg_percent": statistics.mean(gpu_mem_all),
                        "max_percent": max(gpu_mem_all),
                        "p95_percent": np.percentile(gpu_mem_all, 95)
                    }
                }
        
        return stats


class PerformanceTester:
    """Main performance testing orchestrator."""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        websocket_base_url: str = "ws://localhost:8000",
        verbose: bool = False
    ):
        """Initialize performance tester."""
        self.api_base_url = api_base_url.rstrip("/")
        self.websocket_base_url = websocket_base_url.rstrip("/")
        self.verbose = verbose
        
        # Initialize console and logging
        self.console = Console()
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = setup_logging("performance_test", level=log_level)
        
        # Components
        self.doc_generator = LegalDocumentGenerator()
        self.monitor = PerformanceMonitor()
        
        # HTTP client for API testing
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        # Test results
        self.test_results: Dict[str, TestSuiteResult] = {}
        self.start_time: Optional[datetime] = None
        
        # Performance targets (POC success metrics)
        self.performance_targets = {
            "document_processing_time_seconds": 30.0,
            "document_processing_success_rate": 0.95,
            "search_response_time_seconds": 3.0,
            "search_response_p95_seconds": 3.0,
            "gpu_memory_utilization_percent": 80.0,
            "system_memory_utilization_gb": 16.0,
            "concurrent_users_supported": 5,
            "websocket_stability_percent": 99.0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.aclose()
    
    def display_test_banner(self) -> None:
        """Display test banner with system information."""
        banner_text = f"""
[bold blue]Legal AI Chatbot Performance Testing Suite[/bold blue]
[dim]Comprehensive performance validation and benchmarking[/dim]

[bold]Performance Targets (POC Success Metrics):[/bold]
• Document Processing: <30 seconds per document, >95% success rate
• Search Response: <3 seconds (95th percentile)
• Resource Utilization: GPU memory <80%, system RAM <16GB
• Multi-user Concurrency: 5+ simultaneous users without degradation
• WebSocket Stability: >99% uptime during testing sessions

[bold]Test Configuration:[/bold]
• API Base URL: {self.api_base_url}
• WebSocket URL: {self.websocket_base_url}
• GPU Monitoring: {'Available' if GPU_MONITORING_AVAILABLE else 'Unavailable'}
• System: {platform.system()} {platform.release()}
• CPU Cores: {psutil.cpu_count()}
• Memory: {psutil.virtual_memory().total // (1024**3)} GB
        """
        
        self.console.print(Panel(banner_text.strip(), title="Performance Testing", border_style="blue"))
    
    async def check_system_availability(self) -> bool:
        """Check if all required services are available."""
        self.console.print("\n[bold blue]Checking System Availability...[/bold blue]")
        
        services = {
            "FastAPI Backend": f"{self.api_base_url}/health",
            "MongoDB": f"{self.api_base_url}/api/v1/health/mongodb",
            "Weaviate": f"{self.api_base_url}/api/v1/health/weaviate",
            "Ollama": f"{self.api_base_url}/api/v1/health/ollama"
        }
        
        all_available = True
        
        for service_name, url in services.items():
            try:
                response = await self.http_client.get(url)
                if response.status_code == 200:
                    self.console.print(f"✓ {service_name}: [green]Available[/green]")
                else:
                    self.console.print(f"✗ {service_name}: [red]HTTP {response.status_code}[/red]")
                    all_available = False
            except Exception as e:
                self.console.print(f"✗ {service_name}: [red]Connection failed ({e})[/red]")
                all_available = False
        
        if not all_available:
            self.console.print("\n[red]Some services are unavailable. Please ensure all services are running.[/red]")
        
        return all_available
    
    async def test_document_processing_performance(self, num_documents: int = 10) -> TestSuiteResult:
        """Test document processing performance."""
        suite_name = "Document Processing Performance"
        start_time = datetime.now(timezone.utc)
        
        self.console.print(f"\n[bold blue]Running {suite_name}...[/bold blue]")
        
        # Generate test documents
        test_docs = self.doc_generator.generate_test_documents(num_documents, complexity_mix=True)
        
        processing_times = []
        success_count = 0
        errors = []
        
        await self.monitor.start_monitoring()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Processing documents...", total=len(test_docs))
            
            for i, (filename, content) in enumerate(test_docs):
                progress.update(task, description=f"Processing {filename}...")
                
                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        f.write(content)
                        temp_path = f.name
                    
                    # Upload and process document
                    process_start = time.time()
                    
                    # Simulate file upload
                    with open(temp_path, 'rb') as file_data:
                        files = {"file": (filename, file_data, "text/plain")}
                        data = {"case_id": "performance_test_case"}
                        
                        response = await self.http_client.post(
                            f"{self.api_base_url}/api/v1/documents/upload",
                            files=files,
                            data=data
                        )
                    
                    process_time = time.time() - process_start
                    processing_times.append(process_time)
                    
                    if response.status_code in [200, 201]:
                        success_count += 1
                    else:
                        errors.append(f"Document {filename}: HTTP {response.status_code}")
                    
                    # Clean up temporary file
                    os.unlink(temp_path)
                    
                except Exception as e:
                    errors.append(f"Document {filename}: {str(e)}")
                    # Clean up on error
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                
                progress.advance(task)
        
        monitoring_stats = await self.monitor.stop_monitoring()
        end_time = datetime.now(timezone.utc)
        
        # Calculate metrics
        success_rate = success_count / len(test_docs) if test_docs else 0
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0
        max_processing_time = max(processing_times) if processing_times else 0
        p95_processing_time = np.percentile(processing_times, 95) if processing_times else 0
        
        # Create performance metrics
        metrics = [
            PerformanceMetric(
                name="average_processing_time",
                value=avg_processing_time,
                unit="seconds",
                target=self.performance_targets["document_processing_time_seconds"]
            ),
            PerformanceMetric(
                name="max_processing_time",
                value=max_processing_time,
                unit="seconds",
                target=self.performance_targets["document_processing_time_seconds"]
            ),
            PerformanceMetric(
                name="p95_processing_time",
                value=p95_processing_time,
                unit="seconds",
                target=self.performance_targets["document_processing_time_seconds"]
            ),
            PerformanceMetric(
                name="success_rate",
                value=success_rate,
                unit="percentage",
                target=self.performance_targets["document_processing_success_rate"]
            )
        ]
        
        # Determine test results
        passed_tests = 0
        failed_tests = 0
        warning_tests = 0
        
        if avg_processing_time <= self.performance_targets["document_processing_time_seconds"]:
            passed_tests += 1
        else:
            failed_tests += 1
        
        if success_rate >= self.performance_targets["document_processing_success_rate"]:
            passed_tests += 1
        else:
            failed_tests += 1
        
        return TestSuiteResult(
            suite_name=suite_name,
            test_category=TestCategory.DOCUMENT_PROCESSING,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            total_tests=2,  # processing time and success rate
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            error_tests=len(errors),
            metrics=metrics,
            test_results={
                "processing_times": processing_times,
                "success_count": success_count,
                "total_documents": len(test_docs),
                "monitoring_stats": self.monitor.get_summary_stats()
            },
            errors=errors
        )
    
    async def test_search_performance(self, num_queries: int = 50) -> TestSuiteResult:
        """Test search query performance."""
        suite_name = "Search Performance"
        start_time = datetime.now(timezone.utc)
        
        self.console.print(f"\n[bold blue]Running {suite_name}...[/bold blue]")
        
        # Generate test queries
        test_queries = [
            "IP filings related to WiFi6 wireless standard",
            "liability clauses in software licensing agreements",
            "patent applications for OFDMA technology",
            "contract termination provisions",
            "non-disclosure agreement confidentiality terms",
            "IEEE 802.11ax technical specifications",
            "intellectual property ownership clauses",
            "force majeure provisions in contracts",
            "beamforming technology patent claims",
            "indemnification liability limitations"
        ]
        
        # Extend to required number
        while len(test_queries) < num_queries:
            test_queries.extend(test_queries[:min(10, num_queries - len(test_queries))])
        
        test_queries = test_queries[:num_queries]
        
        response_times = []
        errors = []
        zero_result_queries = 0
        
        await self.monitor.start_monitoring()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Executing search queries...", total=len(test_queries))
            
            for i, query in enumerate(test_queries):
                progress.update(task, description=f"Query {i+1}/{len(test_queries)}...")
                
                try:
                    query_start = time.time()
                    
                    response = await self.http_client.post(
                        f"{self.api_base_url}/api/v1/search/query",
                        json={
                            "query": query,
                            "search_type": "hybrid",
                            "limit": 10
                        }
                    )
                    
                    response_time = time.time() - query_start
                    response_times.append(response_time)
                    
                    if response.status_code == 200:
                        result_data = response.json()
                        if result_data.get("total_results", 0) == 0:
                            zero_result_queries += 1
                    else:
                        errors.append(f"Query {i+1}: HTTP {response.status_code}")
                        
                except Exception as e:
                    errors.append(f"Query {i+1}: {str(e)}")
                
                progress.advance(task)
        
        monitoring_stats = await self.monitor.stop_monitoring()
        end_time = datetime.now(timezone.utc)
        
        # Calculate metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        p99_response_time = np.percentile(response_times, 99) if response_times else 0
        
        # Create performance metrics
        metrics = [
            PerformanceMetric(
                name="average_response_time",
                value=avg_response_time,
                unit="seconds",
                target=self.performance_targets["search_response_time_seconds"]
            ),
            PerformanceMetric(
                name="p95_response_time",
                value=p95_response_time,
                unit="seconds",
                target=self.performance_targets["search_response_p95_seconds"]
            ),
            PerformanceMetric(
                name="p99_response_time",
                value=p99_response_time,
                unit="seconds"
            ),
            PerformanceMetric(
                name="zero_result_rate",
                value=zero_result_queries / len(test_queries) if test_queries else 0,
                unit="percentage"
            )
        ]
        
        # Determine test results
        passed_tests = 0
        failed_tests = 0
        
        if p95_response_time <= self.performance_targets["search_response_p95_seconds"]:
            passed_tests += 1
        else:
            failed_tests += 1
        
        if avg_response_time <= self.performance_targets["search_response_time_seconds"]:
            passed_tests += 1
        else:
            failed_tests += 1
        
        return TestSuiteResult(
            suite_name=suite_name,
            test_category=TestCategory.SEARCH_PERFORMANCE,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            total_tests=2,  # avg response time and p95 response time
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=0,
            error_tests=len(errors),
            metrics=metrics,
            test_results={
                "response_times": response_times,
                "zero_result_queries": zero_result_queries,
                "total_queries": len(test_queries),
                "monitoring_stats": self.monitor.get_summary_stats()
            },
            errors=errors
        )
    
    async def test_concurrent_user_load(self, num_users: int = 5, duration_minutes: int = 5) -> TestSuiteResult:
        """Test concurrent user load performance."""
        suite_name = f"Concurrent User Load ({num_users} users)"
        start_time = datetime.now(timezone.utc)
        
        self.console.print(f"\n[bold blue]Running {suite_name}...[/bold blue]")
        
        # Create simulated users
        users = []
        for i in range(num_users):
            user = LoadTestUser(
                user_id=f"test_user_{i+1:03d}",
                session_id=str(uuid.uuid4()),
                websocket_url=f"{self.websocket_base_url}/api/v1/ws",
                api_base_url=self.api_base_url
            )
            users.append(user)
        
        errors = []
        user_metrics = []
        
        await self.monitor.start_monitoring()
        
        # Run concurrent user simulation
        duration_seconds = duration_minutes * 60
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task(
                f"Simulating {num_users} concurrent users...",
                total=duration_seconds
            )
            
            # Start user simulation tasks
            user_tasks = []
            for user in users:
                user_task = asyncio.create_task(self._simulate_user_activity(user, duration_seconds))
                user_tasks.append(user_task)
            
            # Monitor progress
            start_time_sim = time.time()
            while time.time() - start_time_sim < duration_seconds:
                await asyncio.sleep(1)
                elapsed = time.time() - start_time_sim
                progress.update(task, completed=elapsed)
            
            # Wait for all user tasks to complete
            await asyncio.gather(*user_tasks, return_exceptions=True)
        
        monitoring_stats = await self.monitor.stop_monitoring()
        end_time = datetime.now(timezone.utc)
        
        # Collect user metrics
        total_actions = sum(user.actions_completed for user in users)
        total_errors = sum(user.errors_encountered for user in users)
        active_users = sum(1 for user in users if user.active)
        
        avg_response_time = 0
        if total_actions > 0:
            total_response_time = sum(user.total_response_time for user in users)
            avg_response_time = total_response_time / total_actions
        
        # Create performance metrics
        metrics = [
            PerformanceMetric(
                name="concurrent_users_supported",
                value=active_users,
                unit="count",
                target=self.performance_targets["concurrent_users_supported"]
            ),
            PerformanceMetric(
                name="total_actions_completed",
                value=total_actions,
                unit="count"
            ),
            PerformanceMetric(
                name="error_rate",
                value=total_errors / max(total_actions, 1),
                unit="percentage"
            ),
            PerformanceMetric(
                name="average_action_response_time",
                value=avg_response_time,
                unit="seconds"
            )
        ]
        
        # Determine test results
        passed_tests = 0
        failed_tests = 0
        
        if active_users >= self.performance_targets["concurrent_users_supported"]:
            passed_tests += 1
        else:
            failed_tests += 1
        
        error_rate = total_errors / max(total_actions, 1)
        if error_rate <= 0.05:  # 5% error threshold
            passed_tests += 1
        else:
            failed_tests += 1
        
        return TestSuiteResult(
            suite_name=suite_name,
            test_category=TestCategory.LOAD_TESTING,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            total_tests=2,  # user count and error rate
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=0,
            error_tests=0,
            metrics=metrics,
            test_results={
                "users": [
                    {
                        "user_id": user.user_id,
                        "actions_completed": user.actions_completed,
                        "errors_encountered": user.errors_encountered,
                        "active": user.active,
                        "avg_response_time": user.total_response_time / max(user.actions_completed, 1)
                    }
                    for user in users
                ],
                "monitoring_stats": self.monitor.get_summary_stats()
            },
            errors=errors
        )
    
    async def _simulate_user_activity(self, user: LoadTestUser, duration_seconds: float) -> None:
        """Simulate activity for a single user."""
        end_time = time.time() + duration_seconds
        
        # User action patterns
        actions = [
            self._perform_search_action,
            self._perform_document_upload_action,
            self._perform_case_management_action
        ]
        
        while time.time() < end_time and user.active:
            try:
                # Select random action
                action = random.choice(actions)
                
                # Perform action with timing
                action_start = time.time()
                await action(user)
                action_time = time.time() - action_start
                
                user.actions_completed += 1
                user.total_response_time += action_time
                user.last_action_time = datetime.now(timezone.utc)
                
                # Random delay between actions (1-5 seconds)
                await asyncio.sleep(random.uniform(1.0, 5.0))
                
            except Exception as e:
                user.errors_encountered += 1
                if user.errors_encountered > 10:  # Deactivate user after too many errors
                    user.active = False
                    break
    
    async def _perform_search_action(self, user: LoadTestUser) -> None:
        """Perform a search action for simulated user."""
        queries = [
            "patent applications WiFi6",
            "contract liability clauses",
            "intellectual property agreements",
            "IEEE wireless standards"
        ]
        
        query = random.choice(queries)
        
        response = await self.http_client.post(
            f"{user.api_base_url}/api/v1/search/query",
            json={
                "query": query,
                "search_type": "hybrid",
                "limit": 5
            }
        )
        
        response.raise_for_status()
    
    async def _perform_document_upload_action(self, user: LoadTestUser) -> None:
        """Perform a document upload action for simulated user."""
        # Generate small test document
        content = self.doc_generator._generate_legal_paragraph(include_patent_terms=True)
        filename = f"test_doc_{user.user_id}_{int(time.time())}.txt"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as file_data:
                files = {"file": (filename, file_data, "text/plain")}
                data = {"case_id": f"load_test_case_{user.user_id}"}
                
                response = await self.http_client.post(
                    f"{user.api_base_url}/api/v1/documents/upload",
                    files=files,
                    data=data
                )
            
            response.raise_for_status()
        finally:
            os.unlink(temp_path)
    
    async def _perform_case_management_action(self, user: LoadTestUser) -> None:
        """Perform a case management action for simulated user."""
        # Get cases list
        response = await self.http_client.get(
            f"{user.api_base_url}/api/v1/cases"
        )
        
        response.raise_for_status()
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance test report."""
        if not self.test_results:
            return "No test results available for report generation."
        
        report_lines = [
            "# Legal AI Chatbot Performance Test Report",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Test Duration: {(datetime.now(timezone.utc) - self.start_time).total_seconds():.1f} seconds\n" if self.start_time else "",
            "## Executive Summary\n"
        ]
        
        # Overall statistics
        total_tests = sum(result.total_tests for result in self.test_results.values())
        total_passed = sum(result.passed_tests for result in self.test_results.values())
        total_failed = sum(result.failed_tests for result in self.test_results.values())
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report_lines.extend([
            f"- Total Tests Executed: {total_tests}",
            f"- Tests Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)",
            f"- Tests Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)",
            f"- Overall Success Rate: {overall_success_rate:.1f}%\n"
        ])
        
        # Performance targets compliance
        report_lines.append("## Performance Targets Compliance\n")
        
        for suite_name, result in self.test_results.items():
            report_lines.append(f"### {suite_name}\n")
            
            for metric in result.metrics:
                status = "✓ PASS" if metric.target and metric.value <= metric.target else "✗ FAIL"
                target_info = f" (target: {metric.target} {metric.unit})" if metric.target else ""
                report_lines.append(f"- {metric.name}: {metric.value:.2f} {metric.unit}{target_info} {status}")
            
            report_lines.append("")
        
        # Detailed results
        report_lines.append("## Detailed Test Results\n")
        
        for suite_name, result in self.test_results.items():
            report_lines.extend([
                f"### {suite_name}",
                f"- Duration: {result.duration_seconds:.1f} seconds",
                f"- Success Rate: {result.passed_tests/result.total_tests*100:.1f}%",
                f"- Errors: {len(result.errors)}",
                ""
            ])
            
            if result.errors:
                report_lines.append("**Errors:**")
                for error in result.errors[:5]:  # Show first 5 errors
                    report_lines.append(f"- {error}")
                if len(result.errors) > 5:
                    report_lines.append(f"- ... and {len(result.errors) - 5} more errors")
                report_lines.append("")
        
        # Resource utilization summary
        report_lines.append("## Resource Utilization Summary\n")
        
        for suite_name, result in self.test_results.items():
            monitoring_stats = result.test_results.get("monitoring_stats", {})
            if monitoring_stats:
                report_lines.extend([
                    f"### {suite_name}",
                    f"- Average CPU: {monitoring_stats.get('cpu', {}).get('avg', 0):.1f}%",
                    f"- Peak CPU: {monitoring_stats.get('cpu', {}).get('max', 0):.1f}%",
                    f"- Average Memory: {monitoring_stats.get('memory', {}).get('avg_gb', 0):.1f} GB",
                    f"- Peak Memory: {monitoring_stats.get('memory', {}).get('max_gb', 0):.1f} GB",
                    ""
                ])
                
                gpu_stats = monitoring_stats.get('gpu', {})
                if gpu_stats:
                    report_lines.extend([
                        f"- Average GPU Utilization: {gpu_stats.get('utilization', {}).get('avg', 0):.1f}%",
                        f"- Peak GPU Utilization: {gpu_stats.get('utilization', {}).get('max', 0):.1f}%",
                        f"- Average GPU Memory: {gpu_stats.get('memory', {}).get('avg_percent', 0):.1f}%",
                        f"- Peak GPU Memory: {gpu_stats.get('memory', {}).get('max_percent', 0):.1f}%",
                        ""
                    ])
        
        return "\n".join(report_lines)
    
    def display_results_table(self) -> None:
        """Display test results in a formatted table."""
        if not self.test_results:
            self.console.print("[yellow]No test results to display[/yellow]")
            return
        
        table = Table(title="Performance Test Results", show_header=True, header_style="bold magenta")
        
        table.add_column("Test Suite", style="cyan", no_wrap=True)
        table.add_column("Duration", justify="right", style="blue")
        table.add_column("Tests", justify="center", style="green")
        table.add_column("Success Rate", justify="center", style="bold")
        table.add_column("Key Metrics", style="dim")
        
        for suite_name, result in self.test_results.items():
            duration = f"{result.duration_seconds:.1f}s"
            tests = f"{result.passed_tests}/{result.total_tests}"
            success_rate = f"{result.passed_tests/result.total_tests*100:.1f}%"
            
            # Get key metric
            key_metric = ""
            if result.metrics:
                metric = result.metrics[0]
                key_metric = f"{metric.name}: {metric.value:.2f} {metric.unit}"
            
            # Color coding for success rate
            if result.passed_tests == result.total_tests:
                success_style = "[green]✓ " + success_rate + "[/green]"
            elif result.passed_tests > 0:
                success_style = "[yellow]⚠ " + success_rate + "[/yellow]"
            else:
                success_style = "[red]✗ " + success_rate + "[/red]"
            
            table.add_row(
                suite_name,
                duration,
                tests,
                success_style,
                key_metric
            )
        
        self.console.print(table)
    
    async def run_full_test_suite(
        self,
        test_categories: Optional[List[TestCategory]] = None,
        duration_minutes: int = 10,
        concurrent_users: int = 5
    ) -> bool:
        """Run the complete performance test suite."""
        self.start_time = datetime.now(timezone.utc)
        
        if test_categories is None:
            test_categories = [
                TestCategory.DOCUMENT_PROCESSING,
                TestCategory.SEARCH_PERFORMANCE,
                TestCategory.LOAD_TESTING
            ]
        
        # Check system availability
        if not await self.check_system_availability():
            self.console.print("[red]System availability check failed - aborting tests[/red]")
            return False
        
        try:
            # Run test suites
            for category in test_categories:
                if category == TestCategory.DOCUMENT_PROCESSING:
                    result = await self.test_document_processing_performance(num_documents=10)
                    self.test_results["Document Processing"] = result
                
                elif category == TestCategory.SEARCH_PERFORMANCE:
                    result = await self.test_search_performance(num_queries=50)
                    self.test_results["Search Performance"] = result
                
                elif category == TestCategory.LOAD_TESTING:
                    result = await self.test_concurrent_user_load(
                        num_users=concurrent_users,
                        duration_minutes=duration_minutes
                    )
                    self.test_results["Concurrent User Load"] = result
            
            # Display results
            self.display_results_table()
            
            # Generate and display report
            report = self.generate_performance_report()
            self.console.print("\n" + "="*80)
            self.console.print(report)
            
            # Determine overall success
            total_tests = sum(result.total_tests for result in self.test_results.values())
            total_passed = sum(result.passed_tests for result in self.test_results.values())
            overall_success = total_passed == total_tests
            
            if overall_success:
                self.console.print("\n[green]✓ All performance tests passed![/green]")
            else:
                failed_tests = total_tests - total_passed
                self.console.print(f"\n[red]✗ {failed_tests} performance tests failed[/red]")
            
            return overall_success
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Performance testing interrupted by user[/yellow]")
            return False
        except Exception as e:
            self.console.print(f"\n[red]Performance testing failed with error: {e}[/red]")
            if self.verbose:
                self.logger.exception("Unexpected error during performance testing")
            return False


async def main():
    """Main entry point for the performance testing script."""
    parser = argparse.ArgumentParser(
        description="Performance testing suite for legal AI chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python performance_test.py                           # Run all tests with defaults
  python performance_test.py --test-suite document    # Run document processing tests only
  python performance_test.py --users 10 --duration 15 # Load test with 10 users for 15 minutes
  python performance_test.py --verbose --report-only  # Generate report from existing data
        """
    )
    
    parser.add_argument(
        "--test-suite",
        choices=["document", "search", "load", "all"],
        default="all",
        help="Specific test suite to run"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration in minutes for load testing (default: 10)"
    )
    parser.add_argument(
        "--users",
        type=int,
        default=5,
        help="Number of concurrent users for load testing (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging and real-time monitoring"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--websocket-url",
        default="ws://localhost:8000",
        help="WebSocket base URL (default: ws://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    # Determine test categories
    test_categories = []
    if args.test_suite == "all":
        test_categories = [
            TestCategory.DOCUMENT_PROCESSING,
            TestCategory.SEARCH_PERFORMANCE,
            TestCategory.LOAD_TESTING
        ]
    elif args.test_suite == "document":
        test_categories = [TestCategory.DOCUMENT_PROCESSING]
    elif args.test_suite == "search":
        test_categories = [TestCategory.SEARCH_PERFORMANCE]
    elif args.test_suite == "load":
        test_categories = [TestCategory.LOAD_TESTING]
    
    # Create performance tester
    async with PerformanceTester(
        api_base_url=args.api_url,
        websocket_base_url=args.websocket_url,
        verbose=args.verbose
    ) as tester:
        
        # Display banner
        tester.display_test_banner()
        
        # Run tests
        success = await tester.run_full_test_suite(
            test_categories=test_categories,
            duration_minutes=args.duration,
            concurrent_users=args.users
        )
        
        return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nPerformance testing cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)