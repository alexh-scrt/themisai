# Patexia Legal AI Chatbot - Detailed Technology Stack

## System Environment

### Hardware Specifications
- **Operating System**: Ubuntu 24.04 LTS (Noble Numbat)
- **GPU**: NVIDIA H100 (80GB HBM3 memory)
- **Storage**: 500GB SSD/NVMe storage
- **Architecture**: x86_64
- **Docker Support**: Native Linux containers with GPU passthrough

### System Requirements
- **Python Version**: 3.13+ (latest stable)
- **CUDA Version**: 12.0+ (for H100 compatibility)
- **Docker Version**: 24.0+ with Docker Compose v2
- **NVIDIA Container Toolkit**: Latest (for GPU container access)

## Core Technology Stack

### Backend Framework
```yaml
Framework: FastAPI
Version: "^0.110.0"
Purpose: High-performance async API server with automatic OpenAPI docs
Key Features:
  - Native async/await support for concurrent operations
  - WebSocket support for real-time progress tracking
  - Automatic request/response validation with Pydantic
  - Built-in dependency injection for service management
  - Performance: ~10-20x faster than Flask for concurrent workloads
```

### Frontend Interface
```yaml
Framework: Gradio
Version: "^4.15.0"
Purpose: Rapid prototyping web interface for ML applications
Key Features:
  - Two-pane layout support for search interface
  - Real-time component updates via WebSocket
  - File upload with progress tracking
  - Custom CSS/JS injection for legal document styling
  - Built-in user state management for multi-case navigation
```

### AI/ML Processing Framework
```yaml
Framework: LlamaIndex
Version: "^0.10.0"
Purpose: Data framework for LLM applications with RAG capabilities
Key Features:
  - Built-in PDF processing with text extraction
  - Semantic chunking with configurable strategies
  - Multi-modal document loading (.txt, .pdf, future .docx)
  - Query engine abstraction for complex legal reasoning
  - Integration with vector databases and embedding models
  - Support for multi-document agents and sub-question decomposition
```

### Model Management & Serving
```yaml
Service: Ollama
Version: "^0.1.32"
Purpose: Local LLM and embedding model management
Key Features:
  - Automatic model pulling and caching
  - GPU acceleration with CUDA support
  - RESTful API for model operations
  - Concurrent request handling
  - Model versioning and switching
  - Resource monitoring and optimization
Models:
  - mxbai-embed-large: Primary embedding (1000 dimensions)
  - nomic-embed-text: Fallback embedding (768 dimensions)  
  - llama3.1:8b: Text generation (future Phase 2)
```

## Database Technologies

### Vector Database
```yaml
Database: Weaviate
Version: "^1.23.0"
Deployment: Docker container with persistent volumes
Configuration:
  Memory: 8GB allocated for vector operations
  Storage: 50GB for vector indices and metadata
  Collections: Per-case isolation for multi-tenancy
Key Features:
  - Hybrid search (BM25 + vector similarity)
  - GraphQL query interface for complex filtering
  - Built-in reranking capabilities
  - HNSW indexing for fast approximate nearest neighbor search
  - Native metadata filtering and aggregation
  - RESTful API with Python client library
Performance:
  - Query latency: <100ms for 10k+ vectors
  - Indexing throughput: 1000+ documents/minute
  - Concurrent queries: 50+ simultaneous users
```

### Document Database
```yaml
Database: MongoDB
Version: "^7.0"
Deployment: Docker container with named volumes
Configuration:
  Memory: 4GB allocated for document operations
  Storage: 100GB for documents and metadata
  Replication: Single instance for POC (replica set for production)
Collections:
  - cases: Legal case metadata and summaries
  - documents: Full document content and processing metadata
  - search_history: User query history and analytics
  - configurations: System settings and user preferences
Key Features:
  - Flexible schema for evolving document metadata
  - Full-text search capabilities with MongoDB Atlas Search
  - Aggregation pipeline for complex analytics
  - GridFS for large document storage (>16MB)
  - Change streams for real-time data synchronization
Performance:
  - Read throughput: 10k+ ops/second
  - Write throughput: 5k+ ops/second
  - Index performance: <10ms for most queries
```

### Graph Database (Phase 2)
```yaml
Database: Neo4j
Version: "^5.15.0"
Deployment: Docker container (future enhancement)
Configuration:
  Memory: 2GB heap, 1GB page cache
  Storage: 20GB for relationship graphs
Purpose: Document relationship mapping and analysis
Features:
  - Cypher query language for relationship traversal
  - Graph algorithms for document similarity
  - Relationship confidence scoring
  - Visual graph exploration capabilities
```

## Development Tools & Libraries

### Python Core Dependencies
```yaml
Python Version: 3.13
Virtual Environment: venv (standard library)
Package Manager: pip with requirements.txt

Core Libraries:
  fastapi: "^0.110.0"           # Async web framework
  uvicorn: "^0.27.0"            # ASGI server with auto-reload
  gradio: "^4.15.0"             # Web interface framework
  llamaindex: "^0.10.0"         # LLM application framework
  
AI/ML Libraries:
  sentence-transformers: "^2.3.0"  # Embedding model utilities
  torch: "^2.2.0"               # PyTorch for model operations
  transformers: "^4.36.0"       # Hugging Face model library
  numpy: "^1.26.0"              # Numerical computing
  pandas: "^2.1.0"              # Data manipulation and analysis

Vector Database:
  weaviate-client: "^3.25.0"    # Weaviate Python client
  
Document Database:
  pymongo: "^4.6.0"             # MongoDB Python driver
  motor: "^3.3.0"               # Async MongoDB driver

HTTP & WebSocket:
  httpx: "^0.26.0"              # Async HTTP client for Ollama
  websockets: "^12.0"           # WebSocket server implementation
  
PDF Processing:
  pypdf: "^4.0.0"               # PDF text extraction
  pdfplumber: "^0.10.0"         # Advanced PDF parsing
  
Configuration & Monitoring:
  pydantic: "^2.5.0"            # Data validation and settings
  pydantic-settings: "^2.1.0"   # Configuration management
  watchdog: "^3.0.0"            # File system monitoring for hot-reload
  psutil: "^5.9.0"              # System resource monitoring
  
Logging & Debugging:
  structlog: "^23.2.0"          # Structured logging
  rich: "^13.7.0"               # Enhanced console output
  
Testing (Future):
  pytest: "^7.4.0"              # Testing framework
  pytest-asyncio: "^0.23.0"     # Async test support
```

### Container Technologies
```yaml
Container Runtime: Docker Engine
Version: "^24.0.0"
Configuration:
  - Native GPU support with NVIDIA Container Toolkit
  - Named volumes for data persistence
  - Bridge networking for service communication
  - Resource limits and monitoring

Docker Compose:
Version: "^2.20.0"
Services:
  - mongodb: Official MongoDB image with authentication
  - weaviate: Weaviate with vector search modules
  - ollama: Ollama with CUDA GPU support
  - app: Custom Python application container

Volume Management:
  - mongodb_data: 100GB persistent storage
  - weaviate_data: 50GB vector storage
  - ollama_models: 150GB model cache
  - app_logs: 5GB structured logging
```

## GPU & CUDA Configuration

### NVIDIA Software Stack
```yaml
GPU Driver: NVIDIA Driver (Latest stable)
Version: "^545.0"
CUDA Toolkit: CUDA 12.0+
cuDNN: "^8.9.0"
NVIDIA Container Toolkit: Latest
Purpose: GPU acceleration for embedding generation and model inference

GPU Utilization Strategy:
  Primary: Ollama model serving (60-80% GPU utilization)
  Secondary: LlamaIndex document processing (20-40% burst usage)
  Memory Management: Dynamic allocation with 70GB usable memory
  Concurrent Operations: Up to 4 parallel embedding requests
  
Performance Expectations:
  - Embedding Generation: ~1000 tokens/second
  - Model Loading: <30 seconds for large models
  - Batch Processing: 50+ documents simultaneously
  - GPU Memory Usage: 40-60GB during peak operations
```

### CUDA Memory Management
```yaml
Total GPU Memory: 80GB HBM3
Allocation Strategy:
  - Ollama Base: 20GB reserved
  - Model Cache: 30GB for mxbai-embed-large
  - Processing Buffer: 20GB for batch operations
  - System Reserve: 10GB for OS and utilities

Memory Optimization:
  - Dynamic model loading/unloading
  - Batch size optimization based on available memory
  - Garbage collection for embedding operations
  - Memory monitoring with alerts at 90% usage
```

## Storage Architecture

### Storage Allocation (500GB Total)
```yaml
System & OS: 50GB
  - Ubuntu 24.04 base system
  - System packages and dependencies
  - Docker containers and images

Application Storage: 100GB
  - Python virtual environment and packages
  - Application code and configuration files
  - Log files and temporary processing data

Database Storage: 200GB
  - MongoDB: 100GB for documents and metadata
  - Weaviate: 50GB for vector indices
  - Neo4j: 20GB for relationship graphs (Phase 2)
  - Database backups: 30GB

Model Storage: 150GB
  - Ollama model cache directory
  - mxbai-embed-large: ~2GB
  - nomic-embed-text: ~1GB
  - llama3.1:8b: ~5GB
  - Model versioning and fallbacks: 10GB
  - Future model experiments: 130GB buffer

Storage Performance:
  - Sequential Read: 3,500+ MB/s
  - Sequential Write: 3,000+ MB/s
  - Random 4K Read/Write: 500k+ IOPS
  - Sustained throughput for document processing
```

## Network & Security Configuration

### Network Architecture
```yaml
Container Networking: Docker Bridge Network
Internal Communication:
  - MongoDB: Port 27017 (internal only)
  - Weaviate: Port 8080 (internal + admin access)
  - Ollama: Port 11434 (internal API)
  - FastAPI: Port 8000 (external access)
  - Gradio: Port 7860 (external access)

Firewall Configuration:
  - UFW (Uncomplicated Firewall) enabled
  - SSH: Port 22 (restricted IP access)
  - HTTP: Port 8000 (FastAPI API)
  - Web UI: Port 7860 (Gradio interface)
  - Admin: Port 8080 (Weaviate admin, localhost only)
```

### Security Considerations
```yaml
System Security:
  - Automatic security updates enabled
  - Non-root user for application execution
  - SSH key-based authentication only
  - Fail2ban for intrusion prevention

Container Security:
  - Non-privileged containers where possible
  - Read-only file systems for stateless services
  - Resource limits to prevent DoS
  - Regular security scanning of container images

Data Security:
  - MongoDB authentication enabled
  - Weaviate API key authentication
  - SSL/TLS for external communications
  - Local-only deployment (no external model APIs)
```

## Development & Deployment Tools

### Development Environment
```yaml
IDE Support:
  - VS Code with Python, Docker extensions
  - PyCharm Professional (optional)
  - Jupyter notebooks for model experimentation

Code Quality:
  - Black: Code formatting
  - isort: Import sorting
  - flake8: Linting and style checking
  - mypy: Static type checking
  - pre-commit: Git hooks for code quality

Version Control:
  - Git with conventional commits
  - GitHub/GitLab for repository hosting
  - Branch protection and code review
```

### Monitoring & Observability
```yaml
System Monitoring:
  - htop: Real-time system resource monitoring
  - nvidia-smi: GPU utilization monitoring
  - docker stats: Container resource usage
  - Custom dashboard: Resource utilization web interface

Application Monitoring:
  - Structured logging with correlation IDs
  - Performance metrics collection
  - Error tracking and alerting
  - WebSocket connection monitoring
  - Database query performance tracking

Health Checks:
  - Container health checks for all services
  - API endpoint health monitoring
  - Database connectivity checks
  - Model availability validation
  - Disk space and GPU memory alerts
```

## Configuration Management

### Environment Configuration
```yaml
Configuration Format: JSON with hot-reload capability
Configuration Layers:
  - base_config.json: Default system settings
  - environment_config.json: Ubuntu-specific overrides
  - runtime_config.json: Hot-reloadable parameters

Key Configuration Categories:
  ollama_settings:
    base_url: "http://localhost:11434"
    embedding_model: "mxbai-embed-large"
    gpu_memory_fraction: 0.75
    concurrent_requests: 4
    
  system_settings:
    max_workers: 8
    chunk_batch_size: 32
    embedding_cache_size: 10000
    websocket_heartbeat: 30
    
  storage_settings:
    mongodb_uri: "mongodb://localhost:27017"
    weaviate_url: "http://localhost:8080"
    model_cache_dir: "/opt/ollama/models"
    document_storage_limit: "100GB"

Hot-Reload Implementation:
  - File system watcher using Python watchdog
  - Configuration validation before applying changes
  - Graceful service restarts for non-hot-reloadable settings
  - Configuration change logging and rollback capability
```

## Performance Optimization

### System-Level Optimizations
```yaml
Ubuntu Kernel Settings:
  - TCP buffer size optimization for high-throughput operations
  - File descriptor limits increased for concurrent connections
  - Virtual memory settings tuned for large embeddings
  - CPU governor set to performance mode

Docker Optimizations:
  - Resource limits configured per service
  - Shared memory optimization for GPU operations
  - Container restart policies for reliability
  - Log rotation to prevent disk space issues

GPU Optimizations:
  - CUDA memory allocation strategy
  - GPU utilization monitoring and alerting
  - Dynamic batch size adjustment
  - Model caching and preloading
```

### Application-Level Optimizations
```yaml
FastAPI Optimizations:
  - Async database connections with connection pooling
  - Response caching for frequent queries
  - Request batching for embedding operations
  - WebSocket connection pooling and management

LlamaIndex Optimizations:
  - Chunking strategy tuned for legal documents
  - Embedding cache to avoid recomputation
  - Batch processing for document ingestion
  - Query optimization for hybrid search

Database Optimizations:
  - MongoDB indices for fast query performance
  - Weaviate HNSW parameters tuned for legal document similarity
  - Connection pooling and query batching
  - Regular index maintenance and optimization
```

## Backup & Recovery Strategy

### Data Backup
```yaml
MongoDB Backup:
  - Daily automatic backups using mongodump
  - Retention: 7 daily, 4 weekly, 3 monthly
  - Backup location: External storage or cloud
  - Recovery testing: Monthly validation

Weaviate Backup:
  - Vector index snapshots
  - Backup before major configuration changes
  - Schema and data export capabilities
  - Disaster recovery procedures documented

Configuration Backup:
  - Git-based configuration versioning
  - Automated configuration file backups
  - Environment configuration snapshots
  - Rollback procedures for failed deployments
```

## Scalability Considerations

### Horizontal Scaling (Future)
```yaml
Load Balancing:
  - NGINX reverse proxy for API load balancing
  - Session affinity for WebSocket connections
  - Health check integration
  - SSL termination and security headers

Database Scaling:
  - MongoDB replica sets for read scaling
  - Weaviate sharding for large vector collections
  - Read replicas for analytics workloads
  - Connection pooling optimization

Container Orchestration:
  - Kubernetes deployment manifests (future)
  - Docker Swarm for simple multi-node setup
  - Service discovery and configuration management
  - Rolling updates and zero-downtime deployments
```

---

**Hardware Summary**: Ubuntu 24.04 + NVIDIA H100 (80GB) + 500GB Storage
**Software Stack**: Python 3.13 + FastAPI + LlamaIndex + Ollama + Gradio
**Databases**: MongoDB + Weaviate + Neo4j (Phase 2)
**Deployment**: Docker Compose with GPU acceleration
**Performance Target**: <30s document processing, <3s search response
**Development Timeline**: 250 hours over 12.5 weeks