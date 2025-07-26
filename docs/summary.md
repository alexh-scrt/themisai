# Patexia Legal AI Chatbot - Architecture Specification

## Project Overview

**Objective**: Build a legal AI chatbot for case-based document management, semantic search, and text generation using legal documents in .txt and .pdf formats.

**Primary Technology Stack**:
- **Backend**: Python 3.13, FastAPI
- **Frontend**: Gradio (for rapid prototyping)
- **Vector Store**: Weaviate
- **Database**: MongoDB (Docker)
- **Model Management**: Ollama
- **Embedding Model**: mxbai-embed-large (primary), nomic-embed-text (fallback)
- **Document Processing**: LlamaIndex with built-in PDF loader

## Core Architecture Decisions

### Document Processing & Storage
- **Document Capacity**: 25 documents per case (with manual override capability)
- **Chunking Strategy**: Paragraph-based semantic chunking with legal document structure preservation
- **Hierarchy Metadata**: Maintain document hierarchy for proper citation and source attribution
- **File Formats**: .txt and .pdf initially
- **Processing Flow**: Batch-based document ingestion with background relationship extraction

### Embedding & Model Selection
- **Primary Embedding Model**: `mxbai-embed-large`
  - State-of-the-art performance on MTEB benchmark
  - Excellent domain generalization for legal documents
  - 1,000 dimensional embeddings
  - Trained without MTEB data overlap
- **Fallback Model**: `nomic-embed-text`
- **Model Management**: Ollama for automated pulling, caching, and serving
- **Model Switching**: Runtime model changes via admin panel with immediate validation

### Vector Store & Search
- **Vector Database**: Weaviate with per-case collections
- **Search Types**: Hybrid search (semantic + keyword) with configurable alpha parameter
- **Search Features**: Reciprocal rank fusion, re-ranking, metadata filtering
- **Query Processing**: LlamaIndex query engines for multi-step legal reasoning

### Multi-tenancy & Case Management
- **Case Isolation**: Each legal case has its own Weaviate collection
- **Multi-user Support**: Multiple lawyers working on multiple cases simultaneously
- **Case Creation**: Lawyers create cases first with name, summary, and visual markers
- **Visual Markers**: Predefined color/icon combinations for easy case identification

## Technical Infrastructure

### Docker Environment
```yaml
services:
  mongodb:
    image: mongo:latest
    volumes: [mongodb_data:/data/db]
    ports: ["27017:27017"]
    
  weaviate:
    image: semitechnologies/weaviate:latest
    volumes: [weaviate_data:/var/lib/weaviate]
    ports: ["8080:8080"]
    
  ollama:
    image: ollama/ollama:latest
    volumes: [ollama_data:/root/.ollama]
    ports: ["11434:11434"]
    gpu_support: nvidia

volumes: [mongodb_data, weaviate_data, ollama_data]
```

### Model Pre-loading Strategy
```bash
# Essential models to pre-pull during POC setup
docker exec ollama ollama pull mxbai-embed-large     # Primary embedding
docker exec ollama ollama pull nomic-embed-text      # Fallback embedding  
docker exec ollama ollama pull llama3.1:8b          # Future text generation
```

### Hot-Reload Configuration System
- **Configuration Format**: JSON with immediate validation
- **Validation Timing**: Immediate Ollama model availability checking on config changes
- **Invalid Config Handling**: Reject invalid configurations with detailed console error messages
- **Rollback Strategy**: Preserve previous working configuration
- **Scope**: Global configuration for POC (future: per-user, per-case hierarchy)

### WebSocket Connection Pooling
- **Connection Management**: Per-case connection groups for targeted progress broadcasts
- **Multi-user Support**: Multiple concurrent lawyers with connection isolation
- **Progress Tracking**: Real-time document processing updates via WebSocket
- **Resource Monitoring**: Live GPU/CPU/Memory usage broadcasting to admin panel

## User Interface Design

### Gradio Layout Structure
```
┌─────────────────────────────────────────────────────────────────────┐
│                        Header Bar                                   │
├──────────┬──────────────────────────────────────────────────────────┤
│          │                    Main Content Area                     │
│ Sidebar  ├─────────────────┬───────────────────────────────────────┤
│ Case Nav │   Pane 1        │           Pane 2                     │
│          │   Document      │      Document Viewer                 │
│ • Case A │   Results       │      with Highlights                 │
│ • Case B │   List          │                                       │
│          │                 │                                       │
│ [+ New]  │                 │                                       │
│          ├─────────────────┴───────────────────────────────────────┤
│          │               Search History Panel                      │
└──────────┴─────────────────────────────────────────────────────────┘
```

### Case Management UX
- **Case Creation Flow**: Create case → provide summary → assign visual marker → upload documents
- **Case Switching**: Sidebar navigation with instant switching and context preservation
- **Visual Markers**: Predefined color/icon combinations for easy identification
- **Auto-summary**: AI-generated case summaries updated after document processing

### Search Interface
- **Two-Pane Design**: 
  - Pane 1: Document results list with relevance scores and match previews
  - Pane 2: Document viewer with highlighted matching sections
- **Search Trigger**: Button-based search (not keystroke-triggered)
- **Search History**: Persistent search history with quick replay functionality
- **Result Navigation**: Scroll-to-match, context expansion, citation-ready snippets

### Progress Tracking
- **Three-Stage Progress**: Upload (0-33%) → Storage (33-66%) → Vectorization (66-100%)
- **WebSocket Updates**: Real-time progress with granular status messages
- **Error Handling**: Red progress bars with detailed error messages and retry options
- **Batch Processing**: Overall progress indication for multiple document uploads

## Database Schemas

### MongoDB Collections

#### Cases Collection
```json
{
  "_id": ObjectId,
  "case_id": "CASE_2025_001",
  "user_id": "lawyer_123",
  "case_name": "Patent Dispute - WiFi6 Technology",
  "initial_summary": "User-provided brief description",
  "auto_summary": "AI-generated comprehensive summary",
  "visual_marker": {
    "color": "#FF5733",
    "icon": "legal-document"
  },
  "created_at": ISODate,
  "updated_at": ISODate,
  "document_count": 0,
  "processing_status": "active|processing|complete"
}
```

#### Documents Collection
```json
{
  "_id": ObjectId,
  "user_id": "lawyer_123",
  "case_id": "CASE_2025_001",
  "document": "Full document text content...",
  "metadata": {
    "document_name": "Contract_Amendment_v2.pdf",
    "original_filename": "Contract_Amendment_v2.pdf",
    "file_type": "pdf",
    "date_added": ISODate,
    "file_size": 245632,
    "page_count": 12,
    "processing_status": "pending|processing|completed|failed",
    "extraction_metadata": {
      "processing_method": "LlamaIndex_PDF",
      "confidence": 0.95
    }
  }
}
```

#### Search History Collection
```json
{
  "_id": ObjectId,
  "user_id": "lawyer_123",
  "case_id": "CASE_2025_001",
  "search_queries": [
    {
      "query": "Find all IP filings related to WiFi6",
      "timestamp": ISODate,
      "results_count": 15,
      "search_type": "hybrid",
      "execution_time": 2.3
    }
  ],
  "max_history": 100
}
```

## Document Relationship Management (Phase 2)

### Graph Database Integration
- **Database**: Neo4j for complex document relationships
- **Relationship Types**: AMENDS, REFERENCES, SUPERSEDES, RELATES_TO, CITES
- **Extraction Method**: NLP-based relationship extraction (background processing)
- **Confidence Scoring**: Store relationship confidence scores for filtering
- **Processing**: Separate background process to avoid blocking main document flow

### Neo4j Relationship Schema
```cypher
(:Document)-[:RELATES_TO {
    confidence: 0.85,
    extraction_method: "NLP_model_v1",
    extracted_at: timestamp,
    relationship_type: "amendment",
    evidence_text: "extracted context"
}]->(:Document)
```

## Configuration Management

### JSON Configuration Structure
```json
{
  "ollama_settings": {
    "base_url": "http://localhost:11434",
    "embedding_model": "mxbai-embed-large",
    "fallback_model": "nomic-embed-text",
    "llm_model": "llama3.1:8b",
    "timeout": 45,
    "concurrent_requests": 3,
    "gpu_memory_limit": "8GB"
  },
  "llamaindex_settings": {
    "chunk_size": 768,
    "chunk_overlap": 100,
    "hybrid_search_alpha": 0.6,
    "top_k_results": 15,
    "similarity_threshold": 0.7
  },
  "legal_document_settings": {
    "preserve_legal_structure": true,
    "section_aware_chunking": true,
    "citation_extraction": true,
    "metadata_enhancement": true
  },
  "ui_settings": {
    "progress_update_interval": 250,
    "max_search_history": 100,
    "websocket_heartbeat": 25,
    "result_highlight_context": 3
  },
  "capacity_limits": {
    "documents_per_case": 25,
    "manual_override_enabled": true,
    "max_chunk_size": 2048,
    "embedding_cache_size": 10000
  }
}
```

### Admin Configuration Panel
```
┌─────────────────────────────────────────────────────────┐
│                 Admin Configuration                     │
├─────────────────────────────────────────────────────────┤
│ Model Management:                                       │
│ ├─ Embedding Model: [mxbai-embed-large ▼]             │
│ ├─ Fallback Model: [nomic-embed-text ▼]               │
│ └─ Model Status: ✓ Available                           │
├─────────────────────────────────────────────────────────┤
│ LlamaIndex Settings:                                    │
│ ├─ Chunk Size: [768] Overlap: [100]                   │
│ ├─ Hybrid α: [0.6] Top-k: [15]                        │
│ └─ Similarity Threshold: [0.7]                         │
├─────────────────────────────────────────────────────────┤
│ Resource Monitor:                                       │
│ ├─ GPU Usage: ████████░░ 80% (6.4/8GB)                │
│ ├─ CPU Usage: ██████░░░░ 60%                           │
│ ├─ Active Connections: 5                               │
│ └─ Queue Length: 3 requests                            │
├─────────────────────────────────────────────────────────┤
│ [Test Configuration] [Save] [Reset] [Export Settings]  │
└─────────────────────────────────────────────────────────┘
```

## Error Handling & Logging

### Error Recovery Strategy
- **Document Processing Errors**: Non-blocking, retry individual documents first, then re-upload
- **Model Unavailability**: Graceful fallback to cached models with user notification
- **Configuration Validation**: Immediate rejection of invalid configs with detailed error messages
- **WebSocket Disconnections**: Automatic reconnection with state preservation
- **Database Connectivity**: Retry logic with exponential backoff

### Debug Logging Strategy
- **Output**: Console output for POC (structured JSON format)
- **Log Categories**: Document processing, WebSocket events, search operations, config changes, error recovery
- **Correlation IDs**: Track operations across components
- **Performance Metrics**: Processing times, memory usage, query latencies

### Console Logging Examples
```
[2025-07-25 10:30:15] DEBUG [WebSocket] Connection established: user_123, case_456
[2025-07-25 10:30:16] DEBUG [Config] Hot-reload triggered: ollama_settings.embedding_model
[2025-07-25 10:30:17] INFO [Config] Model switch successful: mxbai-embed-large → nomic-embed-text
[2025-07-25 10:30:18] ERROR [Config] Invalid configuration rejected: chunk_size must be > 0
[2025-07-25 10:30:19] DEBUG [Document] Processing started: doc_789, method=LlamaIndex_PDF
[2025-07-25 10:30:20] INFO [Search] Query executed: "IP filings WiFi6", results=15, time=2.3s
```

## Development Phases

### Phase 1A: Infrastructure + Ollama Integration (Week 1)
- Docker environment setup with all services
- Ollama model pre-pulling and validation scripts
- Hot-reload JSON configuration system implementation
- Resource monitoring foundation
- DEBUG console logging framework

### Phase 1B: Document Processing Pipeline (Week 2)
- LlamaIndex integration with mxbai-embed-large
- Legal document chunking with structure preservation
- WebSocket progress tracking with resource awareness
- Document capacity management (25 doc limit)
- Graceful failure handling and fallback mechanisms

### Phase 1C: Search Interface & Case Management (Week 3)
- Two-pane Gradio interface implementation
- Case management sidebar with visual markers
- Search functionality with button triggers
- Search history storage and replay
- Admin configuration panel with resource monitoring

### Phase 1D: Integration & Polish (Week 4)
- End-to-end workflow testing
- Multi-user connection pooling validation
- Performance optimization for legal documents
- Error handling refinement
- Legal professional user acceptance testing

### Phase 2: Relationship Extraction (Future)
- Neo4j integration for document relationships
- NLP-based relationship extraction pipeline
- Background processing with confidence scoring
- Enhanced query engines with graph context
- Complex multi-step legal reasoning capabilities

## Query Complexity Examples

### Simple Queries
- "Find all IP filings related to Wireless standard WiFi6"
- Standard vector similarity search with metadata filtering
- Hybrid search (semantic + keyword)

### Complex Multi-Step Queries
- "Find all contracts related to X, then analyze liability clauses"
- SubQuestionQueryEngine breaks into sub-queries
- Multi-Document Agent reasoning across contracts
- Graph-enhanced retrieval for related amendments

## Performance Targets

### POC Success Metrics
- **Document Processing**: <30 seconds per document, >95% success rate
- **Search Response**: <3 seconds for hybrid search
- **WebSocket Stability**: Reliable connections during multi-user testing
- **System Uptime**: Stable operation during legal professional demos

### Resource Requirements
- **GPU Memory**: 8GB minimum for mxbai-embed-large
- **System RAM**: 16GB recommended for concurrent processing
- **Storage**: Named Docker volumes for data persistence
- **Network**: Local development environment initially

## Security & Multi-tenancy (Future Considerations)

### Authentication & Authorization (Post-POC)
- User authentication with case access control
- Document isolation by user_id in MongoDB
- Role-based access (lawyers, paralegals, admins)
- Audit trail for document access and modifications

### Data Privacy
- Local deployment for sensitive legal documents
- No external API calls for document content
- Encrypted storage for production deployment
- Compliance with legal industry data requirements

## Deployment Strategy

### POC Development Environment
- Local Python virtual environment
- Docker containers for databases and services
- Hot-reload for rapid development iteration
- Console-based debugging and monitoring

### Production Considerations (Future)
- Container orchestration (Kubernetes/Docker Swarm)
- Horizontal scaling for multiple law firms
- Load balancing for WebSocket connections
- Backup and disaster recovery procedures
- Monitoring and alerting systems

---

**Last Updated**: Initial Architecture Specification
**Version**: 1.0 POC
**Next Review**: After Phase 1 completion