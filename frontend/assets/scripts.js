/**
 * =============================================================================
 * Patexia Legal AI Chatbot - Frontend JavaScript
 * 
 * This module provides client-side functionality for the Gradio-based
 * legal AI interface including:
 * - Real-time WebSocket communication for progress updates
 * - Document upload handling with progress tracking
 * - Search interface enhancements and debouncing
 * - Case management and visual markers
 * - Legal document viewer with highlighting
 * - Keyboard shortcuts for legal professionals
 * - Performance monitoring and error handling
 * - Accessibility features for legal compliance
 * =============================================================================
 */

// =============================================================================
// GLOBAL CONFIGURATION AND CONSTANTS
// =============================================================================

const LEGAL_AI_CONFIG = {
    // WebSocket configuration
    websocket: {
        url: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
        reconnectInterval: 3000,
        maxReconnectAttempts: 10,
        heartbeatInterval: 30000
    },

    // API endpoints
    api: {
        base: '/api',
        upload: '/api/upload',
        search: '/api/search',
        cases: '/api/cases',
        documents: '/api/documents',
        health: '/api/health'
    },

    // UI configuration
    ui: {
        searchDebounceMs: 300,
        progressUpdateInterval: 500,
        autoRefreshInterval: 5000,
        maxFileSize: 100 * 1024 * 1024, // 100MB
        supportedFileTypes: ['.pdf', '.txt', '.docx'],
        maxSearchHistory: 100
    },

    // Legal-specific settings
    legal: {
        caseColors: [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ],
        highlightStyle: 'mark',
        citationContextChars: 200,
        maxContentPreview: 500
    }
};

// Global state management
let legalAIState = {
    currentCase: null,
    searchHistory: [],
    uploadProgress: {},
    websocketConnection: null,
    reconnectAttempts: 0,
    isOnline: navigator.onLine,
    performanceMetrics: {
        searchTimes: [],
        uploadTimes: [],
        renderTimes: []
    }
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Debounce function for search input optimization
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

/**
 * Throttle function for performance optimization
 */
function throttle(func, limit) {
    let inThrottle;
    return function () {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Generate correlation ID for request tracking
 */
function generateCorrelationId() {
    return 'req_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

/**
 * Sanitize HTML content for security
 */
function sanitizeHtml(str) {
    const temp = document.createElement('div');
    temp.textContent = str;
    return temp.innerHTML;
}

/**
 * Format date for legal document timestamps
 */
function formatLegalDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// =============================================================================
// WEBSOCKET COMMUNICATION
// =============================================================================

class LegalAIWebSocket {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.heartbeatTimer = null;
        this.reconnectTimer = null;
        this.messageQueue = [];
        this.eventHandlers = new Map();
    }

    connect() {
        try {
            console.log('Connecting to WebSocket:', LEGAL_AI_CONFIG.websocket.url);
            this.ws = new WebSocket(LEGAL_AI_CONFIG.websocket.url);

            this.ws.onopen = this.onOpen.bind(this);
            this.ws.onmessage = this.onMessage.bind(this);
            this.ws.onclose = this.onClose.bind(this);
            this.ws.onerror = this.onError.bind(this);

        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.scheduleReconnect();
        }
    }

    onOpen(event) {
        console.log('WebSocket connected');
        this.isConnected = true;
        legalAIState.reconnectAttempts = 0;

        // Start heartbeat
        this.startHeartbeat();

        // Send queued messages
        this.processMessageQueue();

        // Update UI connection status
        this.updateConnectionStatus(true);

        // Emit connection event
        this.emit('connected', { timestamp: Date.now() });
    }

    onMessage(event) {
        try {
            const message = JSON.parse(event.data);
            console.log('WebSocket message received:', message);

            // Handle different message types
            switch (message.type) {
                case 'progress':
                    this.handleProgressUpdate(message);
                    break;
                case 'document_processed':
                    this.handleDocumentProcessed(message);
                    break;
                case 'search_completed':
                    this.handleSearchCompleted(message);
                    break;
                case 'case_updated':
                    this.handleCaseUpdated(message);
                    break;
                case 'error':
                    this.handleError(message);
                    break;
                case 'pong':
                    // Heartbeat response
                    break;
                default:
                    console.warn('Unknown message type:', message.type);
            }

            // Emit generic message event
            this.emit('message', message);

        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    onClose(event) {
        console.log('WebSocket closed:', event.code, event.reason);
        this.isConnected = false;
        this.stopHeartbeat();
        this.updateConnectionStatus(false);

        // Attempt reconnection unless it was a clean close
        if (event.code !== 1000) {
            this.scheduleReconnect();
        }

        this.emit('disconnected', { code: event.code, reason: event.reason });
    }

    onError(error) {
        console.error('WebSocket error:', error);
        this.emit('error', error);
    }

    send(message) {
        if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
            const messageStr = JSON.stringify(message);
            this.ws.send(messageStr);
            console.log('WebSocket message sent:', message);
        } else {
            // Queue message for later
            this.messageQueue.push(message);
            console.log('WebSocket not connected, message queued:', message);
        }
    }

    processMessageQueue() {
        while (this.messageQueue.length > 0 && this.isConnected) {
            const message = this.messageQueue.shift();
            this.send(message);
        }
    }

    startHeartbeat() {
        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected) {
                this.send({ type: 'ping', timestamp: Date.now() });
            }
        }, LEGAL_AI_CONFIG.websocket.heartbeatInterval);
    }

    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }

    scheduleReconnect() {
        if (legalAIState.reconnectAttempts < LEGAL_AI_CONFIG.websocket.maxReconnectAttempts) {
            legalAIState.reconnectAttempts++;
            const delay = LEGAL_AI_CONFIG.websocket.reconnectInterval * Math.pow(2, legalAIState.reconnectAttempts - 1);

            console.log(`Scheduling reconnect attempt ${legalAIState.reconnectAttempts} in ${delay}ms`);

            this.reconnectTimer = setTimeout(() => {
                this.connect();
            }, delay);
        } else {
            console.error('Max reconnection attempts reached');
            this.emit('max_reconnects_reached');
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.querySelector('.connection-status');
        if (statusElement) {
            statusElement.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
            statusElement.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    // Event handling
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    emit(event, data) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    // Message handlers
    handleProgressUpdate(message) {
        const { operation_id, progress, status, details } = message.data;

        // Update progress UI
        this.updateProgressBar(operation_id, progress, status, details);

        // Store progress state
        legalAIState.uploadProgress[operation_id] = {
            progress,
            status,
            details,
            timestamp: Date.now()
        };

        this.emit('progress', message.data);
    }

    handleDocumentProcessed(message) {
        const { document_id, case_id, status, metadata } = message.data;

        console.log('Document processed:', document_id, status);

        // Update document list
        this.refreshDocumentList(case_id);

        // Show notification
        this.showNotification(`Document processed: ${metadata.document_name}`, 'success');

        this.emit('document_processed', message.data);
    }

    handleSearchCompleted(message) {
        const { query, results, search_time, total_results } = message.data;

        console.log('Search completed:', query, results.length, 'results in', search_time, 'ms');

        // Update search results
        this.updateSearchResults(results, total_results, search_time);

        // Track performance
        legalAIState.performanceMetrics.searchTimes.push(search_time);

        this.emit('search_completed', message.data);
    }

    handleCaseUpdated(message) {
        const { case_id, updates } = message.data;

        console.log('Case updated:', case_id, updates);

        // Refresh case data if it's the current case
        if (legalAIState.currentCase === case_id) {
            this.refreshCaseData(case_id);
        }

        this.emit('case_updated', message.data);
    }

    handleError(message) {
        const { error_type, error_message, context } = message.data;

        console.error('WebSocket error:', error_type, error_message);

        // Show error notification
        this.showNotification(`Error: ${error_message}`, 'error');

        this.emit('websocket_error', message.data);
    }

    updateProgressBar(operationId, progress, status, details) {
        const progressElement = document.querySelector(`[data-operation-id="${operationId}"]`);
        if (progressElement) {
            const progressBar = progressElement.querySelector('.progress-bar');
            const progressText = progressElement.querySelector('.progress-text');
            const statusText = progressElement.querySelector('.status-text');

            if (progressBar) {
                progressBar.style.width = `${progress}%`;
                progressBar.setAttribute('aria-valuenow', progress);
            }

            if (progressText) {
                progressText.textContent = `${Math.round(progress)}%`;
            }

            if (statusText) {
                statusText.textContent = status;
            }

            // Update details if provided
            if (details && details.stage) {
                const detailsElement = progressElement.querySelector('.progress-details');
                if (detailsElement) {
                    detailsElement.textContent = details.stage;
                }
            }
        }
    }

    showNotification(message, type = 'info', duration = 5000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${sanitizeHtml(message)}</span>
                <button class="notification-close" aria-label="Close notification">&times;</button>
            </div>
        `;

        // Add to container
        let container = document.querySelector('.notifications-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'notifications-container';
            document.body.appendChild(container);
        }

        container.appendChild(notification);

        // Close button handler
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            this.removeNotification(notification);
        });

        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                this.removeNotification(notification);
            }, duration);
        }

        // Animate in
        requestAnimationFrame(() => {
            notification.classList.add('notification-show');
        });
    }

    removeNotification(notification) {
        notification.classList.add('notification-hide');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    close() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }
        this.stopHeartbeat();

        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
        }
    }
}

// =============================================================================
// DOCUMENT UPLOAD HANDLING
// =============================================================================

class DocumentUploadHandler {
    constructor() {
        this.activeUploads = new Map();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // File input change handlers
        document.addEventListener('change', (event) => {
            if (event.target.matches('input[type="file"]')) {
                this.handleFileSelection(event);
            }
        });

        // Drag and drop handlers
        document.addEventListener('dragover', this.handleDragOver.bind(this));
        document.addEventListener('drop', this.handleDrop.bind(this));
        document.addEventListener('dragenter', this.handleDragEnter.bind(this));
        document.addEventListener('dragleave', this.handleDragLeave.bind(this));
    }

    handleFileSelection(event) {
        const files = Array.from(event.target.files);
        this.processFiles(files);
    }

    handleDragOver(event) {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
    }

    handleDragEnter(event) {
        event.preventDefault();
        if (this.isValidDropTarget(event.target)) {
            event.target.classList.add('drag-over');
        }
    }

    handleDragLeave(event) {
        event.preventDefault();
        if (this.isValidDropTarget(event.target)) {
            event.target.classList.remove('drag-over');
        }
    }

    handleDrop(event) {
        event.preventDefault();

        if (this.isValidDropTarget(event.target)) {
            event.target.classList.remove('drag-over');

            const files = Array.from(event.dataTransfer.files);
            this.processFiles(files);
        }
    }

    isValidDropTarget(element) {
        return element.classList.contains('drop-zone') ||
            element.closest('.drop-zone') !== null;
    }

    processFiles(files) {
        const validFiles = files.filter(file => this.validateFile(file));

        if (validFiles.length !== files.length) {
            const invalidCount = files.length - validFiles.length;
            websocketManager.showNotification(
                `${invalidCount} file(s) rejected. Only PDF, TXT, and DOCX files under ${formatFileSize(LEGAL_AI_CONFIG.ui.maxFileSize)} are allowed.`,
                'warning'
            );
        }

        validFiles.forEach(file => this.uploadFile(file));
    }

    validateFile(file) {
        // Check file size
        if (file.size > LEGAL_AI_CONFIG.ui.maxFileSize) {
            console.warn(`File too large: ${file.name} (${formatFileSize(file.size)})`);
            return false;
        }

        // Check file type
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!LEGAL_AI_CONFIG.ui.supportedFileTypes.includes(fileExtension)) {
            console.warn(`Unsupported file type: ${file.name}`);
            return false;
        }

        return true;
    }

    async uploadFile(file) {
        const operationId = generateCorrelationId();
        const startTime = Date.now();

        try {
            // Create progress tracking
            this.createProgressUI(operationId, file.name);

            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('case_id', legalAIState.currentCase);
            formData.append('operation_id', operationId);

            // Upload with progress tracking
            const response = await this.uploadWithProgress(formData, operationId);

            if (response.ok) {
                const result = await response.json();

                // Calculate upload time
                const uploadTime = Date.now() - startTime;
                legalAIState.performanceMetrics.uploadTimes.push(uploadTime);

                console.log('Upload successful:', result);
                websocketManager.showNotification(`Upload completed: ${file.name}`, 'success');

                // Document processing will be tracked via WebSocket

            } else {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

        } catch (error) {
            console.error('Upload error:', error);
            websocketManager.showNotification(`Upload failed: ${file.name} - ${error.message}`, 'error');
            this.removeProgressUI(operationId);
        }
    }

    async uploadWithProgress(formData, operationId) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            // Progress tracking
            xhr.upload.addEventListener('progress', (event) => {
                if (event.lengthComputable) {
                    const progress = (event.loaded / event.total) * 100;
                    this.updateUploadProgress(operationId, progress, 'uploading');
                }
            });

            // Handle completion
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    this.updateUploadProgress(operationId, 100, 'processing');
                    resolve(xhr);
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            });

            // Handle errors
            xhr.addEventListener('error', () => {
                reject(new Error('Network error'));
            });

            xhr.addEventListener('abort', () => {
                reject(new Error('Upload cancelled'));
            });

            // Send request
            xhr.open('POST', LEGAL_AI_CONFIG.api.upload);
            xhr.setRequestHeader('X-Correlation-ID', operationId);
            xhr.send(formData);

            // Store XHR for potential cancellation
            this.activeUploads.set(operationId, xhr);
        });
    }

    createProgressUI(operationId, fileName) {
        const progressContainer = document.querySelector('.upload-progress-container');
        if (!progressContainer) return;

        const progressElement = document.createElement('div');
        progressElement.className = 'upload-progress-item';
        progressElement.setAttribute('data-operation-id', operationId);
        progressElement.innerHTML = `
            <div class="upload-info">
                <span class="file-name">${sanitizeHtml(fileName)}</span>
                <button class="cancel-upload" data-operation-id="${operationId}" aria-label="Cancel upload">
                    <span>&times;</span>
                </button>
            </div>
            <div class="progress-container">
                <div class="progress-bar" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>
                <div class="progress-text">0%</div>
            </div>
            <div class="status-text">Preparing...</div>
            <div class="progress-details"></div>
        `;

        progressContainer.appendChild(progressElement);

        // Cancel button handler
        const cancelBtn = progressElement.querySelector('.cancel-upload');
        cancelBtn.addEventListener('click', () => this.cancelUpload(operationId));
    }

    updateUploadProgress(operationId, progress, status) {
        websocketManager.updateProgressBar(operationId, progress, status, null);
    }

    cancelUpload(operationId) {
        const xhr = this.activeUploads.get(operationId);
        if (xhr) {
            xhr.abort();
            this.activeUploads.delete(operationId);
        }

        this.removeProgressUI(operationId);
        websocketManager.showNotification('Upload cancelled', 'info');
    }

    removeProgressUI(operationId) {
        const progressElement = document.querySelector(`[data-operation-id="${operationId}"]`);
        if (progressElement) {
            progressElement.classList.add('removing');
            setTimeout(() => {
                if (progressElement.parentNode) {
                    progressElement.parentNode.removeChild(progressElement);
                }
            }, 300);
        }

        this.activeUploads.delete(operationId);
    }
}

// =============================================================================
// SEARCH FUNCTIONALITY
// =============================================================================

class SearchManager {
    constructor() {
        this.searchCache = new Map();
        this.searchHistory = JSON.parse(localStorage.getItem('legalai_search_history') || '[]');
        this.setupSearchHandlers();
    }

    setupSearchHandlers() {
        // Search input with debouncing
        const searchInputs = document.querySelectorAll('.search-input');
        searchInputs.forEach(input => {
            const debouncedSearch = debounce(this.performSearch.bind(this), LEGAL_AI_CONFIG.ui.searchDebounceMs);
            input.addEventListener('input', debouncedSearch);
            input.addEventListener('keydown', this.handleSearchKeydown.bind(this));
        });

        // Search suggestions
        document.addEventListener('click', this.handleSearchSuggestionClick.bind(this));
    }

    handleSearchKeydown(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            this.performSearch(event);
        } else if (event.key === 'Escape') {
            this.hideSuggestions();
        }
    }

    async performSearch(event) {
        const query = event.target.value.trim();

        if (!query || query.length < 2) {
            this.clearSearchResults();
            return;
        }

        const startTime = Date.now();

        try {
            // Check cache first
            const cacheKey = this.generateCacheKey(query);
            if (this.searchCache.has(cacheKey)) {
                const cachedResult = this.searchCache.get(cacheKey);
                this.displaySearchResults(cachedResult.results, cachedResult.total, cachedResult.searchTime);
                return;
            }

            // Show loading state
            this.showSearchLoading();

            // Perform search
            const searchParams = {
                query: query,
                case_id: legalAIState.currentCase,
                search_type: 'hybrid',
                limit: 15,
                correlation_id: generateCorrelationId()
            };

            const response = await fetch(LEGAL_AI_CONFIG.api.search, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Correlation-ID': searchParams.correlation_id
                },
                body: JSON.stringify(searchParams)
            });

            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }

            const result = await response.json();
            const searchTime = Date.now() - startTime;

            // Cache result
            this.searchCache.set(cacheKey, {
                results: result.results,
                total: result.total_results,
                searchTime: searchTime,
                timestamp: Date.now()
            });

            // Update search history
            this.addToSearchHistory(query);

            // Display results
            this.displaySearchResults(result.results, result.total_results, searchTime);

            // Track performance
            legalAIState.performanceMetrics.searchTimes.push(searchTime);

        } catch (error) {
            console.error('Search error:', error);
            this.showSearchError(error.message);
        }
    }

    generateCacheKey(query) {
        return `${legalAIState.currentCase}_${query.toLowerCase()}`;
    }

    showSearchLoading() {
        const resultsContainer = document.querySelector('.search-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="search-loading">
                    <div class="spinner"></div>
                    <span>Searching documents...</span>
                </div>
            `;
        }
    }

    displaySearchResults(results, totalResults, searchTime) {
        const resultsContainer = document.querySelector('.search-results');
        if (!resultsContainer) return;

        if (results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="no-results">
                    <p>No documents found matching your search.</p>
                    <p>Try different keywords or check your spelling.</p>
                </div>
            `;
            return;
        }

        let html = `
            <div class="search-header">
                <h3>Search Results</h3>
                <div class="search-stats">
                    ${totalResults} results found in ${searchTime}ms
                </div>
            </div>
            <div class="results-list">
        `;

        results.forEach((result, index) => {
            html += this.renderSearchResult(result, index);
        });

        html += '</div>';

        if (results.length < totalResults) {
            html += `
                <div class="load-more-container">
                    <button class="load-more-results" data-offset="${results.length}">
                        Load More Results (${totalResults - results.length} remaining)
                    </button>
                </div>
            `;
        }

        resultsContainer.innerHTML = html;

        // Setup result click handlers
        this.setupResultClickHandlers();
    }

    renderSearchResult(result, index) {
        const relevancePercent = Math.round(result.relevance_score * 100);
        const highlightedContent = result.highlighted_content || result.content;

        return `
            <div class="search-result-item" data-result-index="${index}" data-document-id="${result.document_id}" data-chunk-id="${result.chunk_id}">
                <div class="result-header">
                    <div class="result-title">
                        <h4>${sanitizeHtml(result.document_name)}</h4>
                        <span class="relevance-score" title="Relevance: ${relevancePercent}%">
                            ${relevancePercent}%
                        </span>
                    </div>
                    <div class="result-metadata">
                        ${result.section_title ? `<span class="section">${sanitizeHtml(result.section_title)}</span>` : ''}
                        ${result.page_number ? `<span class="page">Page ${result.page_number}</span>` : ''}
                        ${result.legal_citations.length > 0 ? `<span class="citations">${result.legal_citations.length} citations</span>` : ''}
                    </div>
                </div>
                <div class="result-content">
                    <div class="content-preview">${highlightedContent}</div>
                    ${result.legal_citations.length > 0 ? this.renderCitations(result.legal_citations) : ''}
                </div>
                <div class="result-actions">
                    <button class="view-document" data-document-id="${result.document_id}" data-chunk-id="${result.chunk_id}">
                        View Document
                    </button>
                    <button class="view-context" data-document-id="${result.document_id}" data-chunk-id="${result.chunk_id}">
                        View in Context
                    </button>
                    <button class="copy-citation" data-result-index="${index}">
                        Copy Citation
                    </button>
                </div>
            </div>
        `;
    }

    renderCitations(citations) {
        return `
            <div class="legal-citations">
                <strong>Legal Citations:</strong>
                <ul>
                    ${citations.map(citation => `<li>${sanitizeHtml(citation)}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    setupResultClickHandlers() {
        // View document handlers
        document.querySelectorAll('.view-document').forEach(btn => {
            btn.addEventListener('click', this.handleViewDocument.bind(this));
        });

        // View context handlers
        document.querySelectorAll('.view-context').forEach(btn => {
            btn.addEventListener('click', this.handleViewContext.bind(this));
        });

        // Copy citation handlers
        document.querySelectorAll('.copy-citation').forEach(btn => {
            btn.addEventListener('click', this.handleCopyCitation.bind(this));
        });

        // Load more results handler
        const loadMoreBtn = document.querySelector('.load-more-results');
        if (loadMoreBtn) {
            loadMoreBtn.addEventListener('click', this.handleLoadMore.bind(this));
        }
    }

    handleViewDocument(event) {
        const documentId = event.target.dataset.documentId;
        const chunkId = event.target.dataset.chunkId;

        // Trigger document viewer
        this.openDocumentViewer(documentId, chunkId);
    }

    handleViewContext(event) {
        const documentId = event.target.dataset.documentId;
        const chunkId = event.target.dataset.chunkId;

        // Open document with context highlighting
        this.openDocumentViewer(documentId, chunkId, true);
    }

    handleCopyCitation(event) {
        const resultIndex = parseInt(event.target.dataset.resultIndex);
        const resultElement = document.querySelector(`[data-result-index="${resultIndex}"]`);

        if (resultElement) {
            const documentName = resultElement.querySelector('.result-title h4').textContent;
            const section = resultElement.querySelector('.section')?.textContent || '';
            const page = resultElement.querySelector('.page')?.textContent || '';

            const citation = this.formatCitation(documentName, section, page);

            navigator.clipboard.writeText(citation).then(() => {
                websocketManager.showNotification('Citation copied to clipboard', 'success', 2000);
            }).catch(() => {
                // Fallback for older browsers
                this.fallbackCopyToClipboard(citation);
            });
        }
    }

    formatCitation(documentName, section, page) {
        let citation = documentName;
        if (section) citation += `, ${section}`;
        if (page) citation += `, ${page}`;
        citation += ` (accessed ${formatLegalDate(new Date())})`;
        return citation;
    }

    fallbackCopyToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        try {
            document.execCommand('copy');
            websocketManager.showNotification('Citation copied to clipboard', 'success', 2000);
        } catch (err) {
            websocketManager.showNotification('Failed to copy citation', 'error');
        }

        document.body.removeChild(textArea);
    }

    openDocumentViewer(documentId, chunkId, showContext = false) {
        // This would interface with the Gradio document viewer component
        // For now, we'll emit an event that the Gradio interface can listen to

        const event = new CustomEvent('openDocument', {
            detail: {
                documentId: documentId,
                chunkId: chunkId,
                showContext: showContext
            }
        });

        document.dispatchEvent(event);
    }

    addToSearchHistory(query) {
        // Remove existing occurrence
        this.searchHistory = this.searchHistory.filter(item => item.query !== query);

        // Add to beginning
        this.searchHistory.unshift({
            query: query,
            timestamp: Date.now(),
            case_id: legalAIState.currentCase
        });

        // Limit history size
        if (this.searchHistory.length > LEGAL_AI_CONFIG.ui.maxSearchHistory) {
            this.searchHistory = this.searchHistory.slice(0, LEGAL_AI_CONFIG.ui.maxSearchHistory);
        }

        // Save to localStorage
        localStorage.setItem('legalai_search_history', JSON.stringify(this.searchHistory));

        // Update search history UI
        this.updateSearchHistoryUI();
    }

    updateSearchHistoryUI() {
        const historyContainer = document.querySelector('.search-history');
        if (!historyContainer) return;

        const recentQueries = this.searchHistory
            .filter(item => item.case_id === legalAIState.currentCase)
            .slice(0, 10);

        if (recentQueries.length === 0) {
            historyContainer.innerHTML = '<p class="no-history">No recent searches</p>';
            return;
        }

        let html = '<h4>Recent Searches</h4><ul class="history-list">';

        recentQueries.forEach(item => {
            html += `
                <li class="history-item">
                    <button class="history-query" data-query="${sanitizeHtml(item.query)}">
                        ${sanitizeHtml(item.query)}
                    </button>
                    <span class="history-time">${this.formatRelativeTime(item.timestamp)}</span>
                </li>
            `;
        });

        html += '</ul>';
        historyContainer.innerHTML = html;

        // Setup click handlers
        historyContainer.querySelectorAll('.history-query').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const query = e.target.dataset.query;
                const searchInput = document.querySelector('.search-input');
                if (searchInput) {
                    searchInput.value = query;
                    searchInput.dispatchEvent(new Event('input'));
                }
            });
        });
    }

    formatRelativeTime(timestamp) {
        const now = Date.now();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (minutes < 1) return 'just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        return `${days}d ago`;
    }

    showSearchError(message) {
        const resultsContainer = document.querySelector('.search-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="search-error">
                    <p>Search failed: ${sanitizeHtml(message)}</p>
                    <button class="retry-search">Retry Search</button>
                </div>
            `;

            const retryBtn = resultsContainer.querySelector('.retry-search');
            if (retryBtn) {
                retryBtn.addEventListener('click', () => {
                    const searchInput = document.querySelector('.search-input');
                    if (searchInput && searchInput.value) {
                        this.performSearch({ target: searchInput });
                    }
                });
            }
        }
    }

    clearSearchResults() {
        const resultsContainer = document.querySelector('.search-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = '';
        }
    }
}

// =============================================================================
// CASE MANAGEMENT
// =============================================================================

class CaseManager {
    constructor() {
        this.cases = new Map();
        this.setupCaseHandlers();
        this.loadCases();
    }

    setupCaseHandlers() {
        // Case selection handlers
        document.addEventListener('click', (event) => {
            if (event.target.matches('.case-item') || event.target.closest('.case-item')) {
                this.handleCaseSelection(event);
            }
        });

        // Case creation handlers
        const createCaseBtn = document.querySelector('.create-case-btn');
        if (createCaseBtn) {
            createCaseBtn.addEventListener('click', this.showCreateCaseDialog.bind(this));
        }
    }

    async loadCases() {
        try {
            const response = await fetch(LEGAL_AI_CONFIG.api.cases);
            if (!response.ok) {
                throw new Error(`Failed to load cases: ${response.statusText}`);
            }

            const cases = await response.json();
            this.updateCasesList(cases);

            // Set current case if none selected
            if (!legalAIState.currentCase && cases.length > 0) {
                this.selectCase(cases[0].case_id);
            }

        } catch (error) {
            console.error('Error loading cases:', error);
            websocketManager.showNotification('Failed to load cases', 'error');
        }
    }

    updateCasesList(cases) {
        const casesContainer = document.querySelector('.cases-list');
        if (!casesContainer) return;

        if (cases.length === 0) {
            casesContainer.innerHTML = `
                <div class="no-cases">
                    <p>No cases found.</p>
                    <button class="create-first-case">Create Your First Case</button>
                </div>
            `;
            return;
        }

        let html = '';
        cases.forEach((caseData, index) => {
            const colorIndex = index % LEGAL_AI_CONFIG.legal.caseColors.length;
            const color = LEGAL_AI_CONFIG.legal.caseColors[colorIndex];
            const isActive = caseData.case_id === legalAIState.currentCase;

            html += `
                <div class="case-item ${isActive ? 'active' : ''}" data-case-id="${caseData.case_id}">
                    <div class="case-marker" style="background-color: ${color}"></div>
                    <div class="case-content">
                        <h4 class="case-name">${sanitizeHtml(caseData.case_name)}</h4>
                        <p class="case-summary">${sanitizeHtml(caseData.initial_summary || 'No summary available')}</p>
                        <div class="case-metadata">
                            <span class="document-count">${caseData.document_count || 0} documents</span>
                            <span class="last-updated">${formatLegalDate(caseData.updated_at)}</span>
                        </div>
                    </div>
                    <div class="case-actions">
                        <button class="case-action-btn" data-action="edit" data-case-id="${caseData.case_id}" title="Edit case">
                            ✏️
                        </button>
                        <button class="case-action-btn" data-action="delete" data-case-id="${caseData.case_id}" title="Delete case">
                            🗑️
                        </button>
                    </div>
                </div>
            `;
        });

        casesContainer.innerHTML = html;

        // Store cases data
        cases.forEach(caseData => {
            this.cases.set(caseData.case_id, caseData);
        });
    }

    handleCaseSelection(event) {
        const caseItem = event.target.closest('.case-item');
        if (!caseItem) return;

        const caseId = caseItem.dataset.caseId;

        // Handle action buttons
        if (event.target.matches('.case-action-btn')) {
            const action = event.target.dataset.action;
            if (action === 'edit') {
                this.showEditCaseDialog(caseId);
            } else if (action === 'delete') {
                this.showDeleteCaseDialog(caseId);
            }
            return;
        }

        // Select case
        this.selectCase(caseId);
    }

    selectCase(caseId) {
        if (legalAIState.currentCase === caseId) return;

        // Update UI
        document.querySelectorAll('.case-item').forEach(item => {
            item.classList.remove('active');
        });

        const selectedCase = document.querySelector(`[data-case-id="${caseId}"]`);
        if (selectedCase) {
            selectedCase.classList.add('active');
        }

        // Update state
        legalAIState.currentCase = caseId;

        // Clear search results and cache for the new case
        if (window.searchManager) {
            window.searchManager.clearSearchResults();
            window.searchManager.searchCache.clear();
        }

        // Notify WebSocket of case change
        if (websocketManager && websocketManager.isConnected) {
            websocketManager.send({
                type: 'case_selected',
                data: { case_id: caseId }
            });
        }

        // Update document list for the new case
        this.loadDocumentsForCase(caseId);

        // Emit case change event
        const event = new CustomEvent('caseChanged', {
            detail: { caseId: caseId, caseData: this.cases.get(caseId) }
        });
        document.dispatchEvent(event);

        console.log('Case selected:', caseId);
    }

    async loadDocumentsForCase(caseId) {
        try {
            const response = await fetch(`${LEGAL_AI_CONFIG.api.documents}?case_id=${caseId}`);
            if (!response.ok) {
                throw new Error(`Failed to load documents: ${response.statusText}`);
            }

            const documents = await response.json();
            this.updateDocumentsList(documents);

        } catch (error) {
            console.error('Error loading documents:', error);
        }
    }

    updateDocumentsList(documents) {
        const documentsContainer = document.querySelector('.documents-list');
        if (!documentsContainer) return;

        if (documents.length === 0) {
            documentsContainer.innerHTML = `
                <div class="no-documents">
                    <p>No documents in this case yet.</p>
                    <p>Upload documents to get started.</p>
                </div>
            `;
            return;
        }

        let html = '<h4>Case Documents</h4><div class="documents-grid">';

        documents.forEach(doc => {
            const statusClass = doc.processing_status || 'unknown';
            const fileSize = formatFileSize(doc.file_size || 0);

            html += `
                <div class="document-item ${statusClass}" data-document-id="${doc.document_id}">
                    <div class="document-icon">
                        ${this.getFileIcon(doc.file_type)}
                    </div>
                    <div class="document-info">
                        <h5 class="document-name" title="${sanitizeHtml(doc.document_name)}">
                            ${sanitizeHtml(doc.document_name)}
                        </h5>
                        <div class="document-metadata">
                            <span class="file-size">${fileSize}</span>
                            <span class="file-type">${doc.file_type?.toUpperCase() || 'Unknown'}</span>
                            <span class="status ${statusClass}">${this.getStatusText(statusClass)}</span>
                        </div>
                        <div class="document-date">${formatLegalDate(doc.created_at)}</div>
                    </div>
                    <div class="document-actions">
                        <button class="document-action-btn" data-action="view" data-document-id="${doc.document_id}" title="View document">
                            👁️
                        </button>
                        <button class="document-action-btn" data-action="download" data-document-id="${doc.document_id}" title="Download document">
                            ⬇️
                        </button>
                        <button class="document-action-btn" data-action="delete" data-document-id="${doc.document_id}" title="Delete document">
                            🗑️
                        </button>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        documentsContainer.innerHTML = html;

        // Setup document action handlers
        this.setupDocumentActionHandlers();
    }

    getFileIcon(fileType) {
        switch (fileType?.toLowerCase()) {
            case 'pdf': return '📄';
            case 'txt': return '📝';
            case 'docx':
            case 'doc': return '📘';
            default: return '📄';
        }
    }

    getStatusText(status) {
        switch (status) {
            case 'completed': return 'Ready';
            case 'processing': return 'Processing';
            case 'failed': return 'Failed';
            case 'pending': return 'Pending';
            default: return 'Unknown';
        }
    }

    setupDocumentActionHandlers() {
        document.querySelectorAll('.document-action-btn').forEach(btn => {
            btn.addEventListener('click', this.handleDocumentAction.bind(this));
        });
    }

    handleDocumentAction(event) {
        const action = event.target.dataset.action;
        const documentId = event.target.dataset.documentId;

        switch (action) {
            case 'view':
                this.viewDocument(documentId);
                break;
            case 'download':
                this.downloadDocument(documentId);
                break;
            case 'delete':
                this.showDeleteDocumentDialog(documentId);
                break;
        }
    }

    viewDocument(documentId) {
        const event = new CustomEvent('viewDocument', {
            detail: { documentId: documentId }
        });
        document.dispatchEvent(event);
    }

    downloadDocument(documentId) {
        window.open(`${LEGAL_AI_CONFIG.api.documents}/${documentId}/download`, '_blank');
    }

    showCreateCaseDialog(event) {
        // This would trigger a Gradio modal or interface
        const event = new CustomEvent('showCreateCaseDialog');
        document.dispatchEvent(event);
    }

    showEditCaseDialog(caseId) {
        const event = new CustomEvent('showEditCaseDialog', {
            detail: { caseId: caseId }
        });
        document.dispatchEvent(event);
    }

    showDeleteCaseDialog(caseId) {
        const event = new CustomEvent('showDeleteCaseDialog', {
            detail: { caseId: caseId }
        });
        document.dispatchEvent(event);
    }

    showDeleteDocumentDialog(documentId) {
        const event = new CustomEvent('showDeleteDocumentDialog', {
            detail: { documentId: documentId }
        });
        document.dispatchEvent(event);
    }
}

// =============================================================================
// KEYBOARD SHORTCUTS
// =============================================================================

class KeyboardShortcuts {
    constructor() {
        this.shortcuts = new Map([
            ['ctrl+k', this.focusSearch.bind(this)],
            ['ctrl+n', this.createNewCase.bind(this)],
            ['ctrl+u', this.focusUpload.bind(this)],
            ['escape', this.clearFocus.bind(this)],
            ['ctrl+/', this.showHelp.bind(this)],
            ['f3', this.findNext.bind(this)],
            ['shift+f3', this.findPrevious.bind(this)]
        ]);

        this.setupKeyboardHandlers();
    }

    setupKeyboardHandlers() {
        document.addEventListener('keydown', this.handleKeydown.bind(this));
    }

    handleKeydown(event) {
        // Don't trigger shortcuts when typing in inputs
        if (event.target.matches('input, textarea, [contenteditable]')) {
            return;
        }

        const shortcut = this.getShortcutString(event);
        const handler = this.shortcuts.get(shortcut);

        if (handler) {
            event.preventDefault();
            handler(event);
        }
    }

    getShortcutString(event) {
        const parts = [];

        if (event.ctrlKey) parts.push('ctrl');
        if (event.shiftKey) parts.push('shift');
        if (event.altKey) parts.push('alt');
        if (event.metaKey) parts.push('meta');

        parts.push(event.key.toLowerCase());

        return parts.join('+');
    }

    focusSearch() {
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }

    createNewCase() {
        if (window.caseManager) {
            window.caseManager.showCreateCaseDialog();
        }
    }

    focusUpload() {
        const uploadInput = document.querySelector('input[type="file"]');
        if (uploadInput) {
            uploadInput.click();
        }
    }

    clearFocus() {
        if (document.activeElement) {
            document.activeElement.blur();
        }

        // Hide any open dropdowns or suggestions
        document.querySelectorAll('.dropdown-open, .suggestions-open').forEach(el => {
            el.classList.remove('dropdown-open', 'suggestions-open');
        });
    }

    showHelp() {
        const event = new CustomEvent('showKeyboardHelp');
        document.dispatchEvent(event);
    }

    findNext() {
        // Trigger find next in document viewer
        const event = new CustomEvent('findNext');
        document.dispatchEvent(event);
    }

    findPrevious() {
        // Trigger find previous in document viewer
        const event = new CustomEvent('findPrevious');
        document.dispatchEvent(event);
    }
}

// =============================================================================
// PERFORMANCE MONITORING
// =============================================================================

class PerformanceMonitor {
    constructor() {
        this.metrics = {
            pageLoadTime: 0,
            searchTimes: [],
            uploadTimes: [],
            renderTimes: [],
            memoryUsage: [],
            networkRequests: []
        };

        this.startTime = performance.now();
        this.setupPerformanceTracking();
    }

    setupPerformanceTracking() {
        // Track page load time
        window.addEventListener('load', () => {
            this.metrics.pageLoadTime = performance.now() - this.startTime;
            console.log('Page load time:', this.metrics.pageLoadTime, 'ms');
        });

        // Track memory usage periodically
        if (performance.memory) {
            setInterval(() => {
                this.metrics.memoryUsage.push({
                    used: performance.memory.usedJSHeapSize,
                    total: performance.memory.totalJSHeapSize,
                    limit: performance.memory.jsHeapSizeLimit,
                    timestamp: Date.now()
                });

                // Keep only last 100 measurements
                if (this.metrics.memoryUsage.length > 100) {
                    this.metrics.memoryUsage.shift();
                }
            }, 30000); // Every 30 seconds
        }

        // Track network requests
        this.wrapFetch();
    }

    wrapFetch() {
        const originalFetch = window.fetch;

        window.fetch = async (...args) => {
            const startTime = performance.now();
            const url = args[0];

            try {
                const response = await originalFetch(...args);
                const endTime = performance.now();

                this.metrics.networkRequests.push({
                    url: url,
                    method: args[1]?.method || 'GET',
                    status: response.status,
                    duration: endTime - startTime,
                    timestamp: Date.now()
                });

                // Keep only last 50 requests
                if (this.metrics.networkRequests.length > 50) {
                    this.metrics.networkRequests.shift();
                }

                return response;

            } catch (error) {
                const endTime = performance.now();

                this.metrics.networkRequests.push({
                    url: url,
                    method: args[1]?.method || 'GET',
                    status: 'error',
                    duration: endTime - startTime,
                    error: error.message,
                    timestamp: Date.now()
                });

                throw error;
            }
        };
    }

    trackSearchTime(duration) {
        this.metrics.searchTimes.push(duration);
        if (this.metrics.searchTimes.length > 100) {
            this.metrics.searchTimes.shift();
        }
    }

    trackUploadTime(duration) {
        this.metrics.uploadTimes.push(duration);
        if (this.metrics.uploadTimes.length > 50) {
            this.metrics.uploadTimes.shift();
        }
    }

    trackRenderTime(duration) {
        this.metrics.renderTimes.push(duration);
        if (this.metrics.renderTimes.length > 100) {
            this.metrics.renderTimes.shift();
        }
    }

    getAverageSearchTime() {
        if (this.metrics.searchTimes.length === 0) return 0;
        return this.metrics.searchTimes.reduce((a, b) => a + b, 0) / this.metrics.searchTimes.length;
    }

    getAverageUploadTime() {
        if (this.metrics.uploadTimes.length === 0) return 0;
        return this.metrics.uploadTimes.reduce((a, b) => a + b, 0) / this.metrics.uploadTimes.length;
    }

    getPerformanceReport() {
        return {
            pageLoadTime: this.metrics.pageLoadTime,
            averageSearchTime: this.getAverageSearchTime(),
            averageUploadTime: this.getAverageUploadTime(),
            totalSearches: this.metrics.searchTimes.length,
            totalUploads: this.metrics.uploadTimes.length,
            memoryUsage: this.metrics.memoryUsage.slice(-1)[0],
            recentNetworkRequests: this.metrics.networkRequests.slice(-10)
        };
    }
}

// =============================================================================
// APPLICATION INITIALIZATION
// =============================================================================

class LegalAIApp {
    constructor() {
        this.isInitialized = false;
        this.components = {};
    }

    async initialize() {
        if (this.isInitialized) return;

        console.log('Initializing Patexia Legal AI Frontend...');

        try {
            // Initialize performance monitoring
            this.components.performanceMonitor = new PerformanceMonitor();
            window.performanceMonitor = this.components.performanceMonitor;

            // Initialize WebSocket manager
            this.components.websocketManager = new LegalAIWebSocket();
            window.websocketManager = this.components.websocketManager;

            // Initialize document upload handler
            this.components.uploadHandler = new DocumentUploadHandler();
            window.uploadHandler = this.components.uploadHandler;

            // Initialize search manager
            this.components.searchManager = new SearchManager();
            window.searchManager = this.components.searchManager;

            // Initialize case manager
            this.components.caseManager = new CaseManager();
            window.caseManager = this.components.caseManager;

            // Initialize keyboard shortcuts
            this.components.keyboardShortcuts = new KeyboardShortcuts();
            window.keyboardShortcuts = this.components.keyboardShortcuts;

            // Connect WebSocket
            this.components.websocketManager.connect();

            // Setup global error handling
            this.setupErrorHandling();

            // Setup online/offline detection
            this.setupNetworkDetection();

            // Setup accessibility features
            this.setupAccessibility();

            this.isInitialized = true;
            console.log('Patexia Legal AI Frontend initialized successfully');

            // Emit initialization complete event
            const event = new CustomEvent('legalAIInitialized', {
                detail: { components: Object.keys(this.components) }
            });
            document.dispatchEvent(event);

        } catch (error) {
            console.error('Failed to initialize Legal AI Frontend:', error);
            this.showInitializationError(error);
        }
    }

    setupErrorHandling() {
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            this.handleGlobalError(event.error);
        });

        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            this.handleGlobalError(event.reason);
        });
    }

    setupNetworkDetection() {
        window.addEventListener('online', () => {
            legalAIState.isOnline = true;
            websocketManager.showNotification('Connection restored', 'success', 3000);

            // Attempt to reconnect WebSocket if needed
            if (!this.components.websocketManager.isConnected) {
                this.components.websocketManager.connect();
            }
        });

        window.addEventListener('offline', () => {
            legalAIState.isOnline = false;
            websocketManager.showNotification('Connection lost. Some features may be unavailable.', 'warning', 0);
        });
    }

    setupAccessibility() {
        // Add focus indicators
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Tab') {
                document.body.classList.add('using-keyboard');
            }
        });

        document.addEventListener('mousedown', () => {
            document.body.classList.remove('using-keyboard');
        });

        // Announce important changes to screen readers
        this.setupAriaLiveRegions();
    }

    setupAriaLiveRegions() {
        // Create live regions for dynamic content updates
        const announcements = document.createElement('div');
        announcements.setAttribute('aria-live', 'polite');
        announcements.setAttribute('aria-atomic', 'true');
        announcements.className = 'sr-only';
        announcements.id = 'aria-announcements';
        document.body.appendChild(announcements);

        const status = document.createElement('div');
        status.setAttribute('aria-live', 'assertive');
        status.setAttribute('aria-atomic', 'true');
        status.className = 'sr-only';
        status.id = 'aria-status';
        document.body.appendChild(status);
    }

    announceToScreenReader(message, urgent = false) {
        const targetId = urgent ? 'aria-status' : 'aria-announcements';
        const target = document.getElementById(targetId);

        if (target) {
            target.textContent = message;

            // Clear after announcement
            setTimeout(() => {
                target.textContent = '';
            }, 1000);
        }
    }

    handleGlobalError(error) {
        // Log error for debugging
        console.error('Application error:', error);

        // Show user-friendly error message
        if (websocketManager) {
            websocketManager.showNotification(
                'An unexpected error occurred. Please refresh the page if the problem persists.',
                'error',
                10000
            );
        }

        // Report error to monitoring service (if configured)
        this.reportError(error);
    }

    reportError(error) {
        // This could send errors to a monitoring service
        // For now, we'll just log performance impact
        if (this.components.performanceMonitor) {
            console.warn('Error may impact performance:', error.message);
        }
    }

    showInitializationError(error) {
        const errorHtml = `
            <div class="initialization-error">
                <div class="error-container">
                    <h2>⚠️ Initialization Error</h2>
                    <p>Failed to initialize the Legal AI interface.</p>
                    <details>
                        <summary>Error Details</summary>
                        <pre>${sanitizeHtml(error.stack || error.message)}</pre>
                    </details>
                    <div class="error-actions">
                        <button onclick="window.location.reload()">Reload Page</button>
                        <button onclick="this.reportIssue()">Report Issue</button>
                    </div>
                </div>
            </div>
        `;

        document.body.innerHTML = errorHtml;
    }

    reportIssue() {
        const errorReport = {
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href,
            error: 'Initialization failure',
            performanceMetrics: this.components.performanceMonitor?.getPerformanceReport()
        };

        console.log('Error report:', errorReport);

        // In a real application, this would send the report to a support system
        alert('Error report logged. Please contact support if the issue persists.');
    }

    cleanup() {
        // Cleanup WebSocket connection
        if (this.components.websocketManager) {
            this.components.websocketManager.close();
        }

        // Clear intervals and timeouts
        Object.values(this.components).forEach(component => {
            if (component.cleanup && typeof component.cleanup === 'function') {
                component.cleanup();
            }
        });

        this.isInitialized = false;
        console.log('Legal AI Frontend cleaned up');
    }
}

// =============================================================================
// UTILITY EXTENSIONS AND HELPERS
// =============================================================================

/**
 * Extend String prototype with legal document utilities
 */
String.prototype.highlightLegalTerms = function (terms) {
    let result = this;
    terms.forEach(term => {
        const regex = new RegExp(`\\b${term.replace(/[.*+?^${}()|[\]\\]/g, '\\    showInitializationError(error) {
        const errorHtml = `
            <div class="initialization')}\\b`, 'gi');
    result = result.replace(regex, `<mark class="legal-term">    showInitializationError(error) {
        const errorHtml = `
        < div class= "initialization</mark>`);
    });
return result;
};

String.prototype.extractCitations = function () {
    // Common legal citation patterns
    const patterns = [
        /\b\d+\s+U\.S\.C?\.\s*§?\s*\d+/gi,           // U.S.C. citations
        /\b\d+\s+F\.\d+d?\s+\d+/gi,                  // Federal reporter citations
        /\b\d+\s+S\.Ct\.\s+\d+/gi,                   // Supreme Court citations
        /\bPub\.\s*L\.\s*No\.\s*\d+-\d+/gi,          // Public law citations
        /\b\d+\s+C\.F\.R\.?\s*§?\s*\d+/gi            // Code of Federal Regulations
    ];

    const citations = [];
    patterns.forEach(pattern => {
        const matches = this.match(pattern);
        if (matches) {
            citations.push(...matches);
        }
    });

    return [...new Set(citations)]; // Remove duplicates
};

/**
 * Document viewer integration helpers
 */
function createDocumentViewer(documentId, chunkId, container) {
    // This would create a document viewer component
    // Integration with Gradio document display components

    const viewer = document.createElement('div');
    viewer.className = 'document-viewer';
    viewer.innerHTML = `
        <div class="viewer-header">
            <div class="viewer-controls">
                <button class="zoom-out" title="Zoom out">−</button>
                <span class="zoom-level">100%</span>
                <button class="zoom-in" title="Zoom in">+</button>
                <button class="fit-width" title="Fit to width">⟷</button>
                <button class="fit-page" title="Fit to page">⤢</button>
            </div>
            <div class="viewer-search">
                <input type="text" class="viewer-search-input" placeholder="Search in document...">
                <button class="search-prev" title="Previous result">↑</button>
                <button class="search-next" title="Next result">↓</button>
            </div>
        </div>
        <div class="viewer-content" id="viewer-content-${documentId}">
            <div class="loading-spinner">Loading document...</div>
        </div>
        <div class="viewer-footer">
            <div class="page-info">
                <span class="current-page">1</span> / <span class="total-pages">--</span>
            </div>
        </div>
    `;

    container.appendChild(viewer);

    // Setup viewer controls
    setupDocumentViewerControls(viewer, documentId, chunkId);

    return viewer;
}

function setupDocumentViewerControls(viewer, documentId, chunkId) {
    const zoomIn = viewer.querySelector('.zoom-in');
    const zoomOut = viewer.querySelector('.zoom-out');
    const fitWidth = viewer.querySelector('.fit-width');
    const fitPage = viewer.querySelector('.fit-page');
    const searchInput = viewer.querySelector('.viewer-search-input');
    const searchPrev = viewer.querySelector('.search-prev');
    const searchNext = viewer.querySelector('.search-next');

    let currentZoom = 100;

    zoomIn.addEventListener('click', () => {
        currentZoom = Math.min(300, currentZoom + 25);
        updateZoom(viewer, currentZoom);
    });

    zoomOut.addEventListener('click', () => {
        currentZoom = Math.max(25, currentZoom - 25);
        updateZoom(viewer, currentZoom);
    });

    fitWidth.addEventListener('click', () => {
        // Implement fit to width logic
        updateZoom(viewer, 'fit-width');
    });

    fitPage.addEventListener('click', () => {
        // Implement fit to page logic
        updateZoom(viewer, 'fit-page');
    });

    // Document search within viewer
    const debouncedSearch = debounce((query) => {
        searchInDocument(viewer, query);
    }, 300);

    searchInput.addEventListener('input', (e) => {
        debouncedSearch(e.target.value);
    });

    searchPrev.addEventListener('click', () => {
        navigateSearchResult(viewer, -1);
    });

    searchNext.addEventListener('click', () => {
        navigateSearchResult(viewer, 1);
    });
}

function updateZoom(viewer, zoom) {
    const content = viewer.querySelector('.viewer-content');
    const zoomLevel = viewer.querySelector('.zoom-level');

    if (typeof zoom === 'number') {
        content.style.transform = `scale(${zoom / 100})`;
        zoomLevel.textContent = `${zoom}%`;
    } else if (zoom === 'fit-width') {
        // Calculate fit-to-width zoom
        const containerWidth = content.offsetWidth;
        const documentWidth = content.scrollWidth;
        const fitZoom = Math.floor((containerWidth / documentWidth) * 100);
        content.style.transform = `scale(${fitZoom / 100})`;
        zoomLevel.textContent = `${fitZoom}%`;
    } else if (zoom === 'fit-page') {
        // Calculate fit-to-page zoom
        content.style.transform = 'scale(1)';
        zoomLevel.textContent = 'Fit';
    }
}

function searchInDocument(viewer, query) {
    if (!query.trim()) {
        clearDocumentSearch(viewer);
        return;
    }

    const content = viewer.querySelector('.viewer-content');

    // This would interface with the document content
    // Implementation depends on the document format (PDF.js, plain text, etc.)

    // Highlight search results
    highlightSearchResults(content, query);
}

function highlightSearchResults(content, query) {
    // Remove existing highlights
    content.querySelectorAll('.search-highlight').forEach(el => {
        el.replaceWith(...el.childNodes);
    });

    if (!query) return;

    // Create text walker to find and highlight matches
    const walker = document.createTreeWalker(
        content,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );

    const matches = [];
    let node;

    while (node = walker.nextNode()) {
        const text = node.textContent;
        const regex = new RegExp(escapeRegex(query), 'gi');
        let match;

        while ((match = regex.exec(text)) !== null) {
            matches.push({
                node: node,
                start: match.index,
                end: match.index + match[0].length,
                text: match[0]
            });
        }
    }

    // Apply highlights in reverse order to preserve indices
    matches.reverse().forEach(match => {
        const before = match.node.textContent.substring(0, match.start);
        const highlighted = match.node.textContent.substring(match.start, match.end);
        const after = match.node.textContent.substring(match.end);

        const span = document.createElement('span');
        span.className = 'search-highlight';
        span.textContent = highlighted;

        const parent = match.node.parentNode;
        parent.insertBefore(document.createTextNode(before), match.node);
        parent.insertBefore(span, match.node);
        parent.insertBefore(document.createTextNode(after), match.node);
        parent.removeChild(match.node);
    });
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\    showInitializationError(error) {
        const errorHtml = `
            <div class="initialization');
}

function clearDocumentSearch(viewer) {
    const content = viewer.querySelector('.viewer-content');
    content.querySelectorAll('.search-highlight').forEach(el => {
        el.replaceWith(...el.childNodes);
    });
}

function navigateSearchResult(viewer, direction) {
    const highlights = viewer.querySelectorAll('.search-highlight');
    if (highlights.length === 0) return;
    
    const current = viewer.querySelector('.search-highlight.current');
    let nextIndex = 0;
    
    if (current) {
        const currentIndex = Array.from(highlights).indexOf(current);
        nextIndex = (currentIndex + direction + highlights.length) % highlights.length;
        current.classList.remove('current');
    }
    
    const next = highlights[nextIndex];
    next.classList.add('current');
    next.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Theme and appearance utilities
 */
function applyLegalTheme(themeName = 'professional') {
    const themes = {
        professional: {
            primary: '#2C3E50',
            secondary: '#3498DB',
            accent: '#E74C3C',
            background: '#FFFFFF',
            surface: '#F8F9FA',
            text: '#2C3E50'
        },
        classic: {
            primary: '#1A1A1A',
            secondary: '#4A5568',
            accent: '#D69E2E',
            background: '#FFFFFF',
            surface: '#F7FAFC',
            text: '#1A1A1A'
        },
        modern: {
            primary: '#0F172A',
            secondary: '#0EA5E9',
            accent: '#06B6D4',
            background: '#FFFFFF',
            surface: '#F1F5F9',
            text: '#0F172A'
        }
    };
    
    const theme = themes[themeName] || themes.professional;
    
    Object.entries(theme).forEach(([key, value]) => {
        document.documentElement.style.setProperty(`--color-${ key } `, value);
    });
    
    localStorage.setItem('legalai-theme', themeName);
}

function toggleDarkMode() {
    const isDark = document.documentElement.classList.toggle('dark-mode');
    localStorage.setItem('legalai-dark-mode', isDark.toString());
    
    // Announce to screen readers
    if (window.legalAIApp) {
        window.legalAIApp.announceToScreenReader(
            `Dark mode ${ isDark ? 'enabled' : 'disabled' } `
        );
    }
}

/**
 * Local storage utilities for legal data
 */
function saveLegalPreferences(preferences) {
    const existingPrefs = JSON.parse(localStorage.getItem('legalai-preferences') || '{}');
    const updatedPrefs = { ...existingPrefs, ...preferences };
    localStorage.setItem('legalai-preferences', JSON.stringify(updatedPrefs));
}

function getLegalPreferences() {
    return JSON.parse(localStorage.getItem('legalai-preferences') || '{}');
}

function clearLegalData() {
    const keysToRemove = Object.keys(localStorage).filter(key => 
        key.startsWith('legalai-')
    );
    
    keysToRemove.forEach(key => localStorage.removeItem(key));
    
    // Clear application state
    legalAIState.searchHistory = [];
    legalAIState.currentCase = null;
    
    if (window.searchManager) {
        window.searchManager.searchCache.clear();
    }
}

// =============================================================================
// GRADIO INTEGRATION HELPERS
// =============================================================================

/**
 * Gradio component integration utilities
 */
function updateGradioComponent(componentId, value) {
    // This function helps update Gradio components from JavaScript
    const event = new CustomEvent('gradioUpdate', {
        detail: {
            componentId: componentId,
            value: value
        }
    });
    document.dispatchEvent(event);
}

function listenToGradioEvents() {
    // Listen for events from Gradio components
    document.addEventListener('gradioEvent', (event) => {
        const { type, data } = event.detail;
        
        switch (type) {
            case 'document_processed':
                if (window.websocketManager) {
                    window.websocketManager.handleDocumentProcessed({ data });
                }
                break;
            case 'search_completed':
                if (window.websocketManager) {
                    window.websocketManager.handleSearchCompleted({ data });
                }
                break;
            case 'case_created':
                if (window.caseManager) {
                    window.caseManager.loadCases();
                }
                break;
        }
    });
}

/**
 * Initialize Gradio-specific functionality
 */
function initializeGradioIntegration() {
    // Setup Gradio event listeners
    listenToGradioEvents();
    
    // Apply saved theme
    const savedTheme = localStorage.getItem('legalai-theme');
    if (savedTheme) {
        applyLegalTheme(savedTheme);
    }
    
    // Apply dark mode preference
    const darkMode = localStorage.getItem('legalai-dark-mode') === 'true';
    if (darkMode) {
        document.documentElement.classList.add('dark-mode');
    }
    
    // Setup custom CSS for legal styling
    injectLegalStyles();
}

function injectLegalStyles() {
    const styles = `
        < style id = "legal-ai-styles" >
            /* Legal AI Custom Styles */
            .legal - term {
        background - color: #FFF3CD;
        border: 1px solid #FFEAA7;
        padding: 1px 3px;
        border - radius: 3px;
    }
            
            .search - highlight {
        background - color: #FFEB3B;
        color: #000;
        font - weight: bold;
    }
            
            .search - highlight.current {
        background - color: #FF9800;
        color: #FFF;
    }
            
            .case -marker {
        width: 12px;
        height: 12px;
        border - radius: 50 %;
        display: inline - block;
        margin - right: 8px;
    }
            
            .connection - status {
        position: fixed;
        top: 10px;
        right: 10px;
        padding: 5px 10px;
        border - radius: 4px;
        font - size: 12px;
        z - index: 1000;
    }
            
            .connection - status.connected {
        background - color: #4CAF50;
        color: white;
    }
            
            .connection - status.disconnected {
        background - color: #F44336;
        color: white;
    }
            
            .notification {
        position: fixed;
        top: 50px;
        right: 20px;
        max - width: 400px;
        padding: 15px;
        border - radius: 6px;
        box - shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z - index: 1001;
        transform: translateX(100 %);
        transition: transform 0.3s ease;
    }
            
            .notification.notification - show {
        transform: translateX(0);
    }
            
            .notification.notification - hide {
        transform: translateX(100 %);
    }
            
            .notification - success {
        background - color: #4CAF50;
        color: white;
    }
            
            .notification - error {
        background - color: #F44336;
        color: white;
    }
            
            .notification - warning {
        background - color: #FF9800;
        color: white;
    }
            
            .notification - info {
        background - color: #2196F3;
        color: white;
    }
            
            .sr - only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white - space: nowrap;
        border: 0;
    }
            
            .using - keyboard *:focus {
        outline: 2px solid #2196F3;
        outline - offset: 2px;
    }
        </ >
        `;
    
    document.head.insertAdjacentHTML('beforeend', styles);
}

// =============================================================================
// INITIALIZATION AND STARTUP
// =============================================================================

// Global application instance
let legalAIApp;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeLegalAI);
} else {
    initializeLegalAI();
}

async function initializeLegalAI() {
    try {
        console.log('Starting Legal AI initialization...');
        
        // Initialize Gradio integration first
        initializeGradioIntegration();
        
        // Create and initialize main application
        legalAIApp = new LegalAIApp();
        window.legalAIApp = legalAIApp;
        
        await legalAIApp.initialize();
        
        // Setup cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (legalAIApp) {
                legalAIApp.cleanup();
            }
        });
        
        console.log('Legal AI frontend ready for legal professionals! 🏛️⚖️');
        
    } catch (error) {
        console.error('Critical initialization error:', error);
        
        // Show fallback interface
        document.body.innerHTML = `
        < div class="critical-error" >
                <h1>⚠️ Legal AI Initialization Failed</h1>
                <p>The legal AI interface could not be initialized.</p>
                <button onclick="window.location.reload()">Reload Application</button>
            </ >
        `;
    }
}

// Export for external access (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        LegalAIApp,
        LegalAIWebSocket,
        DocumentUploadHandler,
        SearchManager,
        CaseManager,
        PerformanceMonitor
    };
}