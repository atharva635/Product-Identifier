// --- Constants & Global State ---
let API_URL = localStorage.getItem("API_URL") || "";

// If API_URL is empty, determine sensible defaults:
if (!API_URL) {
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
        if (window.location.port !== "8000") {
            API_URL = "http://localhost:8000";
        } else {
            API_URL = ""; // Served on the same port
        }
    } else {
        API_URL = "http://localhost:8000";
    }
}

let activeTab = 'search';
let uploadedFile = null;
let lossChartInstance = null;
let accuracyChartInstance = null;
let embeddingChartInstance = null;
let pollingInterval = null;
let stream = null; // Webcam stream
let currentVizMethod = 'pca';

// --- Tab Switching ---
function switchTab(tabId) {
    activeTab = tabId;
    
    // Toggle active nav buttons
    document.getElementById('btn-search').classList.toggle('active', tabId === 'search');
    document.getElementById('btn-analytics').classList.toggle('active', tabId === 'analytics');
    
    // Toggle active sections
    document.getElementById('section-search').classList.toggle('active', tabId === 'search');
    document.getElementById('section-analytics').classList.toggle('active', tabId === 'analytics');
    
    if (tabId === 'analytics') {
        fetchMetrics();
        loadEmbeddingViz(currentVizMethod);
    } else {
        fetchCatalog();
    }
}

// --- Toast Notification System ---
function showNotification(message, type = 'success') {
    const container = document.getElementById('notification-container');
    const toast = document.createElement('div');
    toast.className = `notification ${type}`;
    
    let iconClass = 'fa-circle-check';
    if (type === 'warning') iconClass = 'fa-triangle-exclamation';
    if (type === 'error') iconClass = 'fa-circle-exclamation';
    if (type === 'info') iconClass = 'fa-circle-info';
    
    toast.innerHTML = `
        <i class="fa-solid ${iconClass}"></i>
        <div>${message}</div>
    `;
    
    container.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 50);
    
    // Auto remove
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, 4000);
}

// --- Voice Search (Web Speech API) ---
function startVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showNotification("Speech recognition is not supported in this browser. Please try Chrome/Edge.", "warning");
        return;
    }
    
    const recognition = new SpeechRecognition();
    recognition.lang = "en-IN"; // Supports English with Indian accent
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    
    const voiceBtn = document.getElementById('voice-btn');
    voiceBtn.classList.add('recording');
    showNotification("Listening... Speak now.", "info");
    
    recognition.start();
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById('text-query').value = transcript;
        showNotification(`Voice query captured: "${transcript}"`, "success");
        // Trigger auto search
        document.getElementById('search-form').dispatchEvent(new Event('submit'));
    };
    
    recognition.onspeechend = () => {
        recognition.stop();
        voiceBtn.classList.remove('recording');
    };
    
    recognition.onerror = (event) => {
        voiceBtn.classList.remove('recording');
        console.error("Speech recognition error", event.error);
        showNotification(`Voice recognition error: ${event.error}`, "error");
    };
}

// --- Camera Access / Snapshot (webcam) ---
async function openCameraModal() {
    const modal = document.getElementById('camera-modal');
    const video = document.getElementById('webcam-video');
    modal.classList.add('active');
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
    } catch (e) {
        console.error("Webcam open failed", e);
        showNotification("Failed to access webcam. Please check browser permissions.", "error");
        closeCameraModal();
    }
}

function closeCameraModal() {
    const modal = document.getElementById('camera-modal');
    modal.classList.remove('active');
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

function captureCameraSnap() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const ctx = canvas.getContext('2d');
    
    if (!video.srcObject) return;
    
    // Match dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob((blob) => {
        const file = new File([blob], "camera_snap.jpg", { type: "image/jpeg" });
        processUploadedFile(file);
        closeCameraModal();
        showNotification("Captured photo successfully.", "success");
    }, 'image/jpeg', 0.9);
}

// --- Image Drag-n-Drop & Upload Handling ---
function triggerImageSelect() {
    if (!uploadedFile) {
        document.getElementById('image-input').click();
    }
}

function handleImageFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processUploadedFile(file);
    }
}

// Support drag over effect
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processUploadedFile(file);
    } else {
        showNotification('Please drop a valid image file.', 'warning');
    }
});

function processUploadedFile(file) {
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('image-preview').src = e.target.result;
        document.getElementById('image-preview-container').style.display = 'flex';
        
        // Switch to Multimodal Slider balance
        document.getElementById('weight-slider-group').style.display = 'none';
        document.getElementById('multimodal-slider-group').style.display = 'block';
    };
    reader.readAsDataURL(file);
    showNotification('Visual query loaded.', 'success');
}

function clearImagePreview(event) {
    event.stopPropagation(); // Stop click propagating to dropZone browse trigger
    uploadedFile = null;
    document.getElementById('image-input').value = '';
    document.getElementById('image-preview').src = '#';
    document.getElementById('image-preview-container').style.display = 'none';
    
    // Re-enable weight slider, hide multimodal
    document.getElementById('weight-slider-group').style.display = 'block';
    document.getElementById('multimodal-slider-group').style.display = 'none';
    showNotification('Visual search query removed.', 'info');
}

// --- Sliders Display ---
function updateWeightDisplay(value) {
    const valPct = Math.round(value * 100);
    const textVal = `${valPct}% Semantic (CLIP) / ${100 - valPct}% Lexical (BM25)`;
    document.getElementById('weight-display').innerText = textVal;
}

function updateMultimodalDisplay(value) {
    const valPct = Math.round(value * 100);
    const textVal = `${valPct}% Image Context / ${100 - valPct}% Text Modifier`;
    document.getElementById('multimodal-weight-display').innerText = textVal;
}

// --- Catalog Rendering & Search Actions ---
async function fetchCatalog() {
    try {
        const response = await fetch(`${API_URL}/api/catalog`);
        const catalog = await response.json();
        
        document.getElementById('results-title').innerText = "Product Catalog";
        document.getElementById('results-count').innerText = `${catalog.length} items found`;
        
        renderProductGrid(catalog.map(item => ({ product: item, fused_score: null })));
    } catch (e) {
        console.error(e);
        showNotification("Failed to load catalog. Check API server.", "error");
    }
}

function renderProductGrid(items) {
    const grid = document.getElementById('product-grid');
    grid.innerHTML = '';
    
    if (items.length === 0) {
        grid.innerHTML = `
            <div class="glass-panel" style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-secondary);">
                <i class="fa-solid fa-face-meh" style="font-size: 2.5rem; margin-bottom: 15px;"></i>
                <p>No products matched the query. Try adjusting your parameters.</p>
            </div>
        `;
        return;
    }
    
    items.forEach(item => {
        const prod = item.product;
        const scoreText = item.fused_score !== null ? `${Math.round(item.fused_score * 100)}%` : null;
        
        const card = document.createElement('div');
        card.className = 'product-card';
        card.onclick = () => showDiagnosticsModal(item);
        
        let statsHtml = '';
        if (item.fused_score !== null) {
            statsHtml = `
                <div class="card-stats">
                    <span>Semantic: <strong>${Math.round(item.dense_score * 100)}%</strong></span>
                    ${item.sparse_score > 0 ? `<span>Lexical: <strong>${Math.round(item.sparse_score * 100)}%</strong></span>` : ''}
                </div>
            `;
        }
        
        card.innerHTML = `
            <div class="card-image-wrapper">
                <span class="card-category-badge">${prod.category}</span>
                ${scoreText ? `<span class="card-match-badge">Match: ${scoreText}</span>` : ''}
                <img src="${API_URL}/${prod.image_path}" alt="${prod.title}" onerror="this.src='https://placehold.co/224x224?text=Product+Image'">
            </div>
            <div class="card-body">
                <div class="card-brand">${prod.brand}</div>
                <h3 class="card-title">${prod.title}</h3>
                <div class="card-price">₹${prod.price.toLocaleString()}</div>
                <p class="card-description">${prod.description}</p>
                ${statsHtml}
                <button class="recommend-btn" onclick="triggerRecommendationDrawer(event, '${prod.id}')">
                    <i class="fa-solid fa-heart"></i> Similar Items
                </button>
            </div>
        `;
        grid.appendChild(card);
    });
}

async function handleSearch(event) {
    event.preventDefault();
    
    const textQuery = document.getElementById('text-query').value.trim();
    const hybridWeight = parseFloat(document.getElementById('hybrid-weight').value);
    const multimodalAlpha = parseFloat(document.getElementById('multimodal-alpha').value);
    
    // Check if we have any inputs
    if (!textQuery && !uploadedFile) {
        showNotification("Please provide a text query or upload an image.", "warning");
        return;
    }
    
    const submitBtn = document.getElementById('search-submit-btn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Retrieving...`;

    try {
        let results = [];
        let queryMethod = "";
        
        if (uploadedFile && textQuery) {
            // Multimodal Search
            const formData = new FormData();
            formData.append('file', uploadedFile);
            formData.append('query', textQuery);
            formData.append('alpha', multimodalAlpha);
            formData.append('top_k', 6);
            
            queryMethod = "Multimodal Search (Image + Text)";
            
            const response = await fetch(`${API_URL}/api/search/multimodal`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (response.ok) {
                results = data.results;
            } else {
                throw new Error(data.detail || "Multimodal search failed");
            }
        } else if (uploadedFile) {
            // Pure visual search
            const formData = new FormData();
            formData.append('file', uploadedFile);
            formData.append('top_k', 6);
            
            queryMethod = "Visual Query (CLIP Image)";
            
            const response = await fetch(`${API_URL}/api/search/image`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (response.ok) {
                results = data.results;
            } else {
                throw new Error(data.detail || "Visual search failed");
            }
        } else {
            // Text semantic/hybrid search
            queryMethod = hybridWeight === 0 ? "BM25 Lexical" : (hybridWeight === 1 ? "CLIP Semantic" : "Hybrid Fusion");
            
            const response = await fetch(`${API_URL}/api/search/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: textQuery,
                    weight: hybridWeight,
                    top_k: 6
                })
            });
            const data = await response.json();
            if (response.ok) {
                results = data.results;
            } else {
                throw new Error(data.detail || "Semantic search failed");
            }
        }
        
        document.getElementById('results-title').innerText = `Search Results (${queryMethod})`;
        document.getElementById('results-count').innerText = `${results.length} matched items`;
        renderProductGrid(results);
        showNotification("Search executed successfully.", "success");
        
    } catch (e) {
        console.error(e);
        showNotification(e.message || "Retrieval pipeline failed.", "error");
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<span class="btn-text">Execute Search Query</span><i class="fa-solid fa-arrow-right btn-arrow"></i>`;
    }
}

// --- Recommendations Drawer ---
function triggerRecommendationDrawer(event, productId) {
    event.stopPropagation(); // prevent opening detailed modal
    showRecommendations(productId);
}

async function showRecommendations(productId) {
    const drawer = document.getElementById('recommendation-drawer');
    const grid = document.getElementById('recommendations-grid');
    const targetDiv = document.getElementById('drawer-target-product');
    
    grid.innerHTML = `<div style="color: var(--text-secondary); text-align:center;"><i class="fa-solid fa-spinner fa-spin"></i> Finding similar items...</div>`;
    
    // Open drawer first
    drawer.classList.add('active');
    
    try {
        // Find product details from DOM/state to display target
        const responseCatalog = await fetch(`${API_URL}/api/catalog`);
        const catalog = await responseCatalog.json();
        const targetProd = catalog.find(item => item.id === productId);
        
        if (targetProd) {
            targetDiv.innerHTML = `
                <img src="${API_URL}/${targetProd.image_path}" class="drawer-target-img" alt="${targetProd.title}">
                <div class="drawer-target-info">
                    <h5>${targetProd.title}</h5>
                    <p class="drawer-target-brand">${targetProd.brand} | ₹${targetProd.price.toLocaleString()}</p>
                    <p class="drawer-target-tagline">${targetProd.category}</p>
                </div>
            `;
        }
        
        // Fetch recommendations from API
        const responseRec = await fetch(`${API_URL}/api/recommendations/${productId}`);
        const dataRec = await responseRec.json();
        
        grid.innerHTML = '';
        if (dataRec.recommendations && dataRec.recommendations.length > 0) {
            dataRec.recommendations.forEach(rec => {
                const prod = rec.product;
                const score = Math.round(rec.similarity_score * 100);
                
                const recCard = document.createElement('div');
                recCard.className = 'rec-card';
                recCard.onclick = () => {
                    closeRecommendations();
                    showDiagnosticsModal({ product: prod, fused_score: rec.similarity_score, dense_score: rec.similarity_score, sparse_score: 0.0 });
                };
                recCard.innerHTML = `
                    <img src="${API_URL}/${prod.image_path}" alt="${prod.title}">
                    <div class="rec-info">
                        <div class="rec-title">${prod.title}</div>
                        <div class="rec-brand-price">${prod.brand} • ₹${prod.price.toLocaleString()}</div>
                        <div class="rec-desc">${prod.description}</div>
                    </div>
                    <span class="rec-score">${score}% Match</span>
                `;
                grid.appendChild(recCard);
            });
        } else {
            grid.innerHTML = `<div style="color: var(--text-muted); text-align:center;">No recommendations found. Build index first.</div>`;
        }
        
    } catch (e) {
        console.error(e);
        grid.innerHTML = `<div style="color: var(--error); text-align:center;">Error loading recommendations.</div>`;
    }
}

function closeRecommendations() {
    document.getElementById('recommendation-drawer').classList.remove('active');
}

// --- Product Diagnostic Modal (Explainable AI) ---
function showDiagnosticsModal(item) {
    const modal = document.getElementById('detail-modal');
    const body = document.getElementById('detail-modal-body');
    
    const prod = item.product;
    const matchPercentage = item.fused_score !== null ? `${Math.round(item.fused_score * 100)}%` : "N/A";
    
    // Generate explanation reasons list
    let reasonsHtml = "";
    if (item.match_reasons && item.match_reasons.length > 0) {
        item.match_reasons.forEach(reason => {
            reasonsHtml += `<li><i class="fa-solid fa-circle-check reason-check-icon"></i> ${reason}</li>`;
        });
    } else {
        reasonsHtml = `<li><i class="fa-solid fa-circle-info reason-check-icon"></i> Standard database retrieval alignment score.</li>`;
    }

    // Generate tags badge list
    let tagsHtml = "";
    prod.tags.forEach(tag => {
        tagsHtml += `<span class="detail-tag-badge">#${tag}</span>`;
    });
    
    body.innerHTML = `
        <div class="detail-image-box">
            <img src="${API_URL}/${prod.image_path}" alt="${prod.title}">
            <button class="btn btn-primary btn-block" style="margin-top: 15px;" onclick="closeDetailModal(); showRecommendations('${prod.id}')">
                <i class="fa-solid fa-heart"></i> Get Similar Visuals
            </button>
        </div>
        <div class="detail-info-box">
            <div class="detail-brand-lbl">${prod.brand}</div>
            <h2 class="detail-title-lbl">${prod.title}</h2>
            <div class="detail-price-lbl">₹${prod.price.toLocaleString()}</div>
            <p class="detail-desc-lbl">${prod.description}</p>
            <div class="detail-tags-lbl">${tagsHtml}</div>
            
            <div class="diagnostic-divider"></div>
            
            <h4 class="diagnostic-subtitle"><i class="fa-solid fa-shield-halved"></i> Retrieval Explanations (Explainable AI)</h4>
            <div class="diagnostic-stats">
                <div class="diag-stat-card">
                    <span>Overall Match</span>
                    <h4>${matchPercentage}</h4>
                </div>
                <div class="diag-stat-card">
                    <span>Semantic Match</span>
                    <h4>${item.dense_score !== null && item.dense_score !== undefined ? `${Math.round(item.dense_score * 100)}%` : 'N/A'}</h4>
                </div>
                <div class="diag-stat-card">
                    <span>Lexical Match</span>
                    <h4>${item.sparse_score !== null && item.sparse_score !== undefined ? `${Math.round(item.sparse_score * 100)}%` : 'N/A'}</h4>
                </div>
            </div>
            
            <ul class="diagnostic-reasons-list">
                ${reasonsHtml}
            </ul>
        </div>
    `;
    
    modal.classList.add('active');
}

function closeDetailModal() {
    document.getElementById('detail-modal').classList.remove('active');
}

// --- RAG Chatbot System ---
function toggleChatbot() {
    const body = document.getElementById('chatbot-body');
    const chevron = document.getElementById('chatbot-chevron');
    
    if (body.style.display === 'none') {
        body.style.display = 'flex';
        chevron.className = 'fa-solid fa-chevron-down';
    } else {
        body.style.display = 'none';
        chevron.className = 'fa-solid fa-chevron-up';
    }
}

async function handleChatSubmit(event) {
    event.preventDefault();
    const input = document.getElementById('chatbot-input');
    const message = input.value.trim();
    if (!message) return;
    
    // Add user message to UI
    appendChatMessage(message, 'user');
    input.value = '';
    
    // Add assistant thinking message
    const thinkingId = appendChatMessage("Typing...", 'assistant typing-loader');
    
    try {
        const response = await fetch(`${API_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        
        // Remove thinking message
        document.getElementById(thinkingId).remove();
        
        if (response.ok) {
            appendChatMessage(data.response, 'assistant');
        } else {
            throw new Error(data.detail || "Chat request failed");
        }
    } catch (e) {
        document.getElementById(thinkingId).remove();
        appendChatMessage(`Sorry, I had an issue fetching recommendations: ${e.message}`, 'assistant error-msg');
    }
}

function appendChatMessage(text, sender) {
    const messagesDiv = document.getElementById('chatbot-messages');
    const msgDiv = document.createElement('div');
    const msgId = `chat-msg-${Date.now()}-${Math.random()}`;
    msgDiv.id = msgId;
    msgDiv.className = `message ${sender}`;
    
    // Basic Markdown formatting helper for the chatbot responses
    let formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/🛍️/g, '🛍️ ')
        .replace(/\n/g, '<br>');
        
    msgDiv.innerHTML = formattedText;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    return msgId;
}

// --- Analytics, Performance Dashboard & Embedding Visualization ---
async function fetchMetrics() {
    try {
        const response = await fetch(`${API_URL}/api/metrics`);
        const metrics = await response.json();
        
        // Render system info
        document.getElementById('info-device').innerText = metrics.device.toUpperCase();
        document.getElementById('info-checkpoint').innerText = metrics.checkpoint_active ? "Active (Fine-tuned)" : "Pre-trained Base Model";
        document.getElementById('info-catalog-size').innerText = `${metrics.num_products} items`;
        
        // Performance Monitoring Dashboard
        if (metrics.performance) {
            const perf = metrics.performance;
            document.getElementById('perf-avg-latency').innerText = `${perf.avg_latency_ms.toFixed(1)} ms`;
            document.getElementById('perf-p95-latency').innerText = `${perf.p95_latency_ms.toFixed(1)} ms`;
            document.getElementById('perf-cpu-load').innerText = `${perf.cpu_util_pct.toFixed(0)}%`;
            document.getElementById('perf-ram-usage').innerText = `${perf.ram_util_pct.toFixed(0)}%`;
            
            let hardwareText = `Active Hardware: ${perf.gpu_hardware.toUpperCase()}`;
            if (perf.gpu_hardware !== "CPU Only") {
                hardwareText += ` | Allocated: ${perf.gpu_mem_allocated_mb.toFixed(1)} MB VRAM`;
            }
            document.getElementById('perf-hardware').innerText = hardwareText;
        }

        // Render Model Training State
        const stateBadge = document.getElementById('model-state-badge');
        const stateDetail = document.getElementById('model-state-detail');
        const progressPanel = document.getElementById('training-progress-panel');
        const progressText = document.getElementById('progress-status-text');
        
        const state = metrics.training_state;
        
        stateBadge.className = `status-badge state-${state.status}`;
        stateBadge.innerText = state.status.toUpperCase();
        
        if (state.is_training) {
            stateBadge.innerText = "TRAINING";
            stateBadge.className = `status-badge state-training`;
            stateDetail.innerText = state.status;
            progressPanel.style.display = 'flex';
            progressText.innerText = state.status;
            
            // Start polling if not already running
            if (!pollingInterval) {
                pollingInterval = setInterval(fetchMetrics, 2500);
            }
        } else {
            stateDetail.innerText = state.error ? `Error: ${state.error}` : `Contrastive training pipeline is idle. GPU is ready.`;
            progressPanel.style.display = 'none';
            
            // Stop polling if running
            if (pollingInterval) {
                clearInterval(pollingInterval);
                pollingInterval = null;
                showNotification("Model contrastive fine-tuning and index rebuild completed!", "success");
                
                // Reload metrics and reload scatter plot visualization
                setTimeout(() => {
                    fetchMetrics();
                    loadEmbeddingViz(currentVizMethod);
                }, 1000);
            }
        }
        
        // Render Charts if history is available
        if (metrics.history && Object.keys(metrics.history).length > 0) {
            renderLossChart(metrics.history);
            renderAccuracyChart(metrics.history);
        } else {
            renderDummyCharts();
        }
        
    } catch (e) {
        console.error(e);
        showNotification("Failed to fetch model metrics.", "error");
    }
}

async function loadEmbeddingViz(method) {
    currentVizMethod = method;
    
    // Toggle active buttons
    document.getElementById('viz-pca-btn').classList.toggle('active', method === 'pca');
    document.getElementById('viz-tsne-btn').classList.toggle('active', method === 'tsne');
    
    try {
        const response = await fetch(`${API_URL}/api/embeddings/viz?method=${method}`);
        const data = await response.json();
        
        if (!data || data.length === 0) return;
        
        renderEmbeddingScatterPlot(data);
    } catch (e) {
        console.error("Failed to load embedding visualization", e);
        showNotification("Failed to render interactive embedding cluster.", "error");
    }
}

function renderEmbeddingScatterPlot(points) {
    const ctx = document.getElementById('embeddingChart').getContext('2d');
    
    // Group points by category
    const categories = [...new Set(points.map(p => p.category))];
    const categoryColors = {
        'Footwear': '#3b82f6',     // Blue
        'Fashion': '#ec4899',      // Pink
        'Electronics': '#10b981',  // Green
        'Accessories': '#f59e0b'   // Gold
    };
    
    const datasets = categories.map(cat => {
        const catPoints = points.filter(p => p.category === cat);
        return {
            label: cat,
            data: catPoints.map(p => ({
                x: p.x,
                y: p.y,
                title: p.title,
                brand: p.brand,
                price: p.price,
                img: p.image_path,
                id: p.id
            })),
            backgroundColor: categoryColors[cat] || '#8b5cf6',
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderWidth: 1,
            pointRadius: 6,
            pointHoverRadius: 9
        };
    });
    
    if (embeddingChartInstance) {
        embeddingChartInstance.destroy();
    }
    
    embeddingChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f3f4f6', font: { family: 'Outfit' } },
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const p = context.raw;
                            return `[${p.brand}] ${p.title} - Price: ₹${p.price.toLocaleString()}`;
                        }
                    },
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#8b5cf6',
                    bodyColor: '#f3f4f6',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#9ca3af', font: { size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#9ca3af', font: { size: 9 } }
                }
            },
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    const elIdx = elements[0].index;
                    const dsIdx = elements[0].datasetIndex;
                    const point = datasets[dsIdx].data[elIdx];
                    // Open diagnostics modal for clicked product
                    showDiagnosticsModal({
                        product: {
                            id: point.id,
                            title: point.title,
                            brand: point.brand,
                            price: point.price,
                            description: points.find(p => p.id === point.id).description || "Product detail coordinates.",
                            category: datasets[dsIdx].label,
                            tags: points.find(p => p.id === point.id).tags || [],
                            image_path: point.img
                        },
                        fused_score: null,
                        dense_score: null,
                        sparse_score: null,
                        match_reasons: ["Retrieved via 2D Cluster interaction clicks"]
                    });
                }
            }
        }
    });
}

function switchVizMethod(method) {
    loadEmbeddingViz(method);
}

async function triggerTraining(event) {
    event.preventDefault();
    const epochs = document.getElementById('train-epochs').value;
    
    const submitBtn = document.getElementById('train-submit-btn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Handshaking...`;
    
    try {
        const formData = new FormData();
        formData.append('epochs', epochs);
        
        const response = await fetch(`${API_URL}/api/train`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (response.ok && data.status === "success") {
            showNotification("CLIP Fine-tuning thread started in background.", "success");
            fetchMetrics(); // Initial status check to show overlay
        } else {
            throw new Error(data.message || "Failed to trigger training");
        }
    } catch (e) {
        console.error(e);
        showNotification(e.message || "Could not launch training pipeline.", "error");
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i class="fa-solid fa-graduation-cap"></i> Start Contrastive Training`;
    }
}

// --- Chart rendering via Chart.js ---
function renderLossChart(history) {
    const ctx = document.getElementById('lossChart').getContext('2d');
    const epochs = history.train_loss.map((_, i) => `Epoch ${i + 1}`);
    
    if (lossChartInstance) {
        lossChartInstance.destroy();
    }
    
    lossChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Training Loss',
                    data: history.train_loss,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true
                },
                {
                    label: 'Validation Loss',
                    data: history.val_loss,
                    borderColor: '#ec4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.1)',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f3f4f6', font: { family: 'Outfit' } } }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } }
            }
        }
    });
}

function renderAccuracyChart(history) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    const epochs = history.train_acc1.map((_, i) => `Epoch ${i + 1}`);
    
    if (accuracyChartInstance) {
        accuracyChartInstance.destroy();
    }
    
    accuracyChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Top-1 Accuracy',
                    data: history.val_acc1.map(val => val * 100),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.05)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Top-5 Retrieval Accuracy',
                    data: history.val_acc5.map(val => val * 100),
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.05)',
                    borderWidth: 2,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f3f4f6', font: { family: 'Outfit' } } }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.05)' }, 
                    ticks: { color: '#9ca3af', callback: value => `${value}%` },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

function renderDummyCharts() {
    const lossHistory = { train_loss: [1.2, 0.8, 0.5], val_loss: [1.3, 0.9, 0.65] };
    const accHistory = { train_acc1: [0.3, 0.55, 0.72], val_acc1: [0.28, 0.52, 0.68], val_acc5: [0.55, 0.78, 0.90] };
    
    renderLossChart(lossHistory);
    renderAccuracyChart(accHistory);
}

// --- Connection Verification and Settings ---
async function checkAPIConnection() {
    const dot = document.getElementById('api-status-dot');
    const text = document.getElementById('api-status-text');
    const input = document.getElementById('api-url-input');
    
    if (input && document.activeElement !== input) {
        input.value = API_URL;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/catalog`, { method: 'GET' });
        if (response.ok) {
            dot.className = 'status-dot connected';
            text.innerText = 'Connected';
            return true;
        }
    } catch (e) {
        // Fallback checks
    }
    dot.className = 'status-dot disconnected';
    text.innerText = 'Disconnected';
    return false;
}

// --- Startup Initialization ---
window.onload = function() {
    fetchCatalog();
    updateWeightDisplay(0.5);
    updateMultimodalDisplay(0.5);
    
    // Initialize API input event listener
    const apiInput = document.getElementById('api-url-input');
    if (apiInput) {
        apiInput.value = API_URL;
        apiInput.addEventListener('change', (e) => {
            let val = e.target.value.trim();
            // Remove trailing slash if present
            if (val.endsWith('/')) {
                val = val.slice(0, -1);
            }
            API_URL = val;
            localStorage.setItem("API_URL", API_URL);
            showNotification(`API Server URL updated: ${API_URL || "same host/port"}`, "info");
            checkAPIConnection();
            
            // Re-fetch data using new url
            fetchCatalog();
            if (activeTab === 'analytics') {
                fetchMetrics();
                loadEmbeddingViz(currentVizMethod);
            }
        });
    }
    
    // Start periodic connection checking
    checkAPIConnection();
    setInterval(checkAPIConnection, 10000);
};
