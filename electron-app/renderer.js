const { ipcRenderer } = require('electron');
const io = require('socket.io-client');

class AerialThreatDetectionApp {
    constructor() {
        this.socket = null;
        this.isDetecting = false;
        this.currentSource = null;
        this.detectionResults = [];
        this.detectionStarted = false;
        this.settings = {
            confidence: 0.5,
            showLabels: true,
            showConfidence: true,
            modelPath: 'best.pt',
            device: 'auto',
            iouThreshold: 0.5,
            maxDetections: 100,
            boxThickness: 2,
            fontScale: 0.6
        };

        this.canvas = document.getElementById('detectionCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.initializeEventListeners();
        this.setupDropZone();
        this.loadSettings();
        this.updateUI();
    }

    initializeEventListeners() {
        // File input buttons
        document.getElementById('btnImage').addEventListener('click', () => this.selectImage());
        document.getElementById('btnVideo').addEventListener('click', () => this.selectVideo());
        document.getElementById('btnWebcam').addEventListener('click', () => this.startWebcam());

        // Detection controls
        document.getElementById('btnStartDetection').addEventListener('click', () => this.startDetection());
        document.getElementById('btnStopDetection').addEventListener('click', () => this.stopDetection());

        // Settings
        document.getElementById('confidenceSlider').addEventListener('input', (e) => {
            this.settings.confidence = parseFloat(e.target.value);
            document.getElementById('confidenceValue').textContent = e.target.value;
        });

        document.getElementById('showLabels').addEventListener('change', (e) => {
            this.settings.showLabels = e.target.checked;
        });

        document.getElementById('showConfidence').addEventListener('change', (e) => {
            this.settings.showConfidence = e.target.checked;
        });

        // Export buttons
        document.getElementById('btnSaveResults').addEventListener('click', () => this.saveResults());
        document.getElementById('btnExportVideo').addEventListener('click', () => this.exportVideo());

        // Results panel
        document.getElementById('clearResults').addEventListener('click', () => this.clearResults());

        // Menu event listeners
        ipcRenderer.on('menu-open-image', () => this.selectImage());
        ipcRenderer.on('menu-open-video', () => this.selectVideo());
        ipcRenderer.on('menu-start-webcam', () => this.startWebcam());
        ipcRenderer.on('menu-start-detection', () => this.startDetection());
        ipcRenderer.on('menu-stop-detection', () => this.stopDetection());
        ipcRenderer.on('menu-open-settings', () => this.openSettingsModal());

        // Python process outputs
        ipcRenderer.on('python-output', (event, data) => this.handlePythonOutput(data));
        ipcRenderer.on('python-error', (event, data) => this.handlePythonError(data));

        // Modal handlers
        this.setupModalHandlers();

        // Video controls
        this.setupVideoControls();
    }

    setupDropZone() {
        const dropZone = document.getElementById('dropZone');
        const videoContainer = document.querySelector('.video-container');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            videoContainer.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            videoContainer.addEventListener(eventName, () => {
                dropZone.classList.add('drag-over');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            videoContainer.addEventListener(eventName, () => {
                dropZone.classList.remove('drag-over');
            }, false);
        });

        videoContainer.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            const fileType = file.type.toLowerCase();
            
            if (fileType.startsWith('image/')) {
                this.loadImageFile(file);
            } else if (fileType.startsWith('video/')) {
                this.loadVideoFile(file);
            } else {
                this.showMessage('Unsupported file type. Please select an image or video file.', 'error');
            }
        }
    }

    loadImageFile(file) {
        const url = URL.createObjectURL(file);
        this.currentSource = {
            type: 'image',
            path: file.path || file.name,
            file: file
        };

        const img = new Image();
        img.onload = () => {
            this.drawImageToCanvas(img);
            this.updateUI();
            this.hideDropZone();
            URL.revokeObjectURL(url);
        };
        img.onerror = () => {
            this.showMessage('Failed to load image', 'error');
            URL.revokeObjectURL(url);
        };
        img.src = url;
    }

    loadVideoFile(file) {
        const url = URL.createObjectURL(file);
        this.currentSource = {
            type: 'video',
            path: file.path || file.name,
            file: file
        };

        // Create video element for preview
        const video = document.createElement('video');
        video.src = url;
        video.muted = true;
        video.onloadedmetadata = () => {
            video.currentTime = 0;
        };
        video.onloadeddata = () => {
            this.drawVideoFrameToCanvas(video);
            this.updateUI();
            this.hideDropZone();
            this.showVideoControls(true);
            URL.revokeObjectURL(url);
        };
        video.onerror = () => {
            this.showMessage('Failed to load video', 'error');
            URL.revokeObjectURL(url);
        };
    }

    async selectImage() {
        const result = await ipcRenderer.invoke('select-file', {
            title: 'Select Image',
            filters: [
                { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'gif'] },
                { name: 'All Files', extensions: ['*'] }
            ],
            properties: ['openFile']
        });

        if (!result.canceled && result.filePaths.length > 0) {
            this.loadImagePath(result.filePaths[0]);
        }
    }

    async selectVideo() {
        const result = await ipcRenderer.invoke('select-file', {
            title: 'Select Video',
            filters: [
                { name: 'Videos', extensions: ['mp4', 'avi', 'mov', 'mkv', 'wmv'] },
                { name: 'All Files', extensions: ['*'] }
            ],
            properties: ['openFile']
        });

        if (!result.canceled && result.filePaths.length > 0) {
            this.loadVideoPath(result.filePaths[0]);
        }
    }

    loadImagePath(imagePath) {
        this.currentSource = {
            type: 'image',
            path: imagePath
        };

        const img = new Image();
        img.onload = () => {
            this.drawImageToCanvas(img);
            this.updateUI();
            this.hideDropZone();
        };
        img.onerror = () => {
            this.showMessage('Failed to load image', 'error');
        };
        img.src = `file://${imagePath}`;
    }

    loadVideoPath(videoPath) {
        this.currentSource = {
            type: 'video',
            path: videoPath
        };

        // Create video element for preview
        const video = document.createElement('video');
        video.src = `file://${videoPath}`;
        video.muted = true;
        video.onloadedmetadata = () => {
            video.currentTime = 0;
        };
        video.onloadeddata = () => {
            this.drawVideoFrameToCanvas(video);
            this.updateUI();
            this.hideDropZone();
            this.showVideoControls(true);
        };
        video.onerror = () => {
            this.showMessage('Failed to load video', 'error');
        };
    }

    async startWebcam() {
        try {
            this.currentSource = {
                type: 'webcam',
                path: null
            };
            
            this.updateUI();
            this.hideDropZone();
            this.showMessage('Webcam mode selected. Click "Start Detection" to begin.', 'info');
        } catch (error) {
            this.showMessage('Failed to access webcam: ' + error.message, 'error');
        }
    }

    drawImageToCanvas(img) {
        const canvasAspect = this.canvas.width / this.canvas.height;
        const imgAspect = img.width / img.height;

        let drawWidth, drawHeight, offsetX, offsetY;

        if (imgAspect > canvasAspect) {
            drawWidth = this.canvas.width;
            drawHeight = this.canvas.width / imgAspect;
            offsetX = 0;
            offsetY = (this.canvas.height - drawHeight) / 2;
        } else {
            drawWidth = this.canvas.height * imgAspect;
            drawHeight = this.canvas.height;
            offsetX = (this.canvas.width - drawWidth) / 2;
            offsetY = 0;
        }

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
    }

    drawVideoFrameToCanvas(video) {
        const canvasAspect = this.canvas.width / this.canvas.height;
        const videoAspect = video.videoWidth / video.videoHeight;

        let drawWidth, drawHeight, offsetX, offsetY;

        if (videoAspect > canvasAspect) {
            drawWidth = this.canvas.width;
            drawHeight = this.canvas.width / videoAspect;
            offsetX = 0;
            offsetY = (this.canvas.height - drawHeight) / 2;
        } else {
            drawWidth = this.canvas.height * videoAspect;
            drawHeight = this.canvas.height;
            offsetX = (this.canvas.width - drawWidth) / 2;
            offsetY = 0;
        }

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);
    }

    async startDetection() {
        if (!this.currentSource) {
            this.showMessage('Please select an input source first', 'warning');
            return;
        }

        this.isDetecting = true;
        this.updateUI();
        this.showLoading(true, 'Connecting to detection server...');

        try {
            // Connect directly to the already running Flask server
            this.connectToDetectionServer();
            
            // Clear previous results
            this.detectionResults = [];
            this.clearResults();
            
            this.showMessage('Detection started successfully', 'success');

        } catch (error) {
            this.showMessage('Failed to start detection: ' + error.message, 'error');
            this.isDetecting = false;
            this.updateUI();
        } finally {
            this.showLoading(false);
        }
    }

    async stopDetection() {
        if (!this.isDetecting) return;

        this.isDetecting = false;
        this.detectionStarted = false;  // Reset flag when user stops detection
        this.updateUI();

        if (this.socket) {
            // Send stop signal to server
            this.socket.emit('stop_detection');
            
            setTimeout(() => {
                this.socket.disconnect();
                this.socket = null;
            }, 1000);
        }

        this.showMessage('Detection stopped by user', 'warning');
        this.updateStatus('Stopped', 'warning');
        
        // Show final results summary if any
        if (this.detectionResults.length > 0) {
            const totalObjects = this.detectionResults.reduce((sum, result) => 
                sum + (result.detections?.length || 0), 0);
            this.showMessage(`Final results: ${this.detectionResults.length} frames, ${totalObjects} objects detected`, 'info');
        }
    }

    connectToDetectionServer() {
        // Connect to the Python detection server via WebSocket
        this.socket = io('http://localhost:5000', {
            reconnection: true,
            reconnectionAttempts: 3,
            reconnectionDelay: 1000,
            timeout: 5000
        });

        this.socket.on('connect', () => {
            console.log('Connected to detection server');
            this.updateStatus('Connected - Ready for detection', 'success');
            this.showMessage('Connected to detection server', 'success');
            
            // Only start detection if this is the initial connection and user intended to start
            if (this.isDetecting && !this.detectionStarted) {
                this.detectionStarted = true;
                
                let sourcePath = this.currentSource.path;
                
                // If it's a drag-and-drop file, we need to use the actual file path
                if (this.currentSource.file && this.currentSource.file.path) {
                    sourcePath = this.currentSource.file.path;
                }
                
                const detectionData = {
                    source_type: this.currentSource.type,
                    source_path: sourcePath,
                    confidence: this.settings.confidence
                };
                console.log('Starting detection with data:', detectionData);
                this.socket.emit('start_detection', detectionData);
            }
        });

        this.socket.on('detection_result', (data) => {
            this.handleDetectionResult(data);
        });

        this.socket.on('frame_update', (data) => {
            this.updateCanvasWithFrame(data);
        });

        this.socket.on('status', (data) => {
            console.log('Server status:', data);
            this.showMessage(data.message, data.status === 'running' ? 'success' : 'info');
            if (data.status === 'running') {
                this.updateStatus('Detection Running...', 'processing');
            }
        });

        this.socket.on('error', (data) => {
            console.error('Server error:', data);
            this.showMessage('Server error: ' + data.message, 'error');
            this.isDetecting = false;
            this.updateUI();
        });

        this.socket.on('detection_complete', () => {
            console.log('Detection completed');
            this.isDetecting = false;
            this.detectionStarted = false;  // Reset flag to allow new detections
            this.updateUI();
            this.showMessage('🎉 Detection completed successfully!', 'success');
            this.updateStatus('Detection Complete', 'success');
            
            // Show results summary
            const totalResults = this.detectionResults.length;
            const totalObjects = this.detectionResults.reduce((sum, result) => 
                sum + (result.detections?.length || 0), 0);
            
            this.showMessage(`Results: ${totalResults} frames processed, ${totalObjects} objects detected`, 'info');
        });
        
        this.socket.on('detection_progress', (data) => {
            console.log('Progress update:', data);
            
            if (data.progress !== undefined) {
                // Video progress
                this.updateStatus(`Processing: ${data.progress.toFixed(1)}% (${data.processed_frames}/${data.total_frames} frames)`, 'processing');
            } else if (data.time_remaining !== undefined) {
                // Webcam time remaining
                this.updateStatus(`Webcam: ${data.time_remaining.toFixed(1)}s remaining`, 'processing');
            }
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from detection server');
            this.updateStatus('Disconnected', 'error');
            this.isDetecting = false;
            this.detectionStarted = false;  // Reset flag on disconnect
            this.updateUI();
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.showMessage('Failed to connect to detection server. Make sure it\'s running.', 'error');
            this.isDetecting = false;
            this.updateUI();
        });
    }

    handleDetectionResult(data) {
        console.log('Detection result received:', data);
        
        this.detectionResults.push({
            timestamp: new Date(),
            detections: data.detections,
            stats: data.stats,
            frame_id: data.frame_id || this.detectionResults.length + 1
        });

        this.updateDetectionStats(data.stats);
        this.updateResultsPanel(data.detections);
        
        // Update status with frame info
        const frameInfo = data.frame_id ? ` (Frame ${data.frame_id})` : '';
        this.updateStatus(`Detecting... ${data.detections.length} objects found${frameInfo}`, 'processing');
        
        // Show detection activity
        this.showMessage(`Found ${data.detections.length} detections in current frame`, 'success');
    }

    updateCanvasWithFrame(frameData) {
        if (!frameData || !frameData.frame) {
            console.warn('No frame data received');
            return;
        }
        
        const img = new Image();
        img.onload = () => {
            this.drawImageToCanvas(img);
            
            // Draw detection boxes if available
            if (frameData.detections && frameData.detections.length > 0) {
                this.drawDetections(frameData.detections);
                console.log(`Drew ${frameData.detections.length} detection boxes`);
            }
            
            // Update processing info
            document.getElementById('processingInfo').textContent = 
                `Processing frame... ${frameData.detections ? frameData.detections.length : 0} detections`;
        };
        
        img.onerror = () => {
            console.error('Failed to load frame image');
            this.showMessage('Frame display error', 'error');
        };
        
        img.src = 'data:image/jpeg;base64,' + frameData.frame;
    }

    drawDetections(detections) {
        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;

            // Scale coordinates to canvas - use default scaling if image dimensions not available
            const imageWidth = detection.image_width || 640;
            const imageHeight = detection.image_height || 384;
            
            const scaleX = this.canvas.width / imageWidth;
            const scaleY = this.canvas.height / imageHeight;

            const canvasX = x1 * scaleX;
            const canvasY = y1 * scaleY;
            const canvasWidth = width * scaleX;
            const canvasHeight = height * scaleY;

            // Set colors based on class
            let color = '#00FF00'; // Default green
            if (detection.class_name.toLowerCase().includes('soldier')) {
                color = '#FF0000'; // Red for soldiers
            } else if (detection.class_name.toLowerCase().includes('civilian')) {
                color = '#00FF00'; // Green for civilians
            }

            // Draw bounding box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = this.settings.boxThickness;
            this.ctx.strokeRect(canvasX, canvasY, canvasWidth, canvasHeight);

            // Draw label background
            if (this.settings.showLabels) {
                const label = this.settings.showConfidence 
                    ? `${detection.class_name}: ${detection.confidence.toFixed(2)}`
                    : detection.class_name;

                this.ctx.font = `${this.settings.fontScale * 16}px Arial`;
                const textMetrics = this.ctx.measureText(label);
                const textWidth = textMetrics.width;
                const textHeight = this.settings.fontScale * 16;

                // Background
                this.ctx.fillStyle = color;
                this.ctx.fillRect(canvasX, canvasY - textHeight - 4, textWidth + 8, textHeight + 4);

                // Text
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.fillText(label, canvasX + 4, canvasY - 4);
            }
        });
    }

    updateDetectionStats(stats) {
        document.getElementById('totalDetections').textContent = stats.total_detections || 0;
        // Handle both 'soldier' and 'soldiers' naming conventions
        const soldierCount = (stats.class_counts?.soldier || 0) + (stats.class_counts?.soldiers || 0);
        const civilianCount = (stats.class_counts?.civilian || 0) + (stats.class_counts?.civilians || 0);
        document.getElementById('soldierCount').textContent = soldierCount;
        document.getElementById('civilianCount').textContent = civilianCount;
        document.getElementById('avgConfidence').textContent = (stats.avg_confidence || 0).toFixed(2);
    }

    updateResultsPanel(detections) {
        const resultsContent = document.getElementById('resultsContent');
        
        if (!detections || detections.length === 0) {
            resultsContent.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-search"></i>
                    <p>No detections found in current frame.</p>
                </div>
            `;
            return;
        }

        // Clear previous results
        resultsContent.innerHTML = '';

        // Add timestamp
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'detection-timestamp';
        timestampDiv.innerHTML = `<small>Last updated: ${new Date().toLocaleTimeString()}</small>`;
        timestampDiv.style.cssText = 'margin-bottom: 10px; color: #666; font-style: italic;';
        resultsContent.appendChild(timestampDiv);

        detections.forEach((detection, index) => {
            const detectionItem = document.createElement('div');
            detectionItem.className = `detection-item ${detection.class_name.toLowerCase()}`;
            
            const [x1, y1, x2, y2] = detection.bbox;
            
            detectionItem.innerHTML = `
                <div class="detection-info">
                    <div class="detection-class">${detection.class_name}</div>
                    <div class="detection-coords">Box: (${Math.round(x1)}, ${Math.round(y1)}, ${Math.round(x2)}, ${Math.round(y2)})</div>
                </div>
                <div class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</div>
            `;

            resultsContent.appendChild(detectionItem);
        });
        
        // Auto-scroll to show latest results
        resultsContent.scrollTop = 0;
    }

    clearResults() {
        this.detectionResults = [];
        this.updateDetectionStats({});
        document.getElementById('resultsContent').innerHTML = `
            <div class="no-results">
                <i class="fas fa-search"></i>
                <p>Results cleared.</p>
            </div>
        `;
    }

    async saveResults() {
        if (this.detectionResults.length === 0) {
            this.showMessage('No results to save', 'warning');
            return;
        }

        const result = await ipcRenderer.invoke('select-save-file', {
            title: 'Save Detection Results',
            defaultPath: 'detection_results.json',
            filters: [
                { name: 'JSON Files', extensions: ['json'] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });

        if (!result.canceled && result.filePath) {
            try {
                const fs = require('fs');
                fs.writeFileSync(result.filePath, JSON.stringify(this.detectionResults, null, 2));
                this.showMessage('Results saved successfully', 'success');
            } catch (error) {
                this.showMessage('Failed to save results: ' + error.message, 'error');
            }
        }
    }

    exportVideo() {
        this.showMessage('Video export feature coming soon', 'info');
    }

    hideDropZone() {
        document.getElementById('dropZone').style.display = 'none';
    }

    showVideoControls(show) {
        document.getElementById('videoControls').style.display = show ? 'flex' : 'none';
    }

    updateUI() {
        const hasSource = this.currentSource !== null;
        const isDetecting = this.isDetecting;

        document.getElementById('btnStartDetection').disabled = !hasSource || isDetecting;
        document.getElementById('btnStopDetection').disabled = !isDetecting;
        document.getElementById('btnSaveResults').disabled = this.detectionResults.length === 0;
        document.getElementById('btnExportVideo').disabled = !hasSource || this.currentSource.type !== 'video';

        this.updateStatus(
            isDetecting ? 'Detecting...' : hasSource ? 'Ready' : 'No Input',
            isDetecting ? 'processing' : hasSource ? 'ready' : 'waiting'
        );
    }

    updateStatus(text, type) {
        const statusText = document.getElementById('statusText');
        const statusDot = document.getElementById('statusDot');

        statusText.textContent = text;
        statusDot.className = 'status-dot';

        switch (type) {
            case 'processing':
                statusDot.classList.add('warning');
                break;
            case 'error':
                statusDot.classList.add('error');
                break;
            case 'ready':
            case 'success':
            default:
                // Keep default green
                break;
        }
    }

    showMessage(message, type = 'info') {
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        const processingInfo = document.getElementById('processingInfo');
        processingInfo.textContent = message;
        
        // Add color coding based on message type
        processingInfo.className = `processing-info ${type}`;
        
        // Auto-clear after different times based on type
        const clearTime = type === 'error' ? 10000 : type === 'success' ? 3000 : 5000;
        
        clearTimeout(this.messageTimeout);
        this.messageTimeout = setTimeout(() => {
            if (this.isDetecting) {
                processingInfo.textContent = 'Detection in progress...';
            } else {
                processingInfo.textContent = 'Ready for processing';
            }
            processingInfo.className = 'processing-info';
        }, clearTime);
    }

    showLoading(show, message = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const text = document.getElementById('loadingText');

        text.textContent = message;

        if (show) {
            overlay.classList.add('show');
        } else {
            overlay.classList.remove('show');
        }
    }

    setupModalHandlers() {
        const settingsModal = document.getElementById('settingsModal');
        const closeModal = settingsModal.querySelector('.close');
        const cancelBtn = document.getElementById('cancelSettings');
        const saveBtn = document.getElementById('saveSettings');

        closeModal.addEventListener('click', () => this.closeSettingsModal());
        cancelBtn.addEventListener('click', () => this.closeSettingsModal());
        saveBtn.addEventListener('click', () => this.saveSettings());

        // Click outside modal to close
        settingsModal.addEventListener('click', (e) => {
            if (e.target === settingsModal) {
                this.closeSettingsModal();
            }
        });

        // Setup setting sliders
        document.getElementById('iouThreshold').addEventListener('input', (e) => {
            document.getElementById('iouValue').textContent = e.target.value;
        });

        document.getElementById('boxThickness').addEventListener('input', (e) => {
            document.getElementById('boxThicknessValue').textContent = e.target.value;
        });

        document.getElementById('fontScale').addEventListener('input', (e) => {
            document.getElementById('fontScaleValue').textContent = e.target.value;
        });
    }

    openSettingsModal() {
        const modal = document.getElementById('settingsModal');
        
        // Load current settings into modal
        document.getElementById('modelPath').value = this.settings.modelPath;
        document.getElementById('deviceSelect').value = this.settings.device;
        document.getElementById('iouThreshold').value = this.settings.iouThreshold;
        document.getElementById('iouValue').textContent = this.settings.iouThreshold;
        document.getElementById('maxDetections').value = this.settings.maxDetections;
        document.getElementById('boxThickness').value = this.settings.boxThickness;
        document.getElementById('boxThicknessValue').textContent = this.settings.boxThickness;
        document.getElementById('fontScale').value = this.settings.fontScale;
        document.getElementById('fontScaleValue').textContent = this.settings.fontScale;

        modal.classList.add('show');
    }

    closeSettingsModal() {
        document.getElementById('settingsModal').classList.remove('show');
    }

    saveSettings() {
        // Get values from modal
        this.settings.modelPath = document.getElementById('modelPath').value;
        this.settings.device = document.getElementById('deviceSelect').value;
        this.settings.iouThreshold = parseFloat(document.getElementById('iouThreshold').value);
        this.settings.maxDetections = parseInt(document.getElementById('maxDetections').value);
        this.settings.boxThickness = parseInt(document.getElementById('boxThickness').value);
        this.settings.fontScale = parseFloat(document.getElementById('fontScale').value);

        // Save to localStorage
        localStorage.setItem('detectionSettings', JSON.stringify(this.settings));

        this.closeSettingsModal();
        this.showMessage('Settings saved successfully', 'success');
    }

    loadSettings() {
        const saved = localStorage.getItem('detectionSettings');
        if (saved) {
            try {
                const savedSettings = JSON.parse(saved);
                this.settings = { ...this.settings, ...savedSettings };
            } catch (error) {
                console.error('Failed to load settings:', error);
            }
        }

        // Update UI with loaded settings
        document.getElementById('confidenceSlider').value = this.settings.confidence;
        document.getElementById('confidenceValue').textContent = this.settings.confidence;
        document.getElementById('showLabels').checked = this.settings.showLabels;
        document.getElementById('showConfidence').checked = this.settings.showConfidence;
    }

    setupVideoControls() {
        // Placeholder for video control functionality
        // This would be implemented when adding video playback features
    }

    handlePythonOutput(data) {
        console.log('Python output:', data);
        // You can display this in a debug console if needed
    }

    handlePythonError(data) {
        console.error('Python error:', data);
        this.showMessage('Detection error: ' + data, 'error');
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new AerialThreatDetectionApp();
    window.app = app; // Make available globally for debugging
});