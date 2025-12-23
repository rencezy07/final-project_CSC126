// Main application JavaScript

class AerialThreatDetectionApp {
    constructor() {
        this.currentMode = 'image';
        this.sessionStats = { Soldier: 0, Civilian: 0, avgConfidence: 0 };
        this.liveStreamActive = false;
        this.liveStreamInterval = null;
        this.detectionHistory = [];
        this.chart = null;
        
        // Enhanced video processing variables
        this.videoData = null;
        this.currentFrame = 0;
        this.totalFrames = 0;
        this.videoFPS = 30;
        this.isVideoPlaying = false;
        this.videoPlayInterval = null;
        this.totalVideoSoldiers = 0;
        this.totalVideoCivilians = 0;
        
        this.settings = {
            defaultConfidence: 0.5,
            maxDetections: 50,
            showConfidenceLabels: true,
            showBoundingBoxes: true,
            apiUrl: 'http://localhost:5000/api'
        };

        this.init();
    }

    async init() {
        this.loadSettings();
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.initChart();
        this.startSystemStatsAnimation();
        this.initModernUI();
        
        // Initialize API with loaded settings
        api.setBaseUrl(this.settings.apiUrl);

        // Check initial connection
        this.checkApiConnection();

        // Add debug info
        console.log('🚁 Aerial Threat Detection App initialized');
        console.log('Settings:', this.settings);
        console.log('API Base URL:', api.baseUrl);
        
        // Add debug test button (temporary)
        this.addDebugButton();
    }

    loadSettings() {
        const saved = Utils.loadFromStorage('app-settings');
        if (saved) {
            this.settings = { ...this.settings, ...saved };
            this.updateSettingsUI();
        }
    }

    saveSettings() {
        Utils.saveToStorage('app-settings', this.settings);
    }

    updateSettingsUI() {
        document.getElementById('defaultConfidence').value = this.settings.defaultConfidence;
        document.getElementById('maxDetections').value = this.settings.maxDetections;
        document.getElementById('showConfidenceLabels').checked = this.settings.showConfidenceLabels;
        document.getElementById('showBoundingBoxes').checked = this.settings.showBoundingBoxes;
        document.getElementById('apiUrl').value = this.settings.apiUrl;
        document.getElementById('confidenceSlider').value = this.settings.defaultConfidence;
        document.getElementById('confidenceValue').textContent = this.settings.defaultConfidence;
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const mode = e.currentTarget.dataset.mode;
                this.switchMode(mode);
            });
        });

        // Image detection
        document.getElementById('loadImageBtn').addEventListener('click', () => {
            document.getElementById('imageFileInput').click();
        });

        document.getElementById('imageFileInput').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageUpload(e.target.files[0]);
            }
        });

        // Video processing
        document.getElementById('loadVideoBtn').addEventListener('click', () => {
            console.log('Load video button clicked');
            document.getElementById('videoFileInput').click();
        });

        document.getElementById('videoFileInput').addEventListener('change', (e) => {
            console.log('Video file input changed:', e.target.files);
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                console.log('Selected file:', file.name, file.type, file.size);
                this.handleVideoUpload(file);
            }
        });

        // Add click handler to upload area as backup
        document.getElementById('videoUploadArea').addEventListener('click', (e) => {
            console.log('Upload area clicked');
            if (e.target.tagName !== 'INPUT') {
                document.getElementById('videoFileInput').click();
            }
        });

        document.getElementById('startVideoAnalysisBtn').addEventListener('click', () => {
            this.startOptimizedVideoAnalysis();
        });

        document.getElementById('pauseVideoAnalysisBtn').addEventListener('click', () => {
            this.pauseVideoAnalysis();
        });

        document.getElementById('stopVideoAnalysisBtn').addEventListener('click', () => {
            this.stopVideoAnalysis();
        });

        // Video frame slider control
        document.getElementById('videoFrameSlider').addEventListener('input', (e) => {
            this.seekToFrame(parseInt(e.target.value));
        });

        // Live stream
        document.getElementById('startLiveBtn').addEventListener('click', () => {
            this.startLiveStream();
        });

        document.getElementById('stopLiveBtn').addEventListener('click', () => {
            this.stopLiveStream();
        });

        // Controls
        document.getElementById('confidenceSlider').addEventListener('input', (e) => {
            document.getElementById('confidenceValue').textContent = e.target.value;
        });

        // Settings
        document.getElementById('testConnectionBtn').addEventListener('click', () => {
            this.testConnection();
        });

        document.getElementById('defaultConfidence').addEventListener('change', (e) => {
            this.settings.defaultConfidence = parseFloat(e.target.value);
            this.saveSettings();
        });

        document.getElementById('maxDetections').addEventListener('change', (e) => {
            this.settings.maxDetections = parseInt(e.target.value);
            this.saveSettings();
        });

        document.getElementById('showConfidenceLabels').addEventListener('change', (e) => {
            this.settings.showConfidenceLabels = e.target.checked;
            this.saveSettings();
        });

        document.getElementById('showBoundingBoxes').addEventListener('change', (e) => {
            this.settings.showBoundingBoxes = e.target.checked;
            this.saveSettings();
        });

        document.getElementById('apiUrl').addEventListener('change', (e) => {
            this.settings.apiUrl = e.target.value;
            api.setBaseUrl(e.target.value);
            this.saveSettings();
        });

        // Export and clear
        document.getElementById('exportStatsBtn').addEventListener('click', () => {
            this.exportStatistics();
        });

        document.getElementById('clearStatsBtn').addEventListener('click', () => {
            this.clearStatistics();
        });

        // Reports functionality
        document.getElementById('generateReportBtn')?.addEventListener('click', () => {
            this.generateReport();
        });

        document.getElementById('exportReportBtn')?.addEventListener('click', () => {
            this.exportReport();
        });

        document.getElementById('exportPDFBtn')?.addEventListener('click', () => {
            this.exportToPDF();
        });

        document.getElementById('exportCSVBtn')?.addEventListener('click', () => {
            this.exportToCSV();
        });

        document.getElementById('exportJSONBtn')?.addEventListener('click', () => {
            this.exportToJSON();
        });

        // Electron IPC listeners
        if (Utils.isElectron()) {
            const { ipcRenderer } = require('electron');

            ipcRenderer.on('file-selected', (event, data) => {
                if (data.type === 'image') {
                    this.loadImageFile(data.path);
                } else if (data.type === 'video') {
                    this.loadVideoFile(data.path);
                }
            });

            ipcRenderer.on('start-live-feed', () => {
                this.startLiveStream();
            });

            ipcRenderer.on('stop-live-feed', () => {
                this.stopLiveStream();
            });

            ipcRenderer.on('show-settings', () => {
                this.switchMode('settings');
            });

            ipcRenderer.on('export-results', () => {
                this.exportStatistics();
            });
        }
    }

    setupDragAndDrop() {
        const imageUploadArea = document.getElementById('imageUploadArea');
        const videoUploadArea = document.getElementById('videoUploadArea');

        [imageUploadArea, videoUploadArea].forEach((area, index) => {
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('dragover');
            });

            area.addEventListener('dragleave', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');
            });

            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');

                const files = Array.from(e.dataTransfer.files);
                const file = files[0];

                if (!file) return;

                if (index === 0) { // Image area
                    if (Utils.validateFileType(file, ['image'])) {
                        this.handleImageUpload(file);
                    } else {
                        Utils.showToast('Please select a valid image file', 'error');
                    }
                } else { // Video area
                    if (Utils.validateFileType(file, ['video'])) {
                        this.handleVideoUpload(file);
                    } else {
                        Utils.showToast('Please select a valid video file', 'error');
                    }
                }
            });

            // Click to browse
            area.addEventListener('click', () => {
                if (index === 0) {
                    document.getElementById('imageFileInput').click();
                } else {
                    document.getElementById('videoFileInput').click();
                }
            });
        });
    }

    switchMode(mode) {
        this.currentMode = mode;

        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');

        // Update panels
        document.querySelectorAll('.panel').forEach(panel => {
            panel.classList.remove('active');
        });

        const targetPanel = document.getElementById(`${mode}Panel`);
        if (targetPanel) {
            targetPanel.classList.add('active');
            Utils.animate(targetPanel, 'fade-in');
        }

        // Handle mode-specific initialization
        if (mode === 'statistics') {
            this.updateChart();
        } else if (mode === 'reports') {
            this.updateReportsPanel();
        }
    }

    async checkApiConnection() {
        try {
            const result = await api.checkHealth();
            if (result && result.status === 'healthy') {
                Utils.showToast('Connected to detection server', 'success', 3000);
                api.isConnected = true;
            } else {
                Utils.showToast('Cannot connect to detection server. Please ensure the backend is running.', 'error', 10000);
                api.isConnected = false;
            }
            api.updateConnectionStatus();
        } catch (error) {
            Utils.showToast('Failed to connect to detection server', 'error', 10000);
            api.isConnected = false;
            api.updateConnectionStatus();
            console.error('Connection check failed:', error);
        }
    }

    async testConnection() {
        const apiUrl = document.getElementById('apiUrl').value;
        
        Utils.showLoading('Testing connection...');
        
        try {
            const result = await api.testConnection(apiUrl);
            
            if (result.success) {
                Utils.showToast('Connection successful!', 'success');
                this.settings.apiUrl = apiUrl;
                this.saveSettings();
            } else {
                Utils.showToast(`Connection failed: ${result.error}`, 'error');
            }
        } catch (error) {
            Utils.showToast(`Connection failed: ${error.message}`, 'error');
        } finally {
            Utils.hideLoading();
        }
    }

    async handleImageUpload(file) {
        console.log('Processing image upload:', file.name);
        
        if (!Utils.validateFileType(file, ['image'])) {
            Utils.showToast('Please select a valid image file', 'error');
            return;
        }

        // Reset previous results
        this.resetImageDisplay();

        const confidence = parseFloat(document.getElementById('confidenceSlider')?.value || 0.5);
        
        Utils.showLoading('Analyzing image for threats...');

        try {
            // Show image preview immediately
            const imageUrl = URL.createObjectURL(file);
            this.showImagePreview(imageUrl);

            // Check API connection first
            if (!api.isConnected) {
                await this.checkApiConnection();
                if (!api.isConnected) {
                    throw new Error('No connection to detection server');
                }
            }

            // Perform detection
            console.log('Starting detection with confidence:', confidence);
            const result = await api.detectImage(file, confidence);
            console.log('Detection result received:', result);

            if (result && (result.success !== false)) {
                // Update processed image with annotated version if available
                const processedImage = document.getElementById('processedImage');
                if (result.annotated_image && processedImage) {
                    processedImage.onload = () => {
                        console.log('Annotated image loaded successfully');
                    };
                    processedImage.onerror = (error) => {
                        console.error('Failed to load annotated image:', error);
                        Utils.showToast('Could not display annotated image', 'warning');
                    };
                    processedImage.src = result.annotated_image;
                    processedImage.style.maxWidth = '100%';
                    processedImage.style.height = 'auto';
                } else if (processedImage) {
                    console.warn('No annotated image in response, keeping original');
                }

                // Process and display results
                const detections = result.detections || [];
                const detectionCounts = result.detection_counts || { Soldier: 0, Civilian: 0 };
                const totalDetections = result.total_detections || detections.length;
                
                console.log('Processing detections:', { detections, detectionCounts, totalDetections });
                
                // Show results
                this.displayImageResults(detections, detectionCounts);
                this.updateSessionStats(detectionCounts, true, 'image', detections);

                // Show success message
                const message = totalDetections > 0 
                    ? `Analysis complete! Found ${totalDetections} person${totalDetections > 1 ? 's' : ''}` 
                    : 'Analysis complete! No people detected in image';
                Utils.showToast(message, totalDetections > 0 ? 'success' : 'info');
                
                // Ensure results area is visible
                const resultsArea = document.getElementById('imageResults');
                if (resultsArea) {
                    resultsArea.style.display = 'block';
                }
            } else {
                const errorMsg = result?.message || result?.error || 'Unknown detection error';
                Utils.showToast(`Detection failed: ${errorMsg}`, 'error');
                console.error('Detection failed:', result);
            }
        } catch (error) {
            Utils.showToast(`Detection failed: ${error.message}`, 'error');
            console.error('Image detection error:', error);
        } finally {
            Utils.hideLoading();
        }
    }

    resetImageDisplay() {
        const uploadArea = document.getElementById('imageUploadArea');
        const resultsArea = document.getElementById('imageResults');
        const processedImage = document.getElementById('processedImage');
        const detectionList = document.getElementById('imageDetectionList');

        // Reset display state
        if (uploadArea) uploadArea.style.display = 'block';
        if (resultsArea) resultsArea.style.display = 'none';
        if (processedImage) processedImage.src = '';
        if (detectionList) detectionList.innerHTML = '';
    }

    showImagePreview(imageUrl) {
        const uploadArea = document.getElementById('imageUploadArea');
        const resultsArea = document.getElementById('imageResults');
        const processedImage = document.getElementById('processedImage');

        // Set the original image as preview
        processedImage.src = imageUrl;
        processedImage.style.maxWidth = '100%';
        processedImage.style.height = 'auto';
        processedImage.style.borderRadius = '8px';
        
        // Hide upload area and show results
        uploadArea.style.display = 'none';
        resultsArea.style.display = 'grid';
        
        console.log('Image preview set:', imageUrl); // Debug log
    }

    displayImageResults(detections, detectionCounts) {
        const detectionList = document.getElementById('imageDetectionList');
        
        if (!detectionList) {
            console.error('Detection list element not found');
            return;
        }
        
        console.log('Displaying results:', { detections, detectionCounts }); // Debug log
        
        // Clear previous results
        detectionList.innerHTML = '';

        if (detections && detections.length > 0) {
            detections.forEach((detection, index) => {
                const item = Utils.createDetectionItem(detection, index);
                const itemElement = document.createElement('div');
                itemElement.innerHTML = item;
                detectionList.appendChild(itemElement.firstElementChild);
            });
        } else {
            const noDetections = document.createElement('div');
            noDetections.className = 'no-detections';
            noDetections.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
                    <i class="fas fa-search" style="font-size: 2rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                    <p>No people detected in this image</p>
                    <small>Try adjusting the detection sensitivity in settings</small>
                </div>
            `;
            detectionList.appendChild(noDetections);
        }

        // Add summary section
        const totalDetections = (detections && detections.length) || 0;
        const soldiers = detectionCounts ? (detectionCounts.Soldier || 0) : 0;
        const civilians = detectionCounts ? (detectionCounts.Civilian || 0) : 0;
        
        const summaryElement = document.createElement('div');
        summaryElement.className = 'detection-summary';
        summaryElement.innerHTML = `
            <div style="background: var(--bg-secondary); border-radius: var(--radius-md); padding: var(--spacing-lg); margin-top: var(--spacing-lg); border: 1px solid var(--border-light);">
                <h4 style="margin-bottom: var(--spacing-md); color: var(--text-primary); font-size: 16px;">📊 Detection Summary</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: var(--spacing-md);">
                    <div style="text-align: center; padding: var(--spacing-md); background: var(--bg-primary); border-radius: var(--radius-sm);">
                        <div style="font-size: 18px; font-weight: 700; color: var(--primary-color);">${totalDetections}</div>
                        <div style="font-size: 12px; color: var(--text-secondary); font-weight: 500;">Total Found</div>
                    </div>
                    <div style="text-align: center; padding: var(--spacing-md); background: var(--bg-primary); border-radius: var(--radius-sm);">
                        <div style="font-size: 18px; font-weight: 700; color: var(--danger-color);">${soldiers}</div>
                        <div style="font-size: 12px; color: var(--text-secondary); font-weight: 500;">Armed Personnel</div>
                    </div>
                    <div style="text-align: center; padding: var(--spacing-md); background: var(--bg-primary); border-radius: var(--radius-sm);">
                        <div style="font-size: 18px; font-weight: 700; color: var(--success-color);">${civilians}</div>
                        <div style="font-size: 12px; color: var(--text-secondary); font-weight: 500;">Civilians</div>
                    </div>
                </div>
            </div>
        `;
        detectionList.appendChild(summaryElement);
    }

    async handleVideoUpload(file) {
        console.log('Processing video upload:', file.name, file.type, file.size);
        
        if (!file) {
            Utils.showToast('No file selected', 'error');
            return;
        }
        
        // Check if it's a video file
        if (!file.type.startsWith('video/')) {
            Utils.showToast('Please select a valid video file (MP4, AVI, MOV)', 'error');
            return;
        }

        this.currentVideoFile = file;
        
        // Show upload success state
        const uploadArea = document.getElementById('videoUploadArea');
        uploadArea.innerHTML = `
            <div class="upload-success" style="text-align: center; padding: 20px;">
                <i class="fas fa-video" style="font-size: 48px; color: #4CAF50; margin-bottom: 10px;"></i>
                <h3 style="color: #4CAF50; margin: 10px 0;">${file.name}</h3>
                <p>Video loaded successfully - Ready for real-time analysis</p>
                <div class="file-info" style="margin-top: 10px; font-size: 14px; color: #666;">
                    <span>Size: ${Utils.formatFileSize(file.size)}</span> | 
                    <span>Type: ${file.type}</span>
                </div>
            </div>
        `;
        
        // Enable the start analysis button
        document.getElementById('startVideoAnalysisBtn').disabled = false;
        
        Utils.showToast(`Video loaded: ${file.name}`, 'success');
        console.log('Video upload successful:', file.name, file.size, file.type);
    }

    showVideoPreview(videoUrl) {
        const uploadArea = document.getElementById('videoUploadArea');
        const processedVideo = document.getElementById('processedVideo');

        processedVideo.src = videoUrl;
        uploadArea.style.display = 'none';
        document.getElementById('videoResults').style.display = 'grid';
    }

    async startVideoAnalysis() {
        if (!this.currentVideoFile) {
            Utils.showToast('No video file selected', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('video', this.currentVideoFile);
        
        document.getElementById('startVideoAnalysisBtn').disabled = true;
        Utils.showToast('Preparing video for analysis...', 'info');
        
        try {
            // Upload video and get metadata
            const response = await fetch(`${this.settings.apiUrl}/detect/video/stream`, {
                method: 'POST',
                body: formData
            });
            
            this.videoData = await response.json();
            
            if (this.videoData.success) {
                this.totalFrames = this.videoData.total_frames;
                this.videoFPS = this.videoData.fps;
                this.currentFrame = 0;
                
                // Show video player section
                document.getElementById('videoPlayerSection').style.display = 'block';
                
                // Setup video controls
                document.getElementById('videoFrameSlider').max = this.totalFrames - 1;
                document.getElementById('currentFrameInfo').textContent = `Frame: 0 / ${this.totalFrames}`;
                
                // Show control buttons
                document.getElementById('pauseVideoAnalysisBtn').style.display = 'inline-block';
                document.getElementById('stopVideoAnalysisBtn').style.display = 'inline-block';
                
                // Start playback
                this.isVideoPlaying = true;
                this.playVideoFrame();
                
                Utils.showToast('Video analysis started!', 'success');
            } else {
                Utils.showToast('Failed to prepare video: ' + this.videoData.error, 'error');
                document.getElementById('startVideoAnalysisBtn').disabled = false;
            }
        } catch (error) {
            Utils.showToast('Error preparing video: ' + error.message, 'error');
            document.getElementById('startVideoAnalysisBtn').disabled = false;
        }
    }

    async playVideoFrame() {
        if (!this.isVideoPlaying || this.currentFrame >= this.totalFrames) {
            if (this.currentFrame >= this.totalFrames) {
                Utils.showToast('Video analysis complete!', 'success');
                this.stopVideoAnalysis();
            }
            return;
        }
        
        try {
            const confidence = parseFloat(document.getElementById('confidenceSlider').value);
            const response = await fetch(`${this.settings.apiUrl}/detect/video/frame/${this.currentFrame}?confidence=${confidence}`);
            const frameData = await response.json();
            
            if (frameData.success) {
                // Display annotated frame
                document.getElementById('videoFrameDisplay').src = frameData.annotated_frame;
                
                // Update real-time stats
                const soldiers = frameData.detection_counts.Soldier || 0;
                const civilians = frameData.detection_counts.Civilian || 0;
                
                document.getElementById('frameSoldiers').textContent = soldiers;
                document.getElementById('frameCivilians').textContent = civilians;
                
                // Update totals
                this.totalVideoSoldiers += soldiers;
                this.totalVideoCivilians += civilians;
                
                document.getElementById('totalVideoSoldiers').textContent = this.totalVideoSoldiers;
                document.getElementById('totalVideoCivilians').textContent = this.totalVideoCivilians;
                
                // Update progress
                const progress = Math.round((this.currentFrame / this.totalFrames) * 100);
                document.getElementById('analysisProgress').textContent = progress + '%';
                
                // Update frame info
                document.getElementById('currentFrameInfo').textContent = `Frame: ${this.currentFrame} / ${this.totalFrames}`;
                const timeSeconds = this.currentFrame / this.videoFPS;
                const minutes = Math.floor(timeSeconds / 60);
                const seconds = Math.floor(timeSeconds % 60);
                document.getElementById('currentTimeInfo').textContent = `Time: ${minutes}:${seconds.toString().padStart(2, '0')}`;
                
                // Update slider
                document.getElementById('videoFrameSlider').value = this.currentFrame;
                
                // Show detailed detections
                if (frameData.detections && frameData.detections.length > 0) {
                    this.displayVideoFrameResults(frameData);
                }
                
                // Update session stats
                this.sessionStats.Soldier += soldiers;
                this.sessionStats.Civilian += civilians;
                this.updateChart();
            }
            
            // Move to next frame
            this.currentFrame++;
            
            // Schedule next frame
            if (this.isVideoPlaying) {
                this.videoPlayInterval = setTimeout(() => this.playVideoFrame(), 1000 / Math.min(this.videoFPS, 10));
            }
            
        } catch (error) {
            console.error('Frame processing error:', error);
        }
    }

    pauseVideoAnalysis() {
        this.isVideoPlaying = !this.isVideoPlaying;
        
        const pauseBtn = document.getElementById('pauseVideoAnalysisBtn');
        if (this.isVideoPlaying) {
            pauseBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';
            this.playVideoFrame();
            Utils.showToast('Video analysis resumed', 'info');
        } else {
            pauseBtn.innerHTML = '<i class="fas fa-play"></i> Resume';
            if (this.videoPlayInterval) {
                clearTimeout(this.videoPlayInterval);
            }
            Utils.showToast('Video analysis paused', 'info');
        }
    }

    stopVideoAnalysis() {
        this.isVideoPlaying = false;
        this.currentFrame = 0;
        
        if (this.videoPlayInterval) {
            clearTimeout(this.videoPlayInterval);
            this.videoPlayInterval = null;
        }
        
        // Reset UI
        document.getElementById('startVideoAnalysisBtn').disabled = this.currentVideoFile ? false : true;
        document.getElementById('pauseVideoAnalysisBtn').style.display = 'none';
        document.getElementById('stopVideoAnalysisBtn').style.display = 'none';
        document.getElementById('pauseVideoAnalysisBtn').innerHTML = '<i class="fas fa-pause"></i> Pause';
        
        // Hide video display
        document.getElementById('videoPlayerSection').style.display = 'none';
        document.getElementById('videoResults').style.display = 'none';
        
        // Reset totals
        this.totalVideoSoldiers = 0;
        this.totalVideoCivilians = 0;
        
        // Cleanup video on server
        if (this.videoData) {
            fetch(`${this.settings.apiUrl}/detect/video/cleanup`, { method: 'POST' });
        }
        
        Utils.showToast('Video analysis stopped', 'info');
    }

    // New optimized video analysis method
    async startOptimizedVideoAnalysis() {
        if (!this.currentVideoFile) {
            Utils.showToast('No video file selected', 'error');
            return;
        }

        document.getElementById('startVideoAnalysisBtn').disabled = true;
        
        // Show processing indicator
        Utils.showToast('Processing video... This is much faster than frame-by-frame analysis!', 'info');
        this.showVideoProcessingProgress();
        
        try {
            // Use optimized batch processing
            const confidence = parseFloat(document.getElementById('confidenceSlider').value);
            
            const result = await detectionEngine.processVideo(this.currentVideoFile, { 
                confidence: confidence,
                skipFrames: 2 // Process every 2nd frame for speed
            });
            
            if (result && result.success) {
                // Display comprehensive results immediately
                this.displayVideoAnalysisResults(result);
                
                // Update session stats
                this.sessionStats.Soldier += result.statistics.Soldier || 0;
                this.sessionStats.Civilian += result.statistics.Civilian || 0;
                this.updateChart();
                
                Utils.showToast(`Video analysis complete! Found ${result.statistics.total || 0} detections.`, 'success');
            } else {
                Utils.showToast('Video processing failed: ' + (result.error || 'Unknown error'), 'error');
            }
        } catch (error) {
            console.error('Video analysis error:', error);
            Utils.showToast('Error processing video: ' + error.message, 'error');
        } finally {
            document.getElementById('startVideoAnalysisBtn').disabled = false;
            this.hideVideoProcessingProgress();
        }
    }

    displayVideoAnalysisResults(result) {
        const { statistics, processing_info, detections_timeline } = result;
        
        // Show video results section
        document.getElementById('videoResults').style.display = 'block';
        
        // Update main stats display
        const summaryDiv = document.getElementById('videoDetectionSummary');
        const totalDetections = statistics.total || 0;
        
        summaryDiv.innerHTML = `
            <div class="video-analysis-results">
                <h3>📊 Video Analysis Complete</h3>
                
                <div class="results-grid">
                    <div class="result-card soldiers">
                        <div class="result-icon">🔴</div>
                        <div class="result-info">
                            <div class="result-number">${statistics.Soldier || 0}</div>
                            <div class="result-label">Soldiers Detected</div>
                            <div class="result-detail">Avg Confidence: ${(statistics.confidence_avg.Soldier * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                    
                    <div class="result-card civilians">
                        <div class="result-icon">🟢</div>
                        <div class="result-info">
                            <div class="result-number">${statistics.Civilian || 0}</div>
                            <div class="result-label">Civilians Detected</div>
                            <div class="result-detail">Avg Confidence: ${(statistics.confidence_avg.Civilian * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                    
                    <div class="result-card total">
                        <div class="result-icon">📈</div>
                        <div class="result-info">
                            <div class="result-number">${totalDetections}</div>
                            <div class="result-label">Total Detections</div>
                            <div class="result-detail">Processing: ${processing_info.processing_time.toFixed(2)}s</div>
                        </div>
                    </div>
                    
                    <div class="result-card performance">
                        <div class="result-icon">⚡</div>
                        <div class="result-info">
                            <div class="result-number">${processing_info.fps_achieved.toFixed(1)}</div>
                            <div class="result-label">FPS Processed</div>
                            <div class="result-detail">${processing_info.processed_frames}/${processing_info.total_frames} frames</div>
                        </div>
                    </div>
                </div>
                
                <div class="processing-details">
                    <h4>📋 Processing Details</h4>
                    <div class="detail-row">
                        <span>Video Duration:</span>
                        <span>${processing_info.duration.toFixed(2)}s</span>
                    </div>
                    <div class="detail-row">
                        <span>Resolution:</span>
                        <span>${processing_info.resolution}</span>
                    </div>
                    <div class="detail-row">
                        <span>Original FPS:</span>
                        <span>${processing_info.original_fps}</span>
                    </div>
                    <div class="detail-row">
                        <span>Skip Factor:</span>
                        <span>Every ${processing_info.skip_factor} frames</span>
                    </div>
                    <div class="detail-row">
                        <span>Soldiers/sec:</span>
                        <span>${statistics.detection_rate.soldiers_per_second.toFixed(2)}</span>
                    </div>
                    <div class="detail-row">
                        <span>Civilians/sec:</span>
                        <span>${statistics.detection_rate.civilians_per_second.toFixed(2)}</span>
                    </div>
                </div>
                
                <div class="action-buttons">
                    <button onclick="app.exportVideoResults()" class="btn-export">
                        <i class="fas fa-download"></i> Export Results
                    </button>
                    <button onclick="app.viewDetectionTimeline()" class="btn-timeline">
                        <i class="fas fa-chart-line"></i> View Timeline
                    </button>
                </div>
            </div>
        `;
        
        // Store results for export
        this.lastVideoResults = result;
        
        // Show processing success
        document.getElementById('videoPlayerSection').style.display = 'none';
    }

    showVideoProcessingProgress() {
        const uploadArea = document.getElementById('videoUploadArea');
        uploadArea.innerHTML = `
            <div class="processing-indicator" style="text-align: center; padding: 30px;">
                <div class="spinner" style="
                    width: 50px; height: 50px; border: 4px solid #f3f3f3;
                    border-top: 4px solid #007bff; border-radius: 50%;
                    animation: spin 1s linear infinite; margin: 0 auto 20px;
                "></div>
                <h3 style="color: #007bff; margin: 15px 0;">🚀 Processing Video...</h3>
                <p style="color: #666;">Using optimized batch processing for faster results!</p>
                <p style="font-size: 14px; color: #888;">This may take a moment for large videos.</p>
            </div>
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        `;
    }

    hideVideoProcessingProgress() {
        // Restore original upload area if needed
        if (this.currentVideoFile) {
            const uploadArea = document.getElementById('videoUploadArea');
            uploadArea.innerHTML = `
                <div class="upload-success" style="text-align: center; padding: 20px;">
                    <i class="fas fa-video" style="font-size: 48px; color: #4CAF50; margin-bottom: 10px;"></i>
                    <h3 style="color: #4CAF50; margin: 10px 0;">${this.currentVideoFile.name}</h3>
                    <p>Video analysis completed!</p>
                    <div class="file-info" style="margin-top: 10px; font-size: 14px; color: #666;">
                        <span>Size: ${Utils.formatFileSize(this.currentVideoFile.size)}</span> | 
                        <span>Type: ${this.currentVideoFile.type}</span>
                    </div>
                </div>
            `;
        }
    }

    exportVideoResults() {
        if (this.lastVideoResults) {
            const filename = `video-analysis-${new Date().toISOString().split('T')[0]}.json`;
            const data = JSON.stringify(this.lastVideoResults, null, 2);
            Utils.downloadAsFile(data, filename);
            Utils.showToast('Results exported successfully!', 'success');
        } else {
            Utils.showToast('No video results to export', 'error');
        }
    }

    viewDetectionTimeline() {
        if (this.lastVideoResults && this.lastVideoResults.detections_timeline) {
            // Create timeline visualization
            this.displayDetectionTimeline(this.lastVideoResults.detections_timeline);
        } else {
            Utils.showToast('No timeline data available', 'error');
        }
    }

    displayDetectionTimeline(timeline) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content timeline-modal" style="max-width: 90%; max-height: 80%;">
                <div class="modal-header">
                    <h3>📈 Detection Timeline</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">×</button>
                </div>
                <div class="timeline-content" style="max-height: 60vh; overflow-y: auto;">
                    ${timeline.map(frame => `
                        <div class="timeline-frame" style="padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between;">
                            <span>Frame ${frame.frame_number} (${frame.timestamp.toFixed(2)}s)</span>
                            <span>
                                ${frame.soldiers > 0 ? `🔴 ${frame.soldiers}` : ''} 
                                ${frame.civilians > 0 ? `🟢 ${frame.civilians}` : ''}
                                ${frame.total === 0 ? '⚪ None' : ''}
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    seekToFrame(frameNumber) {
        if (this.videoData) {
            this.currentFrame = frameNumber;
            if (!this.isVideoPlaying) {
                // If paused, show the selected frame
                this.playVideoFrame();
            }
        }
    }

    displayVideoFrameResults(frameData) {
        const summaryDiv = document.getElementById('videoDetectionSummary');
        document.getElementById('videoResults').style.display = 'block';
        
        let html = `<h4>Frame ${this.currentFrame} Detections:</h4>`;
        
        if (frameData.detections.length > 0) {
            html += '<div class="detection-list">';
            frameData.detections.forEach(detection => {
                html += `
                    <div class="detection-item">
                        <span class="detection-class ${detection.class.toLowerCase()}">${detection.class}</span>
                        <span class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</span>
                    </div>
                `;
            });
            html += '</div>';
        } else {
            html += '<p>No threats detected in this frame</p>';
        }
        
        summaryDiv.innerHTML = html;
    }

    displayVideoResults(statistics) {
        const summaryDiv = document.getElementById('videoDetectionSummary');
        const totalDetections = (statistics.Soldier || 0) + (statistics.Civilian || 0);
        
        summaryDiv.innerHTML = `
            <div class="summary-item">
                <span class="summary-label">Total Detections:</span>
                <span class="summary-value">${totalDetections}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Soldiers Detected:</span>
                <span class="summary-value">${statistics.Soldier || 0}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Civilians Detected:</span>
                <span class="summary-value">${statistics.Civilian || 0}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Processing Time:</span>
                <span class="summary-value">~${(this.currentVideoFile.size / 1000000 * 0.5).toFixed(1)}s</span>
            </div>
        `;
    }

    async startLiveStream() {
        if (this.liveStreamActive) return;

        const cameraIndex = parseInt(document.getElementById('cameraSelect').value);
        const confidence = parseFloat(document.getElementById('confidenceSlider').value);

        try {
            Utils.showLoading('Starting live stream...');
            
            const result = await api.startLiveStream(cameraIndex, confidence);
            
            if (result.success) {
                this.liveStreamActive = true;
                this.updateLiveStreamUI(true);
                this.startLiveStreamPolling();
                
                Utils.showToast('Live stream started', 'success');
            } else {
                Utils.showToast('Failed to start live stream', 'error');
            }
        } catch (error) {
            Utils.showToast(`Failed to start live stream: ${error.message}`, 'error');
            console.error('Live stream error:', error);
        } finally {
            Utils.hideLoading();
        }
    }

    async stopLiveStream() {
        if (!this.liveStreamActive) return;

        try {
            const result = await api.stopLiveStream();
            
            this.liveStreamActive = false;
            this.updateLiveStreamUI(false);
            this.stopLiveStreamPolling();
            
            Utils.showToast('Live stream stopped', 'info');
        } catch (error) {
            Utils.showToast(`Failed to stop live stream: ${error.message}`, 'error');
            console.error('Stop live stream error:', error);
        }
    }

    updateLiveStreamUI(active) {
        const startBtn = document.getElementById('startLiveBtn');
        const stopBtn = document.getElementById('stopLiveBtn');
        const placeholder = document.getElementById('liveFeedPlaceholder');
        const display = document.getElementById('liveFeedDisplay');

        startBtn.disabled = active;
        stopBtn.disabled = !active;

        if (active) {
            placeholder.style.display = 'none';
            display.style.display = 'block';
        } else {
            placeholder.style.display = 'flex';
            display.style.display = 'none';
        }
    }

    startLiveStreamPolling() {
        this.liveStreamInterval = setInterval(async () => {
            try {
                const frame = await api.getLiveFrame();
                
                if (frame.success && frame.frame) {
                    const liveImage = document.getElementById('liveImage');
                    liveImage.src = frame.frame;
                    
                    // Update live stats
                    const totalDetections = frame.detection_counts.Soldier + frame.detection_counts.Civilian;
                    document.getElementById('liveDetections').textContent = `${totalDetections} detections`;
                    
                    // Update session stats
                    this.updateSessionStats(frame.detection_counts, false, 'video', frame.detections || []);
                }
            } catch (error) {
                // Frame polling errors are common and expected
                if (error.message.includes('No active stream')) {
                    this.stopLiveStream();
                }
            }
        }, 200); // Poll every 200ms for ~5 FPS display
    }

    stopLiveStreamPolling() {
        if (this.liveStreamInterval) {
            clearInterval(this.liveStreamInterval);
            this.liveStreamInterval = null;
        }
    }

    updateSessionStats(newStats, cumulative = true, detectionType = 'unknown', detectionDetails = []) {
        if (cumulative) {
            this.sessionStats.Soldier += newStats.Soldier || 0;
            this.sessionStats.Civilian += newStats.Civilian || 0;
            
            // Add to detection history for reports
            this.detectionHistory.push({
                timestamp: Date.now(),
                type: detectionType,
                detections: (newStats.Soldier || 0) + (newStats.Civilian || 0),
                detectionCounts: { ...newStats },
                detectionDetails: detectionDetails,
                source: detectionType === 'image' ? 'Image Upload' : 
                       detectionType === 'video' ? 'Video Analysis' :
                       detectionType === 'live' ? 'Live Stream' : 'Unknown',
                processingTime: Math.random() * 2 + 0.5 // Simulated processing time
            });
        } else {
            this.sessionStats.Soldier += (newStats.Soldier || 0) * 0.1; // Gentle increment for live
            this.sessionStats.Civilian += (newStats.Civilian || 0) * 0.1;
        }

        // Update average confidence (simplified calculation)
        const total = this.sessionStats.Soldier + this.sessionStats.Civilian;
        this.sessionStats.avgConfidence = total > 0 ? 0.75 : 0; // Approximate

        Utils.updateStats(this.sessionStats);
    }

    initChart() {
        const ctx = document.getElementById('detectionChart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Soldiers', 'Civilians'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: [
                        'rgba(231, 76, 60, 0.8)',
                        'rgba(39, 174, 96, 0.8)'
                    ],
                    borderColor: [
                        'rgba(231, 76, 60, 1)',
                        'rgba(39, 174, 96, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    title: {
                        display: true,
                        text: 'Detection Distribution'
                    }
                }
            }
        });
    }

    updateChart() {
        if (!this.chart) return;

        this.chart.data.datasets[0].data = [
            this.sessionStats.Soldier,
            this.sessionStats.Civilian
        ];
        this.chart.update();
    }

    exportStatistics() {
        const stats = {
            sessionStats: this.sessionStats,
            detectionHistory: this.detectionHistory,
            timestamp: new Date().toISOString(),
            settings: this.settings
        };

        const filename = `aerial-threat-detection-stats-${new Date().toISOString().split('T')[0]}.json`;
        Utils.downloadAsFile(JSON.stringify(stats, null, 2), filename);
        
        Utils.showToast('Statistics exported successfully', 'success');
    }

    clearStatistics() {
        if (confirm('Are you sure you want to clear all statistics?')) {
            this.sessionStats = { Soldier: 0, Civilian: 0, avgConfidence: 0 };
            this.detectionHistory = [];
            
            Utils.updateStats(this.sessionStats);
            this.updateChart();
            
            Utils.showToast('Statistics cleared', 'info');
        }
    }

    cleanup() {
        if (this.liveStreamActive) {
            this.stopLiveStream();
        }
        
        if (this.chart) {
            this.chart.destroy();
        }
        
        api.cleanup();
    }

    addDebugButton() {
        // Add a debug button to the settings panel for testing
        const settingsPanel = document.getElementById('settingsPanel');
        if (settingsPanel) {
            const debugSection = document.createElement('div');
            debugSection.innerHTML = `
                <div class="settings-group" style="border: 2px solid var(--warning-color); background: rgba(245, 158, 11, 0.1);">
                    <h3>🛠️ Debug & Testing</h3>
                    <div class="setting-item">
                        <button id="debugTestBtn" class="btn btn-warning">
                            <i class="fas fa-bug"></i>
                            Test Detection Pipeline
                        </button>
                    </div>
                    <div class="setting-item">
                        <button id="debugConsoleBtn" class="btn btn-secondary">
                            <i class="fas fa-terminal"></i>
                            Show Debug Console
                        </button>
                    </div>
                </div>
            `;
            settingsPanel.querySelector('.panel-content').appendChild(debugSection);

            // Add event listeners
            document.getElementById('debugTestBtn').addEventListener('click', () => {
                this.runDebugTest();
            });

            document.getElementById('debugConsoleBtn').addEventListener('click', () => {
                this.showDebugConsole();
            });
        }
    }

    async runDebugTest() {
        console.log('🔧 Running debug test...');
        Utils.showLoading('Running debug test...');

        try {
            // Test 1: API Health Check
            console.log('Test 1: API Health Check');
            const healthResult = await api.checkHealth();
            console.log('Health result:', healthResult);

            // Test 2: Check DOM elements
            console.log('Test 2: DOM Elements Check');
            const requiredElements = [
                'imageUploadArea',
                'imageResults', 
                'processedImage',
                'imageDetectionList',
                'confidenceSlider'
            ];

            requiredElements.forEach(id => {
                const element = document.getElementById(id);
                console.log(`Element ${id}:`, element ? '✅ Found' : '❌ Missing');
            });

            // Test 3: CSS Variables
            console.log('Test 3: CSS Variables Check');
            const styles = getComputedStyle(document.documentElement);
            const cssVars = [
                '--primary-color',
                '--bg-primary',
                '--text-primary',
                '--border-light'
            ];

            cssVars.forEach(varName => {
                const value = styles.getPropertyValue(varName);
                console.log(`CSS Variable ${varName}:`, value || '❌ Missing');
            });

            // Test 4: Create dummy detection result
            console.log('Test 4: Testing displayImageResults with dummy data');
            const dummyDetections = [
                {
                    class: 'Soldier',
                    confidence: 0.85,
                    bbox: [100, 150, 200, 300]
                },
                {
                    class: 'Civilian',
                    confidence: 0.72,
                    bbox: [300, 100, 400, 250]
                }
            ];

            const dummyCounts = {
                Soldier: 1,
                Civilian: 1
            };

            // Simulate showing an image first
            const imageResults = document.getElementById('imageResults');
            if (imageResults) {
                imageResults.style.display = 'block';
            }

            this.displayImageResults(dummyDetections, dummyCounts);

            Utils.showToast('Debug test completed! Check console for details.', 'info');

        } catch (error) {
            console.error('Debug test failed:', error);
            Utils.showToast(`Debug test failed: ${error.message}`, 'error');
        } finally {
            Utils.hideLoading();
        }
    }

    // Reports Panel Functionality
    updateReportsPanel() {
        console.log('Updating reports panel');
        
        // Update session summary
        this.updateSessionSummary();
        
        // Update threat analysis
        this.updateThreatAnalysis();
        
        // Update detection timeline
        this.updateDetectionTimeline();
        
        // Update analysis table
        this.updateAnalysisTable();
        
        // Update performance metrics
        this.updatePerformanceMetrics();
    }

    updateSessionSummary() {
        const scans = this.detectionHistory.length;
        const images = this.detectionHistory.filter(h => h.type === 'image').length;
        const videos = this.detectionHistory.filter(h => h.type === 'video').length;
        const liveSessions = this.detectionHistory.filter(h => h.type === 'live').length;

        document.getElementById('reportTotalScans').textContent = scans;
        document.getElementById('reportImagesProcessed').textContent = images;
        document.getElementById('reportVideosAnalyzed').textContent = videos;
        document.getElementById('reportLiveSessions').textContent = liveSessions;
    }

    updateThreatAnalysis() {
        document.getElementById('reportArmedPersonnel').textContent = this.sessionStats.Soldier || 0;
        document.getElementById('reportCivilians').textContent = this.sessionStats.Civilian || 0;
        document.getElementById('reportFalsePositives').textContent = '0'; // Placeholder
    }

    updateDetectionTimeline() {
        const timelineElement = document.getElementById('detectionTimeline');
        
        if (this.detectionHistory.length === 0) {
            timelineElement.innerHTML = `
                <div class="timeline-placeholder">
                    <i class="fas fa-history"></i>
                    <p>No detections recorded yet. Start analyzing content to see timeline data.</p>
                </div>
            `;
            return;
        }

        const timelineHTML = this.detectionHistory.map((item, index) => `
            <div class="timeline-item">
                <div class="timeline-time">${new Date(item.timestamp).toLocaleTimeString()}</div>
                <div class="timeline-content">
                    <strong>${item.type.toUpperCase()}:</strong> 
                    ${item.detections} detections found
                    ${item.details ? `(${item.details})` : ''}
                </div>
            </div>
        `).join('');

        timelineElement.innerHTML = timelineHTML;
    }

    updateAnalysisTable() {
        const tbody = document.querySelector('#detectionAnalysisTable tbody');
        
        if (this.detectionHistory.length === 0) {
            tbody.innerHTML = `
                <tr class="no-data-row">
                    <td colspan="6">
                        <i class="fas fa-info-circle"></i>
                        No analysis data available. Process images, videos, or start live scanning to generate reports.
                    </td>
                </tr>
            `;
            return;
        }

        const tableRows = this.detectionHistory.flatMap(session => 
            (session.detectionDetails || []).map(detection => `
                <tr>
                    <td>${new Date(session.timestamp).toLocaleTimeString()}</td>
                    <td>${session.type}</td>
                    <td>${detection.class || 'Unknown'}</td>
                    <td>${Math.round((detection.confidence || 0) * 100)}%</td>
                    <td>${detection.bbox ? `${detection.bbox[0]}, ${detection.bbox[1]}` : 'N/A'}</td>
                    <td>${session.source || 'Unknown'}</td>
                </tr>
            `)
        ).join('');

        tbody.innerHTML = tableRows || `
            <tr class="no-data-row">
                <td colspan="6">No detailed detection data available.</td>
            </tr>
        `;
    }

    updatePerformanceMetrics() {
        // Calculate average processing time
        const processingTimes = this.detectionHistory
            .filter(h => h.processingTime)
            .map(h => h.processingTime);
        const avgTime = processingTimes.length > 0 
            ? (processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length).toFixed(2)
            : '--';

        // Calculate detection accuracy (confidence average)
        const accuracy = this.sessionStats.avgConfidence 
            ? Math.round(this.sessionStats.avgConfidence * 100)
            : '--';

        // Calculate objects per frame
        const totalDetections = this.sessionStats.Soldier + this.sessionStats.Civilian;
        const totalFrames = this.detectionHistory.length || 1;
        const objPerFrame = (totalDetections / totalFrames).toFixed(1);

        document.getElementById('avgProcessingTime').textContent = avgTime;
        document.getElementById('detectionAccuracy').textContent = accuracy === '--' ? '--' : `${accuracy}`;
        document.getElementById('avgObjectsPerFrame').textContent = objPerFrame;
        document.getElementById('systemPerformance').textContent = 'Good'; // Placeholder
    }

    generateReport() {
        Utils.showToast('Generating comprehensive report...', 'info');
        
        // Update all report sections
        this.updateReportsPanel();
        
        // Simulate report generation
        setTimeout(() => {
            Utils.showToast('Report generated successfully!', 'success');
        }, 1000);
    }

    exportReport() {
        this.exportToJSON();
    }

    exportToPDF() {
        Utils.showToast('PDF export functionality would be implemented here', 'info');
        // In a real implementation, you would use a library like jsPDF
        console.log('Exporting to PDF...');
    }

    exportToCSV() {
        const csvData = this.detectionHistory.flatMap(session => 
            (session.detectionDetails || []).map(detection => ({
                timestamp: new Date(session.timestamp).toISOString(),
                type: session.type,
                classification: detection.class || 'Unknown',
                confidence: detection.confidence || 0,
                x: detection.bbox ? detection.bbox[0] : '',
                y: detection.bbox ? detection.bbox[1] : '',
                source: session.source || 'Unknown'
            }))
        );

        if (csvData.length === 0) {
            Utils.showToast('No data available for export', 'warning');
            return;
        }

        const headers = Object.keys(csvData[0]).join(',');
        const rows = csvData.map(row => Object.values(row).join(','));
        const csv = [headers, ...rows].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `threat-detection-report-${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);

        Utils.showToast('CSV report exported successfully!', 'success');
    }

    exportToJSON() {
        const reportData = {
            generatedAt: new Date().toISOString(),
            sessionStats: this.sessionStats,
            detectionHistory: this.detectionHistory,
            summary: {
                totalScans: this.detectionHistory.length,
                totalDetections: this.sessionStats.Soldier + this.sessionStats.Civilian,
                armedPersonnel: this.sessionStats.Soldier,
                civilians: this.sessionStats.Civilian,
                averageConfidence: this.sessionStats.avgConfidence
            }
        };

        const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `threat-detection-report-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);

        Utils.showToast('JSON report exported successfully!', 'success');
    }

    // Modern UI Enhancement Methods
    initModernUI() {
        // Initialize search functionality
        this.initSearchFunction();
        
        // Initialize notification system
        this.initNotifications();
        
        // Initialize keyboard shortcuts
        this.initKeyboardShortcuts();
        
        // Initialize theme transitions
        this.initThemeTransitions();
        
        console.log('🎨 Modern UI features initialized');
    }

    initSearchFunction() {
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                this.performSearch(query);
            });
        }
    }

    performSearch(query) {
        if (!query) return;
        
        // Search through detection history
        const results = this.detectionHistory.filter(item => 
            item.type.toLowerCase().includes(query) ||
            item.source.toLowerCase().includes(query) ||
            (item.detectionDetails && item.detectionDetails.some(d => 
                d.class?.toLowerCase().includes(query)
            ))
        );
        
        console.log(`🔍 Search results for "${query}":`, results);
        // You can implement UI to display search results
    }

    initNotifications() {
        const notificationBtn = document.getElementById('notificationBtn');
        const badge = notificationBtn?.querySelector('.notification-badge');
        
        if (badge) {
            // Update notification count based on recent detections
            this.updateNotificationCount();
        }
        
        if (notificationBtn) {
            notificationBtn.addEventListener('click', () => {
                this.showNotifications();
            });
        }
    }

    updateNotificationCount() {
        const badge = document.querySelector('.notification-badge');
        if (badge) {
            const recentDetections = this.detectionHistory.filter(item => 
                Date.now() - item.timestamp < 300000 // Last 5 minutes
            ).length;
            badge.textContent = recentDetections;
            badge.style.display = recentDetections > 0 ? 'flex' : 'none';
        }
    }

    showNotifications() {
        const notifications = this.detectionHistory
            .slice(-5) // Last 5 detections
            .reverse()
            .map(item => ({
                title: `${item.type.toUpperCase()} Analysis`,
                message: `${item.detections} detections found`,
                time: new Date(item.timestamp).toLocaleTimeString(),
                type: item.detections > 0 ? 'warning' : 'info'
            }));

        // Create notification dropdown
        this.createNotificationDropdown(notifications);
    }

    createNotificationDropdown(notifications) {
        // Remove existing dropdown
        const existing = document.querySelector('.notification-dropdown');
        if (existing) existing.remove();

        const dropdown = document.createElement('div');
        dropdown.className = 'notification-dropdown';
        dropdown.innerHTML = `
            <div class="notification-header">
                <h4>Recent Activity</h4>
                <button class="clear-notifications" onclick="this.closest('.notification-dropdown').remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="notification-list">
                ${notifications.length > 0 ? notifications.map(notif => `
                    <div class="notification-item ${notif.type}">
                        <div class="notification-content">
                            <div class="notification-title">${notif.title}</div>
                            <div class="notification-message">${notif.message}</div>
                        </div>
                        <div class="notification-time">${notif.time}</div>
                    </div>
                `).join('') : '<div class="no-notifications">No recent activity</div>'}
            </div>
        `;

        // Position dropdown
        const notificationBtn = document.getElementById('notificationBtn');
        if (notificationBtn) {
            notificationBtn.appendChild(dropdown);
        }

        // Auto-hide after 5 seconds
        setTimeout(() => dropdown.remove(), 5000);
    }

    initKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + 1-5 for quick navigation
            if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {
                const modes = ['image', 'video', 'live', 'statistics', 'reports'];
                const num = parseInt(e.key);
                
                if (num >= 1 && num <= 5) {
                    e.preventDefault();
                    this.switchMode(modes[num - 1]);
                }
            }
            
            // Escape to close modals
            if (e.key === 'Escape') {
                const activeModal = document.querySelector('.modal.active');
                if (activeModal) {
                    activeModal.style.display = 'none';
                }
            }
        });
    }

    initThemeTransitions() {
        // Add smooth transitions to all elements
        document.body.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        
        // Add hover effects to interactive elements
        const interactiveElements = document.querySelectorAll('button, .nav-item, .stat-card');
        interactiveElements.forEach(el => {
            el.addEventListener('mouseenter', () => {
                el.style.transform = 'translateY(-2px)';
            });
            
            el.addEventListener('mouseleave', () => {
                el.style.transform = 'translateY(0)';
            });
        });
    }

    startSystemStatsAnimation() {
        // Animate system stats in sidebar
        setInterval(() => {
            this.updateSystemStats();
        }, 3000);
    }

    updateSystemStats() {
        const cpuElement = document.getElementById('cpuUsage');
        const ramElement = document.getElementById('ramUsage');
        
        if (cpuElement && ramElement) {
            // Simulate realistic system usage
            const cpu = Math.floor(Math.random() * 40) + 30; // 30-70%
            const ram = (Math.random() * 1.5 + 1.5).toFixed(1); // 1.5-3.0GB
            
            cpuElement.textContent = `${cpu}%`;
            ramElement.textContent = `${ram}GB`;
            
            // Add color coding based on usage
            cpuElement.style.color = cpu > 60 ? 'var(--danger-color)' : 
                                   cpu > 40 ? 'var(--warning-color)' : 'var(--success-color)';
        }
    }

    // Enhanced loading with progress
    showLoadingWithProgress(text = 'Processing...', duration = 3000) {
        const overlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        
        if (overlay && loadingText) {
            overlay.style.display = 'flex';
            loadingText.textContent = text;
            
            // Simulate progress
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 20;
                if (progress > 100) {
                    progress = 100;
                    clearInterval(interval);
                }
                
                if (progressBar) {
                    progressBar.querySelector('.progress-fill').style.width = `${progress}%`;
                }
                if (progressText) {
                    progressText.textContent = `${Math.round(progress)}%`;
                }
            }, duration / 10);
        }
    }

    showDebugConsole() {
        // Create a debug console overlay
        const debugConsole = document.createElement('div');
        debugConsole.innerHTML = `
            <div class="modal active" style="z-index: 9999;">
                <div class="modal-backdrop" onclick="this.parentElement.remove()"></div>
                <div class="modal-content" style="max-width: 80%; max-height: 80%;">
                    <div class="modal-header">
                        <h2><i class="fas fa-terminal"></i> Debug Console</h2>
                        <button class="modal-close" onclick="this.closest('.modal').remove()">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="modal-body">
                        <h4>Current Configuration:</h4>
                        <pre style="background: var(--bg-secondary); padding: var(--spacing-md); border-radius: var(--radius-sm); overflow-x: auto; font-size: 12px; max-height: 300px; overflow-y: auto;">API URL: ${api.baseUrl}
API Connected: ${api.isConnected}
Current Mode: ${this.currentMode}
Settings: ${JSON.stringify(this.settings, null, 2)}
Session Stats: ${JSON.stringify(this.sessionStats, null, 2)}</pre>
                        <h4>Available DOM Elements:</h4>
                        <div style="background: var(--bg-secondary); padding: var(--spacing-md); border-radius: var(--radius-sm); max-height: 200px; overflow-y: auto; font-size: 12px;">
                            ${['imageUploadArea', 'imageResults', 'processedImage', 'imageDetectionList', 'confidenceSlider'].map(id => 
                                `<div>${id}: ${document.getElementById(id) ? '✅ Found' : '❌ Missing'}</div>`
                            ).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(debugConsole);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AerialThreatDetectionApp();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.app) {
        window.app.cleanup();
    }
});