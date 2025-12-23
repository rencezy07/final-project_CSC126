// Detection-specific functionality

class DetectionEngine {
    constructor() {
        this.processingQueue = [];
        this.isProcessing = false;
        this.detectionCache = new Map();
        this.batchResults = [];
    }

    /**
     * Add detection job to queue
     */
    async queueDetection(file, type, options = {}) {
        const job = {
            id: Utils.generateId(),
            file: file,
            type: type, // 'image' or 'video'
            options: options,
            timestamp: Utils.getCurrentTimestamp(),
            status: 'queued'
        };

        this.processingQueue.push(job);
        
        if (!this.isProcessing) {
            this.processQueue();
        }

        return job.id;
    }

    /**
     * Process detection queue
     */
    async processQueue() {
        if (this.isProcessing || this.processingQueue.length === 0) {
            return;
        }

        this.isProcessing = true;

        while (this.processingQueue.length > 0) {
            const job = this.processingQueue.shift();
            
            try {
                job.status = 'processing';
                let result;

                if (job.type === 'image') {
                    result = await this.processImage(job.file, job.options);
                } else if (job.type === 'video') {
                    result = await this.processVideo(job.file, job.options);
                }

                job.status = 'completed';
                job.result = result;
                
                // Cache successful results
                if (result && result.success) {
                    this.detectionCache.set(job.id, result);
                }

                // Dispatch completion event
                this.dispatchDetectionEvent('detection-completed', job);

            } catch (error) {
                job.status = 'failed';
                job.error = error.message;
                
                console.error(`Detection job ${job.id} failed:`, error);
                this.dispatchDetectionEvent('detection-failed', job);
            }
        }

        this.isProcessing = false;
    }

    /**
     * Process single image
     */
    async processImage(file, options = {}) {
        const confidence = options.confidence || 0.5;
        const cacheKey = `${file.name}-${file.lastModified}-${confidence}`;

        // Check cache first
        if (this.detectionCache.has(cacheKey)) {
            return this.detectionCache.get(cacheKey);
        }

        const result = await api.detectImage(file, confidence);
        
        // Cache result
        this.detectionCache.set(cacheKey, result);
        
        return result;
    }

    /**
     * Process single video with optimized backend processing
     */
    async processVideo(file, options = {}) {
        const confidence = options.confidence || 0.5;
        const skipFrames = options.skipFrames || 2; // Process every 2nd frame for speed
        
        const formData = new FormData();
        formData.append('video', file);
        formData.append('confidence', confidence.toString());
        formData.append('skip_frames', skipFrames.toString());
        
        try {
            const result = await api.processVideoOptimized(formData);
            return result;
        } catch (error) {
            console.error('Video processing failed:', error);
            throw error;
        }
    }

    /**
     * Process multiple files
     */
    async processBatch(files, type, options = {}) {
        this.batchResults = [];
        const jobIds = [];

        for (const file of files) {
            const jobId = await this.queueDetection(file, type, options);
            jobIds.push(jobId);
        }

        return new Promise((resolve) => {
            const checkCompletion = () => {
                const completedJobs = this.batchResults.filter(job => 
                    jobIds.includes(job.id) && (job.status === 'completed' || job.status === 'failed')
                );

                if (completedJobs.length === jobIds.length) {
                    resolve(completedJobs);
                } else {
                    setTimeout(checkCompletion, 100);
                }
            };

            checkCompletion();
        });
    }

    /**
     * Dispatch detection events
     */
    dispatchDetectionEvent(eventType, jobData) {
        const event = new CustomEvent(eventType, {
            detail: jobData
        });
        
        document.dispatchEvent(event);
        
        // Add to batch results for tracking
        this.batchResults.push(jobData);
    }

    /**
     * Get detection statistics
     */
    getDetectionStats() {
        const completed = this.batchResults.filter(job => job.status === 'completed');
        const failed = this.batchResults.filter(job => job.status === 'failed');
        
        let totalSoldiers = 0;
        let totalCivilians = 0;
        let totalConfidence = 0;
        let detectionCount = 0;

        completed.forEach(job => {
            if (job.result && job.result.success) {
                if (job.result.detection_counts) {
                    totalSoldiers += job.result.detection_counts.Soldier || 0;
                    totalCivilians += job.result.detection_counts.Civilian || 0;
                }

                if (job.result.detections) {
                    job.result.detections.forEach(detection => {
                        totalConfidence += detection.confidence;
                        detectionCount++;
                    });
                }
            }
        });

        return {
            totalJobs: this.batchResults.length,
            completedJobs: completed.length,
            failedJobs: failed.length,
            successRate: completed.length / this.batchResults.length * 100,
            totalSoldiers,
            totalCivilians,
            averageConfidence: detectionCount > 0 ? totalConfidence / detectionCount : 0,
            processingTime: this.calculateProcessingTime()
        };
    }

    /**
     * Calculate total processing time
     */
    calculateProcessingTime() {
        const completed = this.batchResults.filter(job => job.status === 'completed');
        
        if (completed.length === 0) return 0;
        
        const startTime = Math.min(...completed.map(job => job.timestamp));
        const endTime = Math.max(...completed.map(job => job.timestamp));
        
        return (endTime - startTime) / 1000; // Return in seconds
    }

    /**
     * Clear cache and results
     */
    clearCache() {
        this.detectionCache.clear();
        this.batchResults = [];
    }

    /**
     * Get cache size
     */
    getCacheSize() {
        return this.detectionCache.size;
    }

    /**
     * Export detection results
     */
    exportResults(format = 'json') {
        const stats = this.getDetectionStats();
        const exportData = {
            statistics: stats,
            results: this.batchResults,
            cache_size: this.getCacheSize(),
            export_timestamp: new Date().toISOString()
        };

        let filename;
        let data;

        switch (format.toLowerCase()) {
            case 'json':
                filename = `detection-results-${new Date().toISOString().split('T')[0]}.json`;
                data = JSON.stringify(exportData, null, 2);
                break;
                
            case 'csv':
                filename = `detection-results-${new Date().toISOString().split('T')[0]}.csv`;
                data = this.convertToCSV(this.batchResults);
                break;
                
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }

        Utils.downloadAsFile(data, filename);
    }

    /**
     * Convert results to CSV format
     */
    convertToCSV(results) {
        const headers = ['File', 'Type', 'Status', 'Soldiers', 'Civilians', 'Total Detections', 'Processing Time'];
        const rows = [headers.join(',')];

        results.forEach(job => {
            const row = [
                job.file ? job.file.name : 'Unknown',
                job.type || 'Unknown',
                job.status || 'Unknown',
                job.result?.detection_counts?.Soldier || 0,
                job.result?.detection_counts?.Civilian || 0,
                job.result?.total_detections || 0,
                job.result?.processing_time || 0
            ];
            
            rows.push(row.join(','));
        });

        return rows.join('\n');
    }
}

/**
 * Real-time detection monitor for live streams
 */
class LiveDetectionMonitor {
    constructor() {
        this.isActive = false;
        this.detectionHistory = [];
        this.frameRate = 0;
        this.frameTimestamps = [];
        this.currentStats = { soldiers: 0, civilians: 0 };
        this.alerts = [];
        this.alertThreshold = 5; // Alert when soldiers > 5
    }

    start() {
        this.isActive = true;
        this.detectionHistory = [];
        this.frameTimestamps = [];
        this.currentStats = { soldiers: 0, civilians: 0 };
        
        console.log('Live detection monitor started');
    }

    stop() {
        this.isActive = false;
        console.log('Live detection monitor stopped');
    }

    /**
     * Process live frame data
     */
    processFrame(frameData) {
        if (!this.isActive) return;

        const timestamp = Utils.getCurrentTimestamp();
        this.frameTimestamps.push(timestamp);
        
        // Keep only last 30 timestamps for FPS calculation
        if (this.frameTimestamps.length > 30) {
            this.frameTimestamps.shift();
        }

        // Calculate FPS
        this.frameRate = Utils.calculateFPS(this.frameTimestamps);

        // Process detections
        if (frameData.detections && frameData.detections.length > 0) {
            const frameDetection = {
                timestamp: timestamp,
                detections: frameData.detections,
                soldier_count: frameData.detection_counts?.Soldier || 0,
                civilian_count: frameData.detection_counts?.Civilian || 0,
                total_count: frameData.detections.length
            };

            this.detectionHistory.push(frameDetection);
            this.updateCurrentStats(frameDetection);
            this.checkAlerts(frameDetection);

            // Keep only last 1000 frames to prevent memory issues
            if (this.detectionHistory.length > 1000) {
                this.detectionHistory.shift();
            }
        }

        // Update UI
        this.updateLiveUI();
    }

    /**
     * Update current statistics
     */
    updateCurrentStats(frameDetection) {
        // Use exponential moving average for smooth statistics
        const alpha = 0.1;
        this.currentStats.soldiers = this.currentStats.soldiers * (1 - alpha) + frameDetection.soldier_count * alpha;
        this.currentStats.civilians = this.currentStats.civilians * (1 - alpha) + frameDetection.civilian_count * alpha;
    }

    /**
     * Check for alerts
     */
    checkAlerts(frameDetection) {
        if (frameDetection.soldier_count >= this.alertThreshold) {
            const alert = {
                type: 'high_soldier_count',
                message: `High soldier count detected: ${frameDetection.soldier_count}`,
                timestamp: frameDetection.timestamp,
                data: frameDetection
            };

            this.alerts.push(alert);
            this.triggerAlert(alert);

            // Keep only last 50 alerts
            if (this.alerts.length > 50) {
                this.alerts.shift();
            }
        }
    }

    /**
     * Trigger alert notification
     */
    triggerAlert(alert) {
        Utils.showToast(alert.message, 'warning', 10000);
        
        // Custom event for alert handling
        const alertEvent = new CustomEvent('detection-alert', {
            detail: alert
        });
        document.dispatchEvent(alertEvent);
        
        console.warn('Detection Alert:', alert);
    }

    /**
     * Update live UI elements
     */
    updateLiveUI() {
        const fpsElement = document.getElementById('liveFPS');
        if (fpsElement) {
            fpsElement.textContent = this.frameRate.toFixed(1);
        }

        // Update live stats if in statistics panel
        if (window.app && window.app.currentMode === 'statistics') {
            const currentDetections = Math.round(this.currentStats.soldiers + this.currentStats.civilians);
            document.getElementById('liveDetections').textContent = `${currentDetections} avg detections`;
        }
    }

    /**
     * Get detection timeline for visualization
     */
    getDetectionTimeline(minutes = 5) {
        const cutoffTime = Utils.getCurrentTimestamp() - (minutes * 60 * 1000);
        return this.detectionHistory.filter(detection => detection.timestamp >= cutoffTime);
    }

    /**
     * Get alert history
     */
    getAlertHistory(hours = 1) {
        const cutoffTime = Utils.getCurrentTimestamp() - (hours * 60 * 60 * 1000);
        return this.alerts.filter(alert => alert.timestamp >= cutoffTime);
    }

    /**
     * Export monitoring data
     */
    exportMonitoringData() {
        const data = {
            detection_history: this.detectionHistory,
            alerts: this.alerts,
            current_stats: this.currentStats,
            frame_rate: this.frameRate,
            monitoring_duration: this.getMonitoringDuration(),
            export_timestamp: new Date().toISOString()
        };

        const filename = `live-monitoring-data-${new Date().toISOString().split('T')[0]}.json`;
        Utils.downloadAsFile(JSON.stringify(data, null, 2), filename);
        
        Utils.showToast('Monitoring data exported', 'success');
    }

    /**
     * Get total monitoring duration
     */
    getMonitoringDuration() {
        if (this.detectionHistory.length < 2) return 0;
        
        const firstDetection = this.detectionHistory[0].timestamp;
        const lastDetection = this.detectionHistory[this.detectionHistory.length - 1].timestamp;
        
        return (lastDetection - firstDetection) / 1000; // Return in seconds
    }

    /**
     * Clear monitoring data
     */
    clearHistory() {
        this.detectionHistory = [];
        this.alerts = [];
        this.frameTimestamps = [];
        this.currentStats = { soldiers: 0, civilians: 0 };
        
        console.log('Live detection monitor history cleared');
    }
}

// Create global instances
const detectionEngine = new DetectionEngine();
const liveMonitor = new LiveDetectionMonitor();

// Set up event listeners for detection events
document.addEventListener('detection-completed', (event) => {
    const job = event.detail;
    console.log('Detection completed:', job.id);
    
    if (job.result && job.result.success) {
        // Update app statistics if available
        if (window.app) {
            window.app.updateSessionStats(job.result.detection_counts);
        }
    }
});

document.addEventListener('detection-failed', (event) => {
    const job = event.detail;
    console.error('Detection failed:', job.id, job.error);
    Utils.showToast(`Detection failed: ${job.error}`, 'error');
});

document.addEventListener('detection-alert', (event) => {
    const alert = event.detail;
    console.warn('Detection alert triggered:', alert);
    
    // Could trigger additional actions like:
    // - Sound notifications
    // - Email alerts
    // - Automatic recording
    // - Emergency protocols
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DetectionEngine, LiveDetectionMonitor, detectionEngine, liveMonitor };
}