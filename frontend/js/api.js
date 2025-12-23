// API communication module

class API {
    constructor() {
        this.baseUrl = 'http://localhost:5000/api';
        this.headers = {
            'Content-Type': 'application/json'
        };
        this.isConnected = false;
        this.checkConnectionInterval = null;
        
        // Start connection monitoring
        this.startConnectionMonitoring();
    }

    /**
     * Make HTTP request with error handling
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: this.headers,
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    /**
     * Check API health
     */
    async checkHealth() {
        try {
            const response = await this.request('/health', {
                method: 'GET'
            });
            
            this.isConnected = response.status === 'healthy';
            this.updateConnectionStatus();
            return response;
        } catch (error) {
            this.isConnected = false;
            this.updateConnectionStatus();
            throw error;
        }
    }

    /**
     * Upload and detect objects in image
     */
    async detectImage(imageFile, confidence = 0.5) {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('confidence', confidence.toString());

        try {
            const response = await fetch(`${this.baseUrl}/detect/image`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Detection failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Image detection failed:', error);
            throw error;
        }
    }

    /**
     * Upload and process video with optimized backend processing
     */
    async processVideo(videoFile, confidence = 0.5) {
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('confidence', confidence.toString());
        formData.append('skip_frames', '2'); // Process every 2nd frame for speed

        try {
            const response = await fetch(`${this.baseUrl}/detect/video/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Video processing failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Video processing failed:', error);
            throw error;
        }
    }

    /**
     * Process video with optimized batch processing (alternative method)
     */
    async processVideoOptimized(formData) {
        try {
            const response = await fetch(`${this.baseUrl}/detect/video/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Video processing failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Optimized video processing failed:', error);
            throw error;
        }
    }

    /**
     * Start live video stream
     */
    async startLiveStream(cameraIndex = 0, confidence = 0.5) {
        try {
            const response = await this.request('/stream/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    camera_index: cameraIndex,
                    confidence: confidence
                })
            });

            return response;
        } catch (error) {
            console.error('Failed to start live stream:', error);
            throw error;
        }
    }

    /**
     * Stop live video stream
     */
    async stopLiveStream() {
        try {
            const response = await this.request('/stream/stop', {
                method: 'POST'
            });

            return response;
        } catch (error) {
            console.error('Failed to stop live stream:', error);
            throw error;
        }
    }

    /**
     * Get current frame from live stream
     */
    async getLiveFrame() {
        try {
            const response = await this.request('/stream/frame', {
                method: 'GET'
            });

            return response;
        } catch (error) {
            // Don't log errors for frame requests as they're frequent
            throw error;
        }
    }

    /**
     * Get detection statistics
     */
    async getStats() {
        try {
            const response = await this.request('/stats', {
                method: 'GET'
            });

            return response;
        } catch (error) {
            console.error('Failed to get statistics:', error);
            throw error;
        }
    }

    /**
     * Start monitoring connection status
     */
    startConnectionMonitoring() {
        this.checkConnectionInterval = setInterval(async () => {
            try {
                await this.checkHealth();
            } catch (error) {
                // Connection check failed, status already updated
            }
        }, 5000); // Check every 5 seconds
    }

    /**
     * Stop monitoring connection status
     */
    stopConnectionMonitoring() {
        if (this.checkConnectionInterval) {
            clearInterval(this.checkConnectionInterval);
            this.checkConnectionInterval = null;
        }
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus() {
        const statusIndicator = document.getElementById('apiStatus');
        if (!statusIndicator) return;

        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusTexts = statusIndicator.querySelectorAll('span');
        const statusText = statusTexts.length > 1 ? statusTexts[1] : statusTexts[0];

        if (this.isConnected) {
            statusDot.className = 'status-dot online';
            if (statusText) statusText.textContent = 'Connected';
        } else {
            statusDot.className = 'status-dot offline';
            if (statusText) statusText.textContent = 'Disconnected';
        }
    }

    /**
     * Set new base URL
     */
    setBaseUrl(url) {
        this.baseUrl = url.endsWith('/api') ? url : `${url}/api`;
    }

    /**
     * Test connection to API
     */
    async testConnection(url = null) {
        if (url) {
            const originalUrl = this.baseUrl;
            this.setBaseUrl(url);
            
            try {
                const response = await this.checkHealth();
                return { success: true, data: response };
            } catch (error) {
                this.baseUrl = originalUrl; // Restore original URL on failure
                return { success: false, error: error.message };
            }
        } else {
            try {
                const response = await this.checkHealth();
                return { success: true, data: response };
            } catch (error) {
                return { success: false, error: error.message };
            }
        }
    }

    /**
     * Upload file with progress tracking
     */
    async uploadWithProgress(endpoint, formData, onProgress = null) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            if (onProgress) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });
            }

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (error) {
                        resolve(xhr.responseText);
                    }
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Network error occurred'));
            });

            xhr.addEventListener('timeout', () => {
                reject(new Error('Request timed out'));
            });

            xhr.open('POST', `${this.baseUrl}${endpoint}`);
            xhr.timeout = 30000; // 30 second timeout
            xhr.send(formData);
        });
    }

    /**
     * Batch process multiple images
     */
    async batchProcessImages(imageFiles, confidence = 0.5, onProgress = null) {
        const results = [];
        const total = imageFiles.length;

        for (let i = 0; i < total; i++) {
            try {
                const result = await this.detectImage(imageFiles[i], confidence);
                results.push({
                    file: imageFiles[i].name,
                    result: result,
                    success: true
                });

                if (onProgress) {
                    onProgress((i + 1) / total * 100, i + 1, total);
                }
            } catch (error) {
                results.push({
                    file: imageFiles[i].name,
                    error: error.message,
                    success: false
                });

                if (onProgress) {
                    onProgress((i + 1) / total * 100, i + 1, total);
                }
            }

            // Small delay between requests to prevent overwhelming the server
            if (i < total - 1) {
                await Utils.sleep(100);
            }
        }

        return results;
    }

    /**
     * Get detection history (if supported by backend)
     */
    async getDetectionHistory(limit = 50) {
        try {
            const response = await this.request(`/history?limit=${limit}`, {
                method: 'GET'
            });

            return response;
        } catch (error) {
            console.error('Failed to get detection history:', error);
            throw error;
        }
    }

    /**
     * Clear detection statistics
     */
    async clearStats() {
        try {
            const response = await this.request('/stats/clear', {
                method: 'POST'
            });

            return response;
        } catch (error) {
            console.error('Failed to clear statistics:', error);
            throw error;
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopConnectionMonitoring();
    }
}

// Create global API instance
const api = new API();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API, api };
}