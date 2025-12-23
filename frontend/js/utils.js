// Utility functions for the application

class Utils {
    /**
     * Show loading overlay
     */
    static showLoading(text = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const loadingText = overlay.querySelector('.loading-text');
        loadingText.textContent = text;
        overlay.style.display = 'flex';
    }

    /**
     * Hide loading overlay
     */
    static hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = 'none';
    }

    /**
     * Show toast notification with user-friendly messages
     */
    static showToast(message, type = 'info', duration = 5000) {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        
        // Enhanced user-friendly messages
        const friendlyMessages = {
            'API connection failed': '🔧 Connection issue - Make sure the system is running!',
            'File upload failed': '❌ Upload failed - Please try a different file',
            'Invalid file type': '⚠️ This file type isn\'t supported - Try JPG, PNG, MP4, or AVI',
            'File too large': '📁 File is too big - Please use a smaller file',
            'Detection failed': '🔍 Analysis failed - The file might be corrupted or unsupported',
            'No detections found': '👀 No people found in this image/video',
            'Processing started': '🚀 Starting analysis - This may take a moment...',
            'Processing completed': '✅ Analysis complete! Check your results below',
            'Settings saved': '💾 Settings saved successfully!',
            'Connection restored': '🌐 Connection restored - System is ready!',
            'Camera access denied': '📹 Camera access needed - Please allow camera permissions',
            'Live feed started': '📹 Camera is now active and scanning',
            'Live feed stopped': '⏹️ Camera scanning stopped'
        };
        
        const friendlyMessage = friendlyMessages[message] || message;
        
        // Add appropriate icons based on type
        const typeIcons = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        };
        
        const icon = typeIcons[type] || 'ℹ️';
        
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-icon">${icon}</div>
                <div class="toast-message">${friendlyMessage}</div>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()" title="Close">×</button>
            </div>
        `;
        
        container.appendChild(toast);

        // Animate in
        setTimeout(() => toast.classList.add('show'), 100);

        // Auto remove after duration (unless duration is 0 for persistent)
        if (duration > 0) {
            setTimeout(() => {
                toast.classList.add('fade-out');
                setTimeout(() => {
                    if (toast.parentElement) {
                        toast.remove();
                    }
                }, 300);
            }, duration);
        }

        return toast;
    }

    /**
     * Format confidence score as percentage
     */
    static formatConfidence(confidence) {
        return `${(confidence * 100).toFixed(1)}%`;
    }

    /**
     * Format bounding box coordinates
     */
    static formatBbox(bbox) {
        const [x1, y1, x2, y2] = bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        return `${width}×${height} at (${x1}, ${y1})`;
    }

    /**
     * Validate file type
     */
    static validateFileType(file, allowedTypes) {
        const fileType = file.type.toLowerCase();
        return allowedTypes.some(type => fileType.includes(type));
    }

    /**
     * Format file size
     */
    static formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Create detection item HTML
     */
    static createDetectionItem(detection, index) {
        const { class: className, confidence, bbox } = detection;
        const classColor = className.toLowerCase() === 'soldier' ? 'soldier' : 'civilian';
        const icon = className.toLowerCase() === 'soldier' ? '👤' : '🧑';
        const confidencePercent = this.formatConfidence(confidence);
        const bboxText = bbox ? this.formatBbox(bbox) : 'N/A';
        
        return `
            <div class="detection-item ${classColor}" data-index="${index}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--spacing-sm);">
                    <div style="display: flex; align-items: center; gap: var(--spacing-sm);">
                        <span style="font-size: 18px;">${icon}</span>
                        <span class="detection-class" style="font-weight: 600; color: var(--text-primary);">${className}</span>
                    </div>
                    <span class="confidence" style="font-weight: 700; color: var(--text-secondary); background: var(--bg-secondary); padding: var(--spacing-xs) var(--spacing-sm); border-radius: var(--radius-sm); font-size: 12px;">${confidencePercent}</span>
                </div>
                <div class="detection-bbox" style="font-size: 12px; color: var(--text-muted); font-family: var(--font-family-mono);">Location: ${bboxText}</div>
            </div>
        `;
    }

    /**
     * Update statistics display
     */
    static updateStats(stats) {
        const elements = {
            totalDetections: document.getElementById('totalDetections'),
            totalSoldiers: document.getElementById('totalSoldiers'),
            totalCivilians: document.getElementById('totalCivilians'),
            avgConfidence: document.getElementById('avgConfidence'),
            soldierCount: document.querySelector('.soldier-count'),
            civilianCount: document.querySelector('.civilian-count'),
            confidenceAvg: document.querySelector('.confidence-avg')
        };

        const total = (stats.Soldier || 0) + (stats.Civilian || 0);
        const avgConf = stats.avgConfidence || 0;

        // Update main stats
        if (elements.totalDetections) elements.totalDetections.textContent = total;
        if (elements.totalSoldiers) elements.totalSoldiers.textContent = stats.Soldier || 0;
        if (elements.totalCivilians) elements.totalCivilians.textContent = stats.Civilian || 0;
        if (elements.avgConfidence) elements.avgConfidence.textContent = this.formatConfidence(avgConf);

        // Update sidebar stats
        if (elements.soldierCount) elements.soldierCount.textContent = stats.Soldier || 0;
        if (elements.civilianCount) elements.civilianCount.textContent = stats.Civilian || 0;
        if (elements.confidenceAvg) elements.confidenceAvg.textContent = this.formatConfidence(avgConf);
    }

    /**
     * Debounce function to limit function calls
     */
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Throttle function to limit function calls
     */
    static throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }

    /**
     * Deep clone an object
     */
    static deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    }

    /**
     * Check if the current environment is Electron
     */
    static isElectron() {
        return typeof window !== 'undefined' && window.process && window.process.type;
    }

    /**
     * Save data to localStorage with error handling
     */
    static saveToStorage(key, data) {
        try {
            localStorage.setItem(key, JSON.stringify(data));
            return true;
        } catch (error) {
            console.error('Failed to save to localStorage:', error);
            return false;
        }
    }

    /**
     * Load data from localStorage with error handling
     */
    static loadFromStorage(key, defaultValue = null) {
        try {
            const data = localStorage.getItem(key);
            return data ? JSON.parse(data) : defaultValue;
        } catch (error) {
            console.error('Failed to load from localStorage:', error);
            return defaultValue;
        }
    }

    /**
     * Download data as a file
     */
    static downloadAsFile(data, filename, type = 'application/json') {
        const blob = new Blob([data], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Convert image to base64
     */
    static async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    /**
     * Animate element with CSS classes
     */
    static animate(element, animationClass, duration = 300) {
        element.classList.add(animationClass);
        setTimeout(() => {
            element.classList.remove(animationClass);
        }, duration);
    }

    /**
     * Calculate FPS from timestamps
     */
    static calculateFPS(timestamps, windowSize = 10) {
        if (timestamps.length < 2) return 0;
        
        const recentTimestamps = timestamps.slice(-windowSize);
        if (recentTimestamps.length < 2) return 0;
        
        const timeDiff = recentTimestamps[recentTimestamps.length - 1] - recentTimestamps[0];
        const frameCount = recentTimestamps.length - 1;
        
        return frameCount / (timeDiff / 1000); // FPS
    }

    /**
     * Format duration in seconds to MM:SS
     */
    static formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    /**
     * Get current timestamp
     */
    static getCurrentTimestamp() {
        return Date.now();
    }

    /**
     * Check if object is empty
     */
    static isEmpty(obj) {
        return Object.keys(obj).length === 0 && obj.constructor === Object;
    }

    /**
     * Clamp value between min and max
     */
    static clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    /**
     * Generate random ID
     */
    static generateId() {
        return Math.random().toString(36).substr(2, 9);
    }

    /**
     * Wait for specified time
     */
    static async sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Retry async function with exponential backoff
     */
    static async retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {
        for (let i = 0; i < maxRetries; i++) {
            try {
                return await fn();
            } catch (error) {
                if (i === maxRetries - 1) throw error;
                
                const delay = baseDelay * Math.pow(2, i);
                await this.sleep(delay);
                
                console.warn(`Retry ${i + 1}/${maxRetries} after ${delay}ms delay`);
            }
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Utils;
}