// UI-specific functionality and components

class UIManager {
    constructor() {
        this.modals = new Map();
        this.tooltips = new Map();
        this.notifications = [];
        this.theme = 'light';
        
        this.initializeUI();
    }

    initializeUI() {
        this.setupTooltips();
        this.setupModals();
        this.setupKeyboardShortcuts();
        this.setupResizeHandlers();
        
        // Load theme preference
        const savedTheme = Utils.loadFromStorage('app-theme');
        if (savedTheme) {
            this.setTheme(savedTheme);
        }
    }

    /**
     * Setup tooltips for elements with data-tooltip attribute
     */
    setupTooltips() {
        // Add helpful tooltips to various UI elements
        const tooltipElements = [
            { selector: '#loadImageBtn', text: 'Click to select a photo from your computer' },
            { selector: '#loadVideoBtn', text: 'Click to select a video file for analysis' },
            { selector: '#startLiveBtn', text: 'Start using your camera for real-time person detection' },
            { selector: '#confidenceSlider', text: 'Higher = more accurate but may miss some people. Lower = catches more but less precise' },
            { selector: '#exportStatsBtn', text: 'Download a detailed report of all detections' },
            { selector: '#clearStatsBtn', text: 'Reset all statistics and start fresh' },
            { selector: '.soldier-count', text: 'Number of armed personnel detected' },
            { selector: '.civilian-count', text: 'Number of civilians detected' },
            { selector: '.confidence-avg', text: 'Average accuracy of all detections' }
        ];

        tooltipElements.forEach(item => {
            const elements = document.querySelectorAll(item.selector);
            elements.forEach(element => {
                if (!element.dataset.tooltip) {
                    element.dataset.tooltip = item.text;
                    this.addTooltipToElement(element);
                }
            });
        });

        // Setup existing tooltip elements
        document.querySelectorAll('[data-tooltip]').forEach(element => {
            this.addTooltipToElement(element);
        });
    }

    addTooltipToElement(element) {
        const tooltip = this.createTooltip(element.dataset.tooltip);
        this.tooltips.set(element, tooltip);

        element.addEventListener('mouseenter', (e) => {
            this.showTooltip(e.target);
        });

        element.addEventListener('mouseleave', (e) => {
            this.hideTooltip(e.target);
        });
    }

    /**
     * Create tooltip element
     */
    createTooltip(text) {
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = text;
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 10000;
            opacity: 0;
            transition: opacity 0.2s;
            white-space: nowrap;
        `;
        document.body.appendChild(tooltip);
        return tooltip;
    }

    /**
     * Show tooltip
     */
    showTooltip(element) {
        const tooltip = this.tooltips.get(element);
        if (!tooltip) return;

        const rect = element.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width / 2}px`;
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 5}px`;
        tooltip.style.opacity = '1';
        tooltip.style.transform = 'translateX(-50%)';
    }

    /**
     * Hide tooltip
     */
    hideTooltip(element) {
        const tooltip = this.tooltips.get(element);
        if (!tooltip) return;

        tooltip.style.opacity = '0';
    }

    /**
     * Setup modal dialogs
     */
    setupModals() {
        // Create settings modal
        this.createSettingsModal();
        this.createAboutModal();
        this.createHelpModal();
    }

    /**
     * Create settings modal
     */
    createSettingsModal() {
        const modal = this.createModal('settings-modal', 'Advanced Settings', `
            <div class="modal-content">
                <div class="settings-tab-container">
                    <div class="settings-tabs">
                        <button class="tab-button active" data-tab="detection">Detection</button>
                        <button class="tab-button" data-tab="display">Display</button>
                        <button class="tab-button" data-tab="performance">Performance</button>
                        <button class="tab-button" data-tab="alerts">Alerts</button>
                    </div>
                    <div class="settings-content">
                        <div class="tab-panel active" id="detection-tab">
                            <div class="setting-group">
                                <h4>Detection Parameters</h4>
                                <div class="setting-item">
                                    <label for="batch-confidence">Batch Processing Confidence:</label>
                                    <input type="range" id="batch-confidence" min="0.1" max="1.0" step="0.05" value="0.5">
                                    <span class="range-value">0.5</span>
                                </div>
                                <div class="setting-item">
                                    <label for="detection-timeout">Detection Timeout (seconds):</label>
                                    <input type="number" id="detection-timeout" min="5" max="300" value="30">
                                </div>
                                <div class="setting-item">
                                    <label for="cache-enabled">
                                        <input type="checkbox" id="cache-enabled" checked>
                                        Enable Detection Caching
                                    </label>
                                </div>
                            </div>
                        </div>
                        <div class="tab-panel" id="display-tab">
                            <div class="setting-group">
                                <h4>Display Options</h4>
                                <div class="setting-item">
                                    <label for="bbox-thickness">Bounding Box Thickness:</label>
                                    <input type="range" id="bbox-thickness" min="1" max="5" value="2">
                                    <span class="range-value">2</span>
                                </div>
                                <div class="setting-item">
                                    <label for="label-size">Label Font Size:</label>
                                    <select id="label-size">
                                        <option value="small">Small</option>
                                        <option value="medium" selected>Medium</option>
                                        <option value="large">Large</option>
                                    </select>
                                </div>
                                <div class="setting-item">
                                    <label for="theme-select">Theme:</label>
                                    <select id="theme-select">
                                        <option value="light" selected>Light</option>
                                        <option value="dark">Dark</option>
                                        <option value="auto">Auto</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="tab-panel" id="performance-tab">
                            <div class="setting-group">
                                <h4>Performance Settings</h4>
                                <div class="setting-item">
                                    <label for="max-cache-size">Max Cache Size (MB):</label>
                                    <input type="number" id="max-cache-size" min="10" max="1000" value="100">
                                </div>
                                <div class="setting-item">
                                    <label for="live-fps">Live Stream FPS:</label>
                                    <select id="live-fps">
                                        <option value="5">5 FPS</option>
                                        <option value="10" selected>10 FPS</option>
                                        <option value="15">15 FPS</option>
                                        <option value="30">30 FPS</option>
                                    </select>
                                </div>
                                <div class="setting-item">
                                    <label for="auto-cleanup">
                                        <input type="checkbox" id="auto-cleanup" checked>
                                        Auto-cleanup old results
                                    </label>
                                </div>
                            </div>
                        </div>
                        <div class="tab-panel" id="alerts-tab">
                            <div class="setting-group">
                                <h4>Alert Settings</h4>
                                <div class="setting-item">
                                    <label for="soldier-threshold">Soldier Count Alert Threshold:</label>
                                    <input type="number" id="soldier-threshold" min="1" max="50" value="5">
                                </div>
                                <div class="setting-item">
                                    <label for="sound-alerts">
                                        <input type="checkbox" id="sound-alerts" checked>
                                        Enable Sound Alerts
                                    </label>
                                </div>
                                <div class="setting-item">
                                    <label for="alert-cooldown">Alert Cooldown (seconds):</label>
                                    <input type="number" id="alert-cooldown" min="1" max="300" value="10">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="btn btn-secondary" onclick="uiManager.closeModal('settings-modal')">Cancel</button>
                    <button class="btn btn-primary" onclick="uiManager.saveAdvancedSettings()">Save Settings</button>
                </div>
            </div>
        `);
        
        // Setup tab switching
        modal.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchSettingsTab(e.target.dataset.tab);
            });
        });

        // Setup range value updates
        modal.querySelectorAll('input[type="range"]').forEach(range => {
            const valueSpan = range.nextElementSibling;
            if (valueSpan && valueSpan.classList.contains('range-value')) {
                range.addEventListener('input', () => {
                    valueSpan.textContent = range.value;
                });
            }
        });
    }

    /**
     * Create about modal
     */
    createAboutModal() {
        const modal = this.createModal('about-modal', 'About Aerial Threat Detection', `
            <div class="modal-content">
                <div class="about-content">
                    <div class="app-icon">
                        <i class="fas fa-helicopter"></i>
                    </div>
                    <h3>Aerial Threat Detection System</h3>
                    <p class="version">Version 1.0.0</p>
                    
                    <div class="description">
                        <p>A cutting-edge computer vision system designed for classifying soldiers and civilians from aerial imagery using advanced YOLO deep learning technology.</p>
                        
                        <h4>Key Features:</h4>
                        <ul>
                            <li>Real-time object detection and classification</li>
                            <li>Video processing with detailed analytics</li>
                            <li>Live camera feed integration</li>
                            <li>Comprehensive statistics and reporting</li>
                            <li>Batch processing capabilities</li>
                            <li>Alert system for threat detection</li>
                        </ul>
                        
                        <h4>Technical Specifications:</h4>
                        <ul>
                            <li>YOLOv11 Deep Learning Model</li>
                            <li>Python + Flask Backend</li>
                            <li>Electron Desktop Application</li>
                            <li>OpenCV Video Processing</li>
                            <li>Real-time Performance Optimized</li>
                        </ul>
                        
                        <div class="credits">
                            <p><strong>Developed for:</strong> Computer Science Final Project</p>
                            <p><strong>Purpose:</strong> Defense and Humanitarian Operations</p>
                            <p><strong>Technology Stack:</strong> Python, YOLO, OpenCV, Electron, JavaScript</p>
                        </div>
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="btn btn-primary" onclick="uiManager.closeModal('about-modal')">Close</button>
                </div>
            </div>
        `);
    }

    /**
     * Create help modal
     */
    createHelpModal() {
        const modal = this.createModal('help-modal', '🤔 How to Use the Drone Security Scanner', `
            <div class="modal-content">
                <div class="help-content">
                    <div class="help-sections">
                        <div class="help-section">
                            <h4>🚀 Getting Started (It's Easy!)</h4>
                            <div class="help-steps">
                                <div class="help-step">
                                    <span class="step-number">1</span>
                                    <div>
                                        <strong>Pick what to scan:</strong>
                                        <p>📷 <strong>Scan Photo</strong> - Upload a single image<br>
                                        🎬 <strong>Analyze Video</strong> - Process a video file<br>
                                        📹 <strong>Live Camera</strong> - Use your webcam</p>
                                    </div>
                                </div>
                                <div class="help-step">
                                    <span class="step-number">2</span>
                                    <div>
                                        <strong>Upload your file:</strong>
                                        <p>Just drag and drop or click the "Choose" buttons. Easy!</p>
                                    </div>
                                </div>
                                <div class="help-step">
                                    <span class="step-number">3</span>
                                    <div>
                                        <strong>Wait for results:</strong>
                                        <p>The AI will automatically find people and mark them with colored boxes</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="help-section">
                            <h4>📁 What Files Can I Use?</h4>
                            <ul style="list-style: none; padding: 0;">
                                <li>📷 <strong>Photos:</strong> JPG, PNG, GIF, BMP files</li>
                                <li>🎬 <strong>Videos:</strong> MP4, AVI, MOV files</li>
                                <li>💡 <strong>Tip:</strong> Clear, well-lit images work best!</li>
                            </ul>
                        </div>
                        
                        <div class="help-section">
                            <h4>🎯 Understanding the Results</h4>
                            <ul style="list-style: none; padding: 0;">
                                <li>🔴 <strong>Red boxes:</strong> Armed personnel detected</li>
                                <li>🟢 <strong>Green boxes:</strong> Civilians detected</li>
                                <li>📊 <strong>Percentage:</strong> How confident the AI is (higher = more sure)</li>
                                <li>📈 <strong>Statistics:</strong> Total count of people found</li>
                            </ul>
                        </div>
                        
                        <div class="help-section">
                            <h4>🔧 Having Problems?</h4>
                            <div class="troubleshooting">
                                <p><strong>❌ System Not Ready?</strong><br>
                                Make sure you ran START_HERE.bat first and wait for both windows to load completely.</p>
                                
                                <p><strong>🐌 Taking Too Long?</strong><br>
                                Large videos take time - be patient! Check the progress indicator.</p>
                                
                                <p><strong>👀 Missing People?</strong><br>
                                Try adjusting the detection sensitivity slider in settings.</p>
                                
                                <p><strong>📹 Camera Not Working?</strong><br>
                                Allow camera permissions in your browser when prompted.</p>
                            </div>
                        </div>
                        
                        <div class="help-section">
                            <h4>💡 Pro Tips for Best Results</h4>
                            <ul>
                                <li>Use high-quality, clear images</li>
                                <li>Make sure there's good lighting</li>
                                <li>Keep video files under 100MB for faster processing</li>
                                <li>Position your camera steady for live scanning</li>
                                <li>Download reports using "Save Report" for documentation</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="btn btn-primary" onclick="uiManager.closeModal('help-modal')">Got It! 👍</button>
                </div>
            </div>
        `);
    }

    /**
     * Create modal dialog
     */
    createModal(id, title, content) {
        const modal = document.createElement('div');
        modal.id = id;
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-overlay" onclick="uiManager.closeModal('${id}')"></div>
            <div class="modal-dialog">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button class="modal-close" onclick="uiManager.closeModal('${id}')">&times;</button>
                </div>
                ${content}
            </div>
        `;

        document.body.appendChild(modal);
        this.modals.set(id, modal);

        return modal;
    }

    /**
     * Show modal
     */
    showModal(id) {
        const modal = this.modals.get(id);
        if (modal) {
            modal.style.display = 'flex';
            modal.classList.add('show');
            
            // Focus trap
            const focusableElements = modal.querySelectorAll(
                'button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            
            if (focusableElements.length > 0) {
                focusableElements[0].focus();
            }
        }
    }

    /**
     * Close modal
     */
    closeModal(id) {
        const modal = this.modals.get(id);
        if (modal) {
            modal.classList.remove('show');
            setTimeout(() => {
                modal.style.display = 'none';
            }, 300);
        }
    }

    /**
     * Switch settings tab
     */
    switchSettingsTab(tabName) {
        const modal = this.modals.get('settings-modal');
        if (!modal) return;

        // Update tab buttons
        modal.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        modal.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab panels
        modal.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        modal.querySelector(`#${tabName}-tab`).classList.add('active');
    }

    /**
     * Save advanced settings
     */
    saveAdvancedSettings() {
        const modal = this.modals.get('settings-modal');
        if (!modal) return;

        const settings = {
            batchConfidence: modal.querySelector('#batch-confidence').value,
            detectionTimeout: modal.querySelector('#detection-timeout').value,
            cacheEnabled: modal.querySelector('#cache-enabled').checked,
            bboxThickness: modal.querySelector('#bbox-thickness').value,
            labelSize: modal.querySelector('#label-size').value,
            theme: modal.querySelector('#theme-select').value,
            maxCacheSize: modal.querySelector('#max-cache-size').value,
            liveFps: modal.querySelector('#live-fps').value,
            autoCleanup: modal.querySelector('#auto-cleanup').checked,
            soldierThreshold: modal.querySelector('#soldier-threshold').value,
            soundAlerts: modal.querySelector('#sound-alerts').checked,
            alertCooldown: modal.querySelector('#alert-cooldown').value
        };

        // Save to storage
        Utils.saveToStorage('advanced-settings', settings);
        
        // Apply theme change
        if (settings.theme !== this.theme) {
            this.setTheme(settings.theme);
        }

        // Update live monitor threshold
        if (window.liveMonitor) {
            liveMonitor.alertThreshold = parseInt(settings.soldierThreshold);
        }

        this.closeModal('settings-modal');
        Utils.showToast('Advanced settings saved', 'success');
    }

    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ignore shortcuts when typing in inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            const ctrlOrCmd = e.ctrlKey || e.metaKey;

            switch (true) {
                case ctrlOrCmd && e.key === 'o' && !e.shiftKey:
                    e.preventDefault();
                    document.getElementById('loadImageBtn').click();
                    break;
                    
                case ctrlOrCmd && e.key === 'O' && e.shiftKey:
                    e.preventDefault();
                    document.getElementById('loadVideoBtn').click();
                    break;
                    
                case ctrlOrCmd && e.key === 'l':
                    e.preventDefault();
                    if (window.app) {
                        if (window.app.liveStreamActive) {
                            window.app.stopLiveStream();
                        } else {
                            window.app.startLiveStream();
                        }
                    }
                    break;
                    
                case ctrlOrCmd && e.key === 'e':
                    e.preventDefault();
                    if (window.app) {
                        window.app.exportStatistics();
                    }
                    break;
                    
                case ctrlOrCmd && e.key === ',':
                    e.preventDefault();
                    this.showModal('settings-modal');
                    break;
                    
                case e.key === 'F1':
                    e.preventDefault();
                    this.showModal('help-modal');
                    break;
                    
                case e.key === 'Escape':
                    // Close any open modals
                    this.modals.forEach((modal, id) => {
                        if (modal.classList.contains('show')) {
                            this.closeModal(id);
                        }
                    });
                    break;
            }
        });
    }

    /**
     * Setup resize handlers
     */
    setupResizeHandlers() {
        let resizeTimeout;
        
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleResize();
            }, 250);
        });
    }

    /**
     * Handle window resize
     */
    handleResize() {
        // Update chart sizing
        if (window.app && window.app.chart) {
            window.app.chart.resize();
        }

        // Hide tooltips on resize
        this.tooltips.forEach((tooltip) => {
            tooltip.style.opacity = '0';
        });

        // Adjust modal sizing for small screens
        this.modals.forEach((modal) => {
            if (window.innerWidth < 768) {
                modal.classList.add('mobile-modal');
            } else {
                modal.classList.remove('mobile-modal');
            }
        });
    }

    /**
     * Set application theme
     */
    setTheme(theme) {
        const body = document.body;
        
        body.classList.remove('theme-light', 'theme-dark');
        
        if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            theme = prefersDark ? 'dark' : 'light';
        }
        
        body.classList.add(`theme-${theme}`);
        this.theme = theme;
        
        // Save theme preference
        Utils.saveToStorage('app-theme', theme);
    }

    /**
     * Add notification
     */
    addNotification(message, type = 'info', persistent = false) {
        const notification = {
            id: Utils.generateId(),
            message: message,
            type: type,
            timestamp: Utils.getCurrentTimestamp(),
            persistent: persistent
        };

        this.notifications.push(notification);
        
        // Show toast
        const duration = persistent ? 0 : 5000;
        Utils.showToast(message, type, duration);

        return notification.id;
    }

    /**
     * Remove notification
     */
    removeNotification(id) {
        this.notifications = this.notifications.filter(n => n.id !== id);
    }

    /**
     * Clear all notifications
     */
    clearNotifications() {
        this.notifications = [];
        
        // Clear toast container
        const container = document.getElementById('toastContainer');
        if (container) {
            container.innerHTML = '';
        }
    }

    /**
     * Cleanup UI resources
     */
    cleanup() {
        // Remove tooltips
        this.tooltips.forEach((tooltip) => {
            tooltip.remove();
        });
        this.tooltips.clear();

        // Remove modals
        this.modals.forEach((modal) => {
            modal.remove();
        });
        this.modals.clear();

        // Clear notifications
        this.clearNotifications();
    }
}

// Create global UI manager instance
const uiManager = new UIManager();

// Add modal styles
const modalStyles = `
<style>
.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 10000;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.modal.show {
    opacity: 1;
}

.modal-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(5px);
}

.modal-dialog {
    position: relative;
    background: white;
    border-radius: 10px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    margin: 5vh auto;
    max-width: 90vw;
    max-height: 90vh;
    overflow: hidden;
    animation: modalSlideIn 0.3s ease-out;
}

@keyframes modalSlideIn {
    from {
        transform: translateY(-50px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.modal-header {
    padding: 20px 25px;
    border-bottom: 1px solid #e9ecef;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f8f9fa;
}

.modal-header h3 {
    margin: 0;
    color: #2c3e50;
}

.modal-close {
    background: none;
    border: none;
    font-size: 24px;
    cursor: pointer;
    color: #6c757d;
    padding: 0;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.modal-close:hover {
    color: #e74c3c;
}

.modal-content {
    padding: 25px;
    overflow-y: auto;
    max-height: 70vh;
}

.modal-actions {
    padding: 20px 25px;
    border-top: 1px solid #e9ecef;
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    background: #f8f9fa;
}

/* Settings modal specific styles */
.settings-tab-container {
    display: flex;
    gap: 20px;
}

.settings-tabs {
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 120px;
}

.tab-button {
    padding: 10px 15px;
    border: none;
    background: #f8f9fa;
    border-radius: 5px;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s;
}

.tab-button.active {
    background: var(--secondary-color);
    color: white;
}

.tab-button:hover:not(.active) {
    background: #e9ecef;
}

.settings-content {
    flex: 1;
}

.tab-panel {
    display: none;
}

.tab-panel.active {
    display: block;
}

.setting-group {
    margin-bottom: 25px;
}

.setting-group h4 {
    margin-bottom: 15px;
    color: #2c3e50;
    font-size: 16px;
}

.setting-item {
    margin-bottom: 15px;
}

.setting-item label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
    color: #495057;
}

.setting-item input,
.setting-item select {
    width: 100%;
    max-width: 300px;
    padding: 8px 12px;
    border: 1px solid #ced4da;
    border-radius: 4px;
    font-size: 14px;
}

.range-value {
    margin-left: 10px;
    font-weight: bold;
    color: var(--secondary-color);
}

/* About modal styles */
.about-content {
    text-align: center;
}

.app-icon i {
    font-size: 64px;
    color: var(--secondary-color);
    margin-bottom: 20px;
}

.about-content h3 {
    margin-bottom: 10px;
    color: #2c3e50;
}

.version {
    color: #6c757d;
    margin-bottom: 30px;
    font-style: italic;
}

.description {
    text-align: left;
    max-width: 600px;
    margin: 0 auto;
}

.description ul {
    padding-left: 20px;
    margin-bottom: 20px;
}

.description li {
    margin-bottom: 5px;
}

.credits {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #e9ecef;
    font-size: 14px;
    color: #6c757d;
}

/* Help modal styles */
.help-content {
    max-width: 800px;
}

.help-section {
    margin-bottom: 30px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
}

.help-section h4 {
    color: #2c3e50;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.help-section i {
    color: var(--secondary-color);
}

.help-section ul {
    padding-left: 20px;
}

.help-section li {
    margin-bottom: 8px;
}

kbd {
    background: #f8f9fa;
    border: 1px solid #ced4da;
    border-radius: 3px;
    padding: 2px 6px;
    font-family: monospace;
    font-size: 12px;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .modal-dialog {
        margin: 0;
        max-width: 100vw;
        max-height: 100vh;
        border-radius: 0;
    }
    
    .settings-tab-container {
        flex-direction: column;
    }
    
    .settings-tabs {
        flex-direction: row;
        overflow-x: auto;
        min-width: auto;
    }
    
    .tab-button {
        white-space: nowrap;
        min-width: 100px;
    }
}
</style>
`;

// Inject modal styles
document.head.insertAdjacentHTML('beforeend', modalStyles);

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UIManager, uiManager };
}