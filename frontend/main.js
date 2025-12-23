const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require('electron');
const path = require('path');

class AerialThreatDetectionApp {
    constructor() {
        this.mainWindow = null;
        this.isDev = process.argv.includes('--dev');
        this.apiBaseUrl = 'http://localhost:5000/api';
    }

    async createMainWindow() {
        // Create the main application window
        this.mainWindow = new BrowserWindow({
            width: 1400,
            height: 900,
            minWidth: 1200,
            minHeight: 700,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false,
                enableRemoteModule: true
            },
            icon: path.join(__dirname, 'assets', 'icon.png'),
            title: 'Aerial Threat Detection System',
            show: true, // Show immediately
            titleBarStyle: 'default',
            center: true, // Center the window
            alwaysOnTop: false,
            skipTaskbar: false
        });

        // Load the main HTML file
        await this.mainWindow.loadFile('index.html');

        // Ensure window is visible and focused
        this.mainWindow.show();
        this.mainWindow.focus();
        this.mainWindow.moveTop();
        
        // Show window when ready (backup)
        this.mainWindow.once('ready-to-show', () => {
            this.mainWindow.show();
            this.mainWindow.focus();
            
            if (this.isDev) {
                this.mainWindow.webContents.openDevTools();
            }
        });

        // Handle window closed
        this.mainWindow.on('closed', () => {
            this.mainWindow = null;
        });

        // Set up menu
        this.createMenu();

        return this.mainWindow;
    }

    createMenu() {
        const template = [
            {
                label: 'File',
                submenu: [
                    {
                        label: 'Load Image',
                        accelerator: 'CmdOrCtrl+O',
                        click: async () => {
                            const result = await dialog.showOpenDialog(this.mainWindow, {
                                properties: ['openFile'],
                                filters: [
                                    { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'tiff'] }
                                ]
                            });

                            if (!result.canceled && result.filePaths.length > 0) {
                                this.mainWindow.webContents.send('file-selected', {
                                    type: 'image',
                                    path: result.filePaths[0]
                                });
                            }
                        }
                    },
                    {
                        label: 'Load Video',
                        accelerator: 'CmdOrCtrl+Shift+O',
                        click: async () => {
                            const result = await dialog.showOpenDialog(this.mainWindow, {
                                properties: ['openFile'],
                                filters: [
                                    { name: 'Videos', extensions: ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'] }
                                ]
                            });

                            if (!result.canceled && result.filePaths.length > 0) {
                                this.mainWindow.webContents.send('file-selected', {
                                    type: 'video',
                                    path: result.filePaths[0]
                                });
                            }
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Export Results',
                        accelerator: 'CmdOrCtrl+E',
                        click: () => {
                            this.mainWindow.webContents.send('export-results');
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Exit',
                        accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
                        click: () => {
                            app.quit();
                        }
                    }
                ]
            },
            {
                label: 'Detection',
                submenu: [
                    {
                        label: 'Start Live Feed',
                        accelerator: 'CmdOrCtrl+L',
                        click: () => {
                            this.mainWindow.webContents.send('start-live-feed');
                        }
                    },
                    {
                        label: 'Stop Live Feed',
                        accelerator: 'CmdOrCtrl+Shift+L',
                        click: () => {
                            this.mainWindow.webContents.send('stop-live-feed');
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Settings',
                        accelerator: 'CmdOrCtrl+,',
                        click: () => {
                            this.mainWindow.webContents.send('show-settings');
                        }
                    }
                ]
            },
            {
                label: 'View',
                submenu: [
                    {
                        label: 'Reload',
                        accelerator: 'CmdOrCtrl+R',
                        click: () => {
                            this.mainWindow.reload();
                        }
                    },
                    {
                        label: 'Force Reload',
                        accelerator: 'CmdOrCtrl+Shift+R',
                        click: () => {
                            this.mainWindow.webContents.reloadIgnoringCache();
                        }
                    },
                    {
                        label: 'Toggle Developer Tools',
                        accelerator: process.platform === 'darwin' ? 'Alt+Cmd+I' : 'Ctrl+Shift+I',
                        click: () => {
                            this.mainWindow.webContents.toggleDevTools();
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Actual Size',
                        accelerator: 'CmdOrCtrl+0',
                        click: () => {
                            this.mainWindow.webContents.setZoomLevel(0);
                        }
                    },
                    {
                        label: 'Zoom In',
                        accelerator: 'CmdOrCtrl+Plus',
                        click: () => {
                            const zoomLevel = this.mainWindow.webContents.getZoomLevel();
                            this.mainWindow.webContents.setZoomLevel(zoomLevel + 0.5);
                        }
                    },
                    {
                        label: 'Zoom Out',
                        accelerator: 'CmdOrCtrl+-',
                        click: () => {
                            const zoomLevel = this.mainWindow.webContents.getZoomLevel();
                            this.mainWindow.webContents.setZoomLevel(zoomLevel - 0.5);
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Toggle Fullscreen',
                        accelerator: process.platform === 'darwin' ? 'Ctrl+Cmd+F' : 'F11',
                        click: () => {
                            this.mainWindow.setFullScreen(!this.mainWindow.isFullScreen());
                        }
                    }
                ]
            },
            {
                label: 'Help',
                submenu: [
                    {
                        label: 'About',
                        click: () => {
                            dialog.showMessageBox(this.mainWindow, {
                                type: 'info',
                                title: 'About Aerial Threat Detection',
                                message: 'Aerial Threat Detection System',
                                detail: 'Version 1.0.0\\n\\nA computer vision system for classifying soldiers and civilians from aerial imagery using YOLO deep learning.\\n\\nDeveloped for CSC Final Project',
                                buttons: ['OK']
                            });
                        }
                    },
                    {
                        label: 'Project Documentation',
                        click: () => {
                            // Open documentation
                            shell.openExternal('https://github.com/your-repo/aerial-threat-detection');
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Report Issue',
                        click: () => {
                            shell.openExternal('https://github.com/your-repo/aerial-threat-detection/issues');
                        }
                    }
                ]
            }
        ];

        // macOS specific menu adjustments
        if (process.platform === 'darwin') {
            template.unshift({
                label: app.getName(),
                submenu: [
                    {
                        label: 'About ' + app.getName(),
                        role: 'about'
                    },
                    { type: 'separator' },
                    {
                        label: 'Services',
                        role: 'services'
                    },
                    { type: 'separator' },
                    {
                        label: 'Hide ' + app.getName(),
                        accelerator: 'Command+H',
                        role: 'hide'
                    },
                    {
                        label: 'Hide Others',
                        accelerator: 'Command+Shift+H',
                        role: 'hideothers'
                    },
                    {
                        label: 'Show All',
                        role: 'unhide'
                    },
                    { type: 'separator' },
                    {
                        label: 'Quit',
                        accelerator: 'Command+Q',
                        click: () => {
                            app.quit();
                        }
                    }
                ]
            });
        }

        const menu = Menu.buildFromTemplate(template);
        Menu.setApplicationMenu(menu);
    }

    setupIPCHandlers() {
        // Handle API base URL requests
        ipcMain.handle('get-api-base-url', () => {
            return this.apiBaseUrl;
        });

        // Handle file system operations
        ipcMain.handle('show-save-dialog', async (event, options) => {
            const result = await dialog.showSaveDialog(this.mainWindow, options);
            return result;
        });

        // Handle opening external links
        ipcMain.handle('open-external', async (event, url) => {
            shell.openExternal(url);
        });

        // Handle showing message boxes
        ipcMain.handle('show-message-box', async (event, options) => {
            const result = await dialog.showMessageBox(this.mainWindow, options);
            return result;
        });
    }

    async initialize() {
        // Disable GPU acceleration to prevent GPU process errors
        app.disableHardwareAcceleration();
        
        // Additional command line switches for stability
        app.commandLine.appendSwitch('disable-gpu');
        app.commandLine.appendSwitch('disable-gpu-compositing');
        app.commandLine.appendSwitch('disable-gpu-rasterization');
        app.commandLine.appendSwitch('disable-gpu-sandbox');
        
        // Set up app event handlers
        app.whenReady().then(async () => {
            await this.createMainWindow();
            this.setupIPCHandlers();

            app.on('activate', async () => {
                if (BrowserWindow.getAllWindows().length === 0) {
                    await this.createMainWindow();
                }
            });
        });

        app.on('window-all-closed', () => {
            if (process.platform !== 'darwin') {
                app.quit();
            }
        });

        app.on('before-quit', (event) => {
            // Cleanup operations before quitting
            console.log('Application is closing...');
        });

        // Security: Prevent new window creation
        app.on('web-contents-created', (event, contents) => {
            contents.on('new-window', (event, navigationUrl) => {
                event.preventDefault();
                shell.openExternal(navigationUrl);
            });
        });
    }
}

// Create and initialize the application
const aerialThreatApp = new AerialThreatDetectionApp();
aerialThreatApp.initialize();