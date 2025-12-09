const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
    // Create the browser window
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1000,
        minHeight: 700,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
        icon: path.join(__dirname, 'assets', 'icon.png'), // Optional: Add an icon
        titleBarStyle: 'default',
        show: false
    });

    // Load the HTML file
    mainWindow.loadFile('index.html');

    // Show window when ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    // Open DevTools in development
    if (process.argv.includes('--dev')) {
        mainWindow.webContents.openDevTools();
    }

    // Handle window closed
    mainWindow.on('closed', () => {
        mainWindow = null;
        // Kill Python process if running
        if (pythonProcess) {
            pythonProcess.kill();
        }
    });
}

function createMenu() {
    const template = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open Image',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => {
                        mainWindow.webContents.send('menu-open-image');
                    }
                },
                {
                    label: 'Open Video',
                    accelerator: 'CmdOrCtrl+Shift+O',
                    click: () => {
                        mainWindow.webContents.send('menu-open-video');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Start Webcam',
                    accelerator: 'CmdOrCtrl+W',
                    click: () => {
                        mainWindow.webContents.send('menu-start-webcam');
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
                    label: 'Start Detection',
                    accelerator: 'CmdOrCtrl+D',
                    click: () => {
                        mainWindow.webContents.send('menu-start-detection');
                    }
                },
                {
                    label: 'Stop Detection',
                    accelerator: 'CmdOrCtrl+S',
                    click: () => {
                        mainWindow.webContents.send('menu-stop-detection');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Settings',
                    click: () => {
                        mainWindow.webContents.send('menu-open-settings');
                    }
                }
            ]
        },
        {
            label: 'View',
            submenu: [
                {
                    label: 'Toggle Fullscreen',
                    accelerator: 'F11',
                    click: () => {
                        mainWindow.setFullScreen(!mainWindow.isFullScreen());
                    }
                },
                {
                    label: 'Reload',
                    accelerator: 'CmdOrCtrl+R',
                    click: () => {
                        mainWindow.reload();
                    }
                },
                {
                    label: 'Developer Tools',
                    accelerator: 'F12',
                    click: () => {
                        mainWindow.webContents.toggleDevTools();
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
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About Aerial Threat Detection System',
                            message: 'Aerial Threat Detection System v1.0.0',
                            detail: 'Soldier and Civilian Classification Using Drone Vision and Deep Learning\n\nDeveloped for CSC Final Project\n\nBuilt with Electron and Python/YOLO'
                        });
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

// App event listeners
app.whenReady().then(() => {
    createWindow();
    createMenu();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// IPC handlers
ipcMain.handle('select-file', async (event, options) => {
    const result = await dialog.showOpenDialog(mainWindow, options);
    return result;
});

ipcMain.handle('select-save-file', async (event, options) => {
    const result = await dialog.showSaveDialog(mainWindow, options);
    return result;
});

ipcMain.handle('start-python-detection', async (event, args) => {
    return new Promise((resolve, reject) => {
        try {
            // Path to your Python script
            const pythonScript = path.join(__dirname, '..', 'src', 'detection_server.py');
            
            // Set environment variables for Python path
            const env = { 
                ...process.env, 
                PYTHONPATH: path.join(__dirname, '..', 'src')
            };
            
            // Start Python process
            pythonProcess = spawn('python', [pythonScript, ...args], {
                cwd: path.join(__dirname, '..'),
                stdio: ['pipe', 'pipe', 'pipe'],
                env: env
            });

            let output = '';
            let errorOutput = '';

            pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
                // Send real-time output to renderer
                mainWindow.webContents.send('python-output', data.toString());
            });

            pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
                mainWindow.webContents.send('python-error', data.toString());
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    resolve({ success: true, output: output });
                } else {
                    reject({ success: false, error: errorOutput, code: code });
                }
                pythonProcess = null;
            });

            pythonProcess.on('error', (error) => {
                reject({ success: false, error: error.message });
                pythonProcess = null;
            });

        } catch (error) {
            reject({ success: false, error: error.message });
        }
    });
});

ipcMain.handle('stop-python-detection', async (event) => {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
        return { success: true };
    }
    return { success: false, message: 'No process running' };
});

ipcMain.handle('get-app-path', async (event) => {
    return {
        appPath: app.getAppPath(),
        userData: app.getPath('userData'),
        documents: app.getPath('documents')
    };
});

// Handle app closing
app.on('before-quit', (event) => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});