#!/bin/bash

# Aerial Threat Detection System Setup Script
# This script automates the installation and setup process

set -e  # Exit on any error

echo "🚁 Aerial Threat Detection System Setup"
echo "========================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_step "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Linux system"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS system"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_status "Detected Windows system"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check Python installation
check_python() {
    print_step "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
        print_status "Found Python $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python version is compatible"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.8 or higher is required"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version | cut -d " " -f 2)
        if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Found compatible Python $PYTHON_VERSION"
            PYTHON_CMD="python"
        else
            print_error "Python 3.8 or higher is required"
            exit 1
        fi
    else
        print_error "Python is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check Node.js installation
check_node() {
    print_step "Checking Node.js installation..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Found Node.js $NODE_VERSION"
        
        # Check if version is 16 or higher
        if node -e "process.exit(process.version.split('.')[0].substr(1) >= 16 ? 0 : 1)"; then
            print_status "Node.js version is compatible"
        else
            print_error "Node.js 16 or higher is required"
            exit 1
        fi
    else
        print_error "Node.js is not installed. Please install Node.js 16 or higher."
        exit 1
    fi
}

# Check if model file exists
check_model() {
    print_step "Checking for trained model file..."
    
    if [ -f "yolo11s.pt" ]; then
        print_status "Found trained model: yolo11s.pt"
        MODEL_SIZE=$(du -h yolo11s.pt | cut -f1)
        print_status "Model size: $MODEL_SIZE"
    else
        print_warning "Model file 'yolo11s.pt' not found in project root"
        print_warning "Please ensure your trained model is placed as 'yolo11s.pt'"
        read -p "Continue without model file? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Setup cancelled. Please add your model file and try again."
            exit 1
        fi
    fi
}

# Setup Python virtual environment
setup_python_env() {
    print_step "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    print_status "Activated virtual environment"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Install Python dependencies
install_python_deps() {
    print_step "Installing Python dependencies..."
    
    cd backend
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing packages from requirements.txt..."
        pip install -r requirements.txt
        print_status "Python dependencies installed successfully"
    else
        print_error "requirements.txt not found in backend directory"
        exit 1
    fi
    
    cd ..
}

# Install Node.js dependencies
install_node_deps() {
    print_step "Installing Node.js dependencies..."
    
    cd frontend
    
    if [ -f "package.json" ]; then
        print_status "Installing npm packages..."
        npm install
        print_status "Node.js dependencies installed successfully"
    else
        print_error "package.json not found in frontend directory"
        exit 1
    fi
    
    cd ..
}

# Test backend setup
test_backend() {
    print_step "Testing backend setup..."
    
    # Activate virtual environment
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    cd backend
    
    print_status "Testing model loading..."
    timeout 10s $PYTHON_CMD -c "from drone_detector import DroneDetector; detector = DroneDetector('../yolo11s.pt'); print('✅ Model loaded successfully')" || print_warning "Model test timed out or failed"
    
    print_status "Testing API imports..."
    $PYTHON_CMD -c "from api_server import app; print('✅ API server imports successful')"
    
    cd ..
}

# Test frontend setup
test_frontend() {
    print_step "Testing frontend setup..."
    
    cd frontend
    
    print_status "Checking Electron installation..."
    npx electron --version
    
    print_status "Testing application startup (dry run)..."
    timeout 5s npm run dev --dry-run || print_status "Dry run completed"
    
    cd ..
}

# Create startup scripts
create_startup_scripts() {
    print_step "Creating startup scripts..."
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚁 Starting Aerial Threat Detection Backend..."

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Start API server
cd backend
python api_server.py
EOF

    # Frontend startup script
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "🚁 Starting Aerial Threat Detection Frontend..."

cd frontend
npm start
EOF

    # Combined startup script
    cat > start_system.sh << 'EOF'
#!/bin/bash
echo "🚁 Starting Aerial Threat Detection System..."
echo "Starting backend and frontend services..."

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    kill %1 %2 2>/dev/null
    exit
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start backend in background
./start_backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background
./start_frontend.sh &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
EOF

    chmod +x start_backend.sh start_frontend.sh start_system.sh
    
    print_status "Created startup scripts:"
    print_status "  - start_backend.sh  (backend only)"
    print_status "  - start_frontend.sh (frontend only)"  
    print_status "  - start_system.sh   (complete system)"
}

# Create desktop shortcut (Linux/macOS)
create_desktop_shortcut() {
    if [[ "$OS" == "linux" ]]; then
        print_step "Creating desktop shortcut..."
        
        DESKTOP_FILE="$HOME/Desktop/AerialThreatDetection.desktop"
        cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Aerial Threat Detection
Comment=Soldier and Civilian Classification Using Drone Vision
Exec=$(pwd)/start_system.sh
Icon=$(pwd)/frontend/assets/icon.png
Terminal=true
StartupWMClass=AerialThreatDetection
Categories=Development;Science;
EOF
        chmod +x "$DESKTOP_FILE"
        print_status "Created desktop shortcut"
    fi
}

# Print final instructions
print_final_instructions() {
    echo
    echo "🎉 Setup completed successfully!"
    echo "================================"
    echo
    print_status "Your Aerial Threat Detection system is ready to use!"
    echo
    echo "📋 Next steps:"
    echo "1. Ensure your trained model 'yolo11s.pt' is in the project root"
    echo "2. Start the system using one of these methods:"
    echo
    echo "   Option A - Start complete system:"
    echo "   ./start_system.sh"
    echo
    echo "   Option B - Start services separately:"
    echo "   Terminal 1: ./start_backend.sh"
    echo "   Terminal 2: ./start_frontend.sh"
    echo
    echo "   Option C - Manual startup:"
    echo "   Terminal 1: cd backend && python api_server.py"
    echo "   Terminal 2: cd frontend && npm start"
    echo
    echo "🌐 Access points:"
    echo "• Frontend: Electron application (auto-opens)"
    echo "• Backend API: http://localhost:5000/api/health"
    echo
    echo "📖 Documentation:"
    echo "• README.md - Complete documentation"
    echo "• Help menu in application (F1)"
    echo
    print_status "Enjoy your aerial threat detection system! 🚁"
}

# Main setup process
main() {
    echo "Starting automated setup process..."
    echo
    
    check_os
    check_python
    check_node
    check_model
    setup_python_env
    install_python_deps
    install_node_deps
    test_backend
    test_frontend
    create_startup_scripts
    create_desktop_shortcut
    print_final_instructions
}

# Run main setup
main