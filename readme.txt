# AR Interior Design Studio - README

## 📋 Project Overview

**AR Interior Design Studio** is an advanced gesture-controlled interior design application that allows users to interact with virtual furniture and design elements using hand gestures, inspired by Apple Vision Pro's spatial computing interface.

***

## 🎯 Problem Statement

Traditional interior design tools require keyboard/mouse input, making the design process less intuitive and immersive. This project aims to create a **touchless, gesture-based interface** where users can:

- Browse and select furniture/design items using hand gestures
- Interact with a floating menu system similar to Apple Glass/Vision Pro
- Place, move, and manipulate virtual objects in real-time
- Experience a natural, intuitive design workflow without physical controllers

***

## 🚀 Current Implementation Status

### ✅ **Completed Features**

#### **1. Hand Tracking System (Phase 1)**
- ✅ Real-time hand detection using MediaPipe
- ✅ Multi-hand support (up to 2 hands simultaneously)
- ✅ Landmark extraction (21 points per hand)
- ✅ Left/Right hand classification
- ✅ Optimized for stable tracking with confidence thresholds
- ✅ Temporal tracking mode (non-static) for smooth motion
- ✅ Visibility-based landmark validation

#### **2. Gesture Detection Engine (Phase 2)**
- ✅ Pinch gesture detection (thumb + index finger)
- ✅ Adaptive pinch threshold based on hand size
- ✅ Temporal filtering to reduce jitter and false positives
- ✅ State machine with gesture states: NONE → START → HOLD → RELEASE
- ✅ Click event detection
- ✅ Drag/hold detection
- ✅ Distance-based gesture recognition
- ✅ Exponential smoothing for stable gesture tracking
- ✅ Hysteresis to prevent flickering

#### **3. Apple Glass Menu System**
- ✅ Minimized floating button (top-right corner)
- ✅ Expandable full menu with Apple Vision Pro styling
- ✅ White translucent glass effect with blur
- ✅ Split layout: Sidebar (25%) + Content Area (75%)
- ✅ Category-based navigation (Walls, Furniture, Doors, Windows, Lighting, Decor, Flooring, Colors)
- ✅ Scrollable content with card-based item display
- ✅ Menu dragging (click header to move)
- ✅ Corner/edge resizing with 8 resize handles
- ✅ Hover effects and visual feedback
- ✅ macOS-style window controls
- ✅ Item database with 50+ design elements
- ✅ Gesture-based menu interaction (pinch to click/drag)

#### **4. Pointer System**
- ✅ Visual pointer indicator at index finger tip
- ✅ Multi-hand pointer support with role indicators
- ✅ Color-coded roles: Orange (Pointer), Yellow (Gesture)
- ✅ Independent hand roles: H1 (Pointer), H2 (Gesture)
- ✅ Smart fallback positioning when tracking is lost
- ✅ Smooth pointer rendering with anti-aliasing

#### **5. UI/UX Features**
- ✅ FPS counter
- ✅ Debug mode with detailed hand tracking info
- ✅ Help overlay with keyboard shortcuts
- ✅ Fullscreen toggle
- ✅ Menu hints for user guidance
- ✅ Keyboard controls (Q: Quit, M: Menu, F: Fullscreen, G: Debug, H: Help)

#### **6. Performance Optimizations**
- ✅ GPU acceleration for MediaPipe
- ✅ Frame mirroring for natural interaction
- ✅ Optimized rendering pipeline (detect → process → draw)
- ✅ Clean frame processing (hand detection before overlay drawing)
- ✅ Model complexity optimization (fastest mode)
- ✅ Camera buffer optimization

***

## ⏳ **In Progress / Pending Features**

### **Phase 3: Object Placement & Manipulation**
- ⏳ Virtual object spawning from menu items
- ⏳ 3D object rendering (furniture, walls, decor)
- ⏳ Object positioning in 3D space
- ⏳ Object rotation gesture (two-finger twist)
- ⏳ Object scaling gesture (pinch-to-zoom)
- ⏳ Object deletion (gesture or menu option)
- ⏳ Object selection/deselection
- ⏳ Multi-object management

### **Phase 4: AR Integration**
- ⏳ Floor/wall plane detection
- ⏳ Spatial anchoring
- ⏳ Realistic object shadows
- ⏳ Collision detection
- ⏳ Snap-to-grid functionality
- ⏳ Depth estimation for realistic placement

### **Phase 5: Advanced Features**
- ⏳ Room layout creation
- ⏳ Measurement tools (distance, area)
- ⏳ Color picker with live preview
- ⏳ Material/texture swapping
- ⏳ Lighting simulation
- ⏳ Multiple room support
- ⏳ Project save/load functionality
- ⏳ Export to 3D formats (OBJ, FBX)
- ⏳ Screenshot/video capture
- ⏳ Undo/redo system

### **Phase 6: AI Integration**
- ⏳ AI-powered design suggestions
- ⏳ Style matching (modern, traditional, minimalist)
- ⏳ Color palette recommendations
- ⏳ Space optimization suggestions
- ⏳ Budget estimation

### **Phase 7: User Experience**
- ⏳ Tutorial/onboarding system
- ⏳ Voice commands integration
- ⏳ Multi-language support
- ⏳ Accessibility features
- ⏳ Custom gesture configuration
- ⏳ Performance settings menu

### **Phase 8: Data & Storage**
- ⏳ Cloud storage integration (Firebase/Supabase)
- ⏳ User authentication
- ⏳ Design history/version control
- ⏳ Sharing designs with others
- ⏳ Community design gallery

***

## 🐛 **Known Issues & Bug Fixes**

### **Recently Fixed:**
1. ✅ Menu button positioning (moved to top-right corner)
2. ✅ Gesture accuracy when pointer is over menu (fixed rendering order)
3. ✅ Gesture breaking during hand movement (added smoothing + lowered confidence thresholds)
4. ✅ Temporal filter optimization (instant mode for speed)
5. ✅ State machine responsiveness (removed hold frame delays)

### **Current Bugs:**
1. ⚠️ Occasional gesture flickering in low light
2. ⚠️ Menu resize can glitch at minimum size
3. ⚠️ Multi-hand pointer assignment can swap roles unexpectedly
4. ⚠️ No visual feedback when menu item is clicked

***

## 🏗️ **Technical Architecture**

### **Project Structure:**
```
AR-Interior-Design-Studio/
├── main.py                 # Main application loop
├── config.py              # Configuration settings
├── hand_tracker.py        # MediaPipe hand tracking (Phase 1)
├── gesture_engine.py      # Gesture detection logic (Phase 2)
├── distance.py            # Distance calculation utilities
├── filters.py             # Temporal filtering for stability
├── state_machine.py       # Gesture state management
├── utils.py               # Helper functions (FPS counter, etc.)
├── Menu_System/
│   ├── menu_system.py     # Menu logic and interaction
│   └── menu_renderer.py   # Menu rendering (Apple Glass style)
└── README.md
```

### **Technology Stack:**
- **Python 3.8+**
- **OpenCV** - Camera capture and image processing
- **MediaPipe** - Hand tracking and landmark detection
- **NumPy** - Mathematical operations
- **n8n** (Future) - Automation and workflow orchestration

### **Key Algorithms:**
- **Adaptive Pinch Threshold**: Dynamic threshold based on hand size
- **Temporal Filtering**: Multi-frame confirmation to reduce noise
- **Exponential Smoothing**: Weighted average for stable values
- **Hysteresis**: Different thresholds for activation/deactivation
- **State Machine**: Finite state automaton for gesture lifecycle

***

## 🎮 **Usage Instructions**

### **Installation:**
```bash
# Install dependencies
pip install opencv-python mediapipe numpy

# Run application
python main.py
```

### **Controls:**
| Key | Action |
|-----|--------|
| **Q / ESC** | Quit application |
| **M** | Toggle menu (minimize/expand) |
| **F** | Toggle fullscreen mode |
| **G** | Toggle debug overlay |
| **H** | Toggle help overlay |

### **Gesture Controls:**
| Gesture | Action |
|---------|--------|
| **Point** | Move pointer (index finger) |
| **Pinch** | Click/Select (thumb + index) |
| **Pinch + Hold** | Drag menu or objects |
| **Release** | Drop object or end drag |

### **Hand Modes:**
- **1 Hand**: Same hand for pointing and gestures
- **2 Hands**: H1 (orange) = pointer, H2 (yellow) = gestures

***

## 🔮 **Future Vision**

The end goal is a **fully immersive AR interior design platform** where users can:
- Walk through virtual rooms
- Design spaces using only hand gestures
- Get AI-powered design recommendations
- Collaborate with others in real-time
- Export designs to VR/AR headsets
- Generate photorealistic renders
- Estimate costs and purchase furniture directly

***

## 📊 **Development Roadmap**

### **Q1 2026** (Current)
- ✅ Core gesture system
- ✅ Menu interface
- ⏳ Object placement basics

### **Q2 2026**
- ⏳ 3D object manipulation
- ⏳ AR plane detection
- ⏳ Room layout tools

### **Q3 2026**
- ⏳ AI design assistant
- ⏳ Cloud integration
- ⏳ User authentication

### **Q4 2026**
- ⏳ Mobile app version
- ⏳ VR headset support
- ⏳ Marketplace integration

***

## 🤝 **Contributing**

This is currently a solo project by an aspiring startup founder. Future contributions welcome after MVP launch!

***

## 📝 **License**

Proprietary - All rights reserved (for now, will consider open-source later)

***

## 📧 **Contact**

- **Developer**: Computer Science Student, Hyderabad
- **Status**: Pre-seed startup phase
- **Looking for**: Feedback, beta testers, potential investors

***

## 🎓 **Learning Outcomes**

This project has helped develop skills in:
- Computer Vision & Image Processing
- Machine Learning (MediaPipe models)
- UI/UX Design (Apple-inspired interfaces)
- Real-time system optimization
- State management and event-driven programming
- Gesture recognition algorithms
- Software architecture design

***

**Last Updated**: February 14, 2026  
**Version**: 0.3.0-alpha  
**Status**: Active Development 🚧