"""
Configuration File
OPTIMIZED FOR GESTURE ACCURACY
"""

# === CAMERA SETTINGS ===
CAMERA_INDEX_HAND = 1   # Laptop Webcam (Hand Tracking)
CAMERA_INDEX_WORLD = 0  # Mobile Camera (AR World)
CAMERA_BACKEND = 1400  # cv2.CAP_MSMF for Windows (Better MJPG support)

# Resolutions
FRAME_WIDTH = 1920     # World Camera (High Quality)
FRAME_HEIGHT = 1080
HAND_CAM_WIDTH = 1920   # Hand Camera (Optimized for Speed)
HAND_CAM_HEIGHT = 1080

CAMERA_FOV = 60.0  # Field of View for 3D Projection
MIRROR_FRAME = True

# === HAND TRACKING (MediaPipe) ===
USE_GPU = True
MODEL_COMPLEXITY = 0  # 0=Fast, 1=Balanced

MAX_NUM_HANDS = 2

# ✅ LOWER CONFIDENCE = BETTER DETECTION (more sensitive)
MIN_DETECTION_CONFIDENCE = 0.5  # Lower = detects hands easier
MIN_TRACKING_CONFIDENCE = 0.5   # Lower = tracks better

# === GESTURE DETECTION ===
CLICK_COOLDOWN = 1.0  # Seconds
PINCH_THRESHOLD = 0.05  # Distance between fingers to trigger pinch
HOLD_ZOOM_SPEED = 0.1  # For resizing
FIST_GRAB_THRESHOLD = 0.8 # Tightness of fist

# === VISUAL SETTINGS ===
POINTER_COLOR = (0, 255, 0)
POINTER_SIZE = 10
MENU_COLOR = (30, 30, 30)

# === GRID & SNAPPING SETTINGS ===
GRID_SIZE = 50.0       # Snap objects to 50 unit increments if enabled
MAGNETIC_BORDER_PX = 40  # Snap to screen edges within 40px

# === SLAM / TRACKING SETTINGS ===
SLAM_ENABLED = False   # Disabled as per user request
SLAM_ORB_FEATURES = 1500
SLAM_MATCH_RATIO = 0.8
SLAM_REPROJ_THRESHOLD = 2.0
SLAM_ASSUMED_HEIGHT = 1.5
SLAM_PROC_WIDTH = 640

# Pose Filtering (One-Euro Filter)
POSE_FILTER_MIN_CUTOFF = 1.0
POSE_FILTER_BETA = 0.007
POSE_FILTER_D_CUTOFF = 1.0
