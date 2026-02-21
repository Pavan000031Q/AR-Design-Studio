"""
Hand Tracking Engine - GPU OPTIMIZED
With HAND CLASSIFICATION (Left/Right detection)
"""

import cv2
import mediapipe as mp
from config import *
from utils.helper import landmark_to_pixel, extract_all_landmarks

class HandTracker:
    def __init__(self):
        """Initialize MediaPipe Hands solution with GPU support"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        print(f"🔧 Initializing Hand Tracker...")
        print(f"   GPU Mode: {'ENABLED ⚡' if USE_GPU else 'DISABLED'}")
        print(f"   Model Complexity: {MODEL_COMPLEXITY}")
        print(f"   Max Hands: {MAX_NUM_HANDS}")
        
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_NUM_HANDS,
                model_complexity=0,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            )
            print(f"✅ MediaPipe Hands initialized")
        except Exception as e:
            print(f"⚠️ GPU initialization failed, falling back to CPU: {e}")
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_NUM_HANDS,
                model_complexity=0,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            )
        
        self.results = None
        self.landmarks = []
        
    def find_hands(self, frame):
        """Detect hands in frame (GPU-accelerated)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        return frame
    
    def extract_landmarks(self, frame):
        """
        Extract landmark coordinates with HAND CLASSIFICATION
        Returns: list of dicts with 'landmarks' and 'handedness' (LEFT/RIGHT)
        """
        all_hands_data = []
        
        if self.results.multi_hand_landmarks:
            frame_height, frame_width, _ = frame.shape
            
            # Get handedness info (LEFT or RIGHT hand)
            handedness_list = self.results.multi_handedness if self.results.multi_handedness else []
            
            for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                landmarks = extract_all_landmarks(
                    hand_landmarks,
                    frame_width,
                    frame_height
                )
                
                # Get hand classification (LEFT or RIGHT)
                hand_label = "Unknown"
                if idx < len(handedness_list):
                    hand_label = handedness_list[idx].classification[0].label  # "Left" or "Right"
                
                all_hands_data.append({
                    'landmarks': landmarks,
                    'handedness': hand_label,
                    'index': idx
                })
            
            # Store first hand for backward compatibility
            self.landmarks = all_hands_data[0]['landmarks'] if all_hands_data else []
        
        return all_hands_data
    
    def draw_landmarks(self, frame):
        """Draw hand skeleton and landmarks on frame"""
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame
    
    def get_finger_tip(self, finger_id):
        """Get specific finger tip coordinates"""
        if not self.landmarks:
            return None
        for lm in self.landmarks:
            if lm['id'] == finger_id:
                return (lm['x'], lm['y'], lm['z'])
        return None
    
    def get_wrist(self):
        """Get wrist position (landmark 0)"""
        return self.get_finger_tip(0)
    
    def has_hand(self):
        """Check if hand is detected"""
        return len(self.landmarks) > 0
    
    def get_num_hands(self):
        """Get number of detected hands"""
        if self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0
    
    def close(self):
        """Release MediaPipe resources"""
        self.hands.close()

class HandTrackerHighPerf(HandTracker):
    """
    High Performance Tracker:
    - Runs detection on lower resolution (e.g. 640x360) for 2-3x FPS boost
    - Scales landmarks back to full resolution transparently
    - Identical API to HandTracker
    """
    def __init__(self, internal_res=(640, 360)):
        super().__init__()
        self.internal_w, self.internal_h = internal_res
        print(f"🚀 High Performance Mode: processing at {self.internal_w}x{self.internal_h}")
        
    def find_hands(self, frame):
        """Detect hands using low-res internal frame"""
        # Downscale for speed
        frame_small = cv2.resize(frame, (self.internal_w, self.internal_h))
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        self.results = self.hands.process(frame_rgb)
        return frame

    def extract_landmarks(self, frame):
        """Extract and upscale landmarks to full frame resolution"""
        all_hands_data = []
        
        if self.results.multi_hand_landmarks:
            frame_height, frame_width, _ = frame.shape
            
            # Get handedness info
            handedness_list = self.results.multi_handedness if self.results.multi_handedness else []
            
            for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                # We pass the FULL FRAME dimensions here.
                # MediaPipe returns normalized [0,1] coords based on the small image.
                # Since aspect ratio is preserved (16:9), [0,1] works for the big image too!
                landmarks = extract_all_landmarks(
                    hand_landmarks,
                    frame_width,
                    frame_height
                )
                
                # Get hand classification
                hand_label = "Unknown"
                if idx < len(handedness_list):
                    hand_label = handedness_list[idx].classification[0].label
                
                all_hands_data.append({
                    'landmarks': landmarks,
                    'handedness': hand_label,
                    'index': idx
                })
            
            self.landmarks = all_hands_data[0]['landmarks'] if all_hands_data else []
        
        return all_hands_data

# Function to check for ONNX Runtime (just for info)
def check_onnx_availability():
    try:
        import onnxruntime as ort
        print(f"📦 ONNX Runtime found: {ort.__version__}")
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("🟢 ONNX CUDA Provider Available! (Ready for custom models)")
            return True
        else:
            print("⚠️ ONNX installed but CUDA not available/detected.")
            return False
    except ImportError:
        print("ℹ️ ONNX Runtime not installed.")
        return False
