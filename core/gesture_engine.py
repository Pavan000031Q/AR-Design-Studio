
"""
Gesture Detection Engine
Phase 2: Convert landmarks to gestures
"""
import numpy as np
import time

# Constants
PINCH_THRESHOLD = 0.05
PINCH_FRAME_THRESHOLD = 2

class TemporalFilter:
    def __init__(self, threshold_frames=3):
        self.threshold_frames = threshold_frames
        self.history = []
        
    def update(self, state):
        self.history.append(state)
        if len(self.history) > self.threshold_frames:
            self.history.pop(0)
        return all(self.history)

class GestureStateMachine:
    def __init__(self):
        self.state = "NONE"
        
    def update(self, is_pinching):
        if is_pinching:
            if self.state == "NONE":
                self.state = "START"
            elif self.state == "START":
                self.state = "HOLD"
        else:
            if self.state in ["START", "HOLD"]:
                self.state = "RELEASE"
            else:
                self.state = "NONE"
                
    def get_state_name(self):
        return self.state

class GestureEngine:
    """
    Main gesture detection engine
    Processes hand landmarks and outputs gesture events
    """
    
    def __init__(self):
        """Initialize gesture engine"""
        self.pinch_filter = TemporalFilter(threshold_frames=PINCH_FRAME_THRESHOLD)
        self.state_machine = GestureStateMachine()
        
        # Gesture data
        self.pinch_distance = 0.0
        self.pinch_threshold = 0.05
        self.hand_scale = 0.0
        self.raw_pinch = False
        self.filtered_pinch = False
        
        self.is_fist = False
        
        # Key landmarks
        self.thumb_tip = None
        self.index_tip = None
        self.wrist = None
        self.middle_mcp = None
    
    def extract_key_landmarks(self, landmarks):
        if not landmarks: return False
        try:
            # Assuming landmarks is a list of dicts with 'id', 'x', 'y'
            lm_dict = {lm['id']: lm for lm in landmarks}
            self.thumb_tip = lm_dict[4]
            self.index_tip = lm_dict[8]
            self.wrist = lm_dict[0]
            self.middle_mcp = lm_dict[9]
            return True
        except (IndexError, KeyError):
            return False

    def calculate_distance(self, p1, p2):
        return math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])

    def process_gestures(self, landmarks):
        if not self.extract_key_landmarks(landmarks):
            return {'state': 'NONE'}
            
        # 1. Pinch Detection
        dist = np.hypot(self.thumb_tip['x'] - self.index_tip['x'], self.thumb_tip['y'] - self.index_tip['y'])
        
        # Adaptive threshold based on hand size (wrist to middle knuckle)
        hand_size = np.hypot(self.wrist['x'] - self.middle_mcp['x'], self.wrist['y'] - self.middle_mcp['y'])
        
        # Hysteresis Thresholds
        # START pinch only when very close
        pinch_start_thresh = hand_size * 0.20  # Relaxed from 0.08 (was too tight)
        # RELEASE pinch only when fingers move apart significantly
        pinch_release_thresh = hand_size * 0.18 # Relaxed from 0.15

        
        # Debug
        self.pinch_threshold = pinch_start_thresh
        
        if self.state_machine.state in ["START", "HOLD"]:
             is_pinching = dist < pinch_release_thresh
        else:
             is_pinching = dist < pinch_start_thresh
        
        # Temporal Filter (ensure stability)
        filtered_pinch = self.pinch_filter.update(is_pinching)
        
        # State Machine
        self.state_machine.update(filtered_pinch)
        
        # 2. Fist Detection (Simplified)
        # Check if all fingertips are close to palm/wrist
        fingers_curled = 0
        tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
        pips = [6, 10, 14, 18] # PIP joints
        
        lm_dict = {lm['id']: lm for lm in landmarks}
        
        for tip_idx, pip_idx in zip(tips, pips):
            if tip_idx in lm_dict and pip_idx in lm_dict:
                d_tip = np.hypot(lm_dict[tip_idx]['x'] - self.wrist['x'], lm_dict[tip_idx]['y'] - self.wrist['y'])
                d_pip = np.hypot(lm_dict[pip_idx]['x'] - self.wrist['x'], lm_dict[pip_idx]['y'] - self.wrist['y'])
                if d_tip < d_pip: # Tip closer than PIP = curled
                    fingers_curled += 1
        
        self.is_fist = (fingers_curled >= 4)
        
        return {
            'state': self.state_machine.get_state_name(),
            'is_pinching': filtered_pinch,
            'is_fist': self.is_fist,
            'distance': dist,
            'threshold': pinch_start_thresh
        }
