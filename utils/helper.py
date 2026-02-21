"""
Utility functions for hand tracking
"""
import time
import cv2


class FPSCounter:
    """Calculate and display FPS"""
    
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0
    
    def update(self):
        """Update FPS calculation"""
        curr_time = time.time()
        diff = curr_time - self.prev_time
        if diff > 0:
            self.fps = 1 / diff
        self.prev_time = curr_time
        return self.fps
    
    def get_fps(self):
        """Get current FPS"""
        return int(self.fps)


def landmark_to_pixel(landmark, frame_width, frame_height):
    """
    Convert normalized landmark coordinates (0-1) to pixel coordinates
    
    Args:
        landmark: MediaPipe landmark object
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    
    Returns:
        tuple: (x, y, z) pixel coordinates
    """
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    z = landmark.z  # Depth (relative to wrist)
    
    return x, y, z


def extract_all_landmarks(hand_landmarks, frame_width, frame_height):
    """
    Extract all 21 landmarks as pixel coordinates
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        frame_width: Frame width
        frame_height: Frame height
    
    Returns:
        list: List of tuples [(id, x, y, z), ...]
    """
    landmarks = []
    
    for idx, landmark in enumerate(hand_landmarks.landmark):
        x, y, z = landmark_to_pixel(landmark, frame_width, frame_height)
        landmarks.append({
            'id': idx,
            'x': x,
            'y': y,
            'z': z
        })
    
    return landmarks


def draw_fps(frame, fps):
    """Draw FPS counter on frame"""
    cv2.putText(
        frame,
        f'FPS: {int(fps)}',
        (frame.shape[1] - 150, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    return frame 

def draw_hand_info(frame, num_hands):
    """Draw hand detection status"""
    status = "Hand Detected" if num_hands > 0 else "No Hand"
    color = (0, 255, 0) if num_hands > 0 else (0, 0, 255)
    
    cv2.putText(
        frame,
        status,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def draw_landmark_ids(frame, landmarks, show_ids=False):
    """
    Optionally draw landmark IDs on frame
    Useful for debugging
    """
    if not show_ids:
        return
    
    # Key landmarks to show
    key_points = [0, 4, 8, 12, 16, 20]  # wrist, thumb, index, middle, ring, pinky tips
    
    for lm in landmarks:
        if lm['id'] in key_points:
            cv2.putText(
                frame,
                str(lm['id']),
                (lm['x'] + 10, lm['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1
            )

def check_cuda_availability():
    """Check if OpenCV is CUDA enabled"""
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            print(f"🚀 CUDA is AVAILABLE! Found {count} GPU(s).")
            return True
        else:
            print("⚠️ OpenCV is NOT using CUDA/GPU. Running on CPU.")
            return False
    except AttributeError:
        print("⚠️ This OpenCV build does not support CUDA functions.")
        return False

import threading

class ThreadedCamera:
    """
    Extremely low-latency camera reader.
    Runs frame capture in a background thread to prevent blocking the main loop.
    Enables true 60FPS potential for cameras that support it.
    """
    def __init__(self, src, backend=None, width=None, height=None, fps=None, fourcc=None):
        if backend:
            self.cap = cv2.VideoCapture(src, backend)
        else:
            self.cap = cv2.VideoCapture(src)
            
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                # Slight sleep if capture fails or is waiting
                time.sleep(0.001)

    def read(self):
        with self.read_lock:
            # Note: We return a copy to prevent the main thread from modifying 
            # the same buffer the background thread is writing to.
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def release(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=0.5)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def set(self, prop, val):
        return self.cap.set(prop, val)
