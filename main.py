"""
AR Interior Design Studio
FIXED: Icons + Independent Hand Roles (H1=Pointer, H2=Gesture)
"""

import cv2
import numpy as np
from core.hand_tracker import HandTracker
from core.gesture_engine import GestureEngine
from core.camera_tracker import CameraTracker, TrackingState
from utils.helper import FPSCounter, ThreadedCamera
from config import *
import time
import sys

from ui.menu_system import AppleGlassMenu, MenuState
from ui.menu_renderer import draw_menu, draw_placed_objects_and_held, draw_color_palette
from utils.performance_profiler import PerformanceProfiler
from utils.smoothing import KalmanSmoother

def get_pointer_position(landmarks):
    """Get pointer position with fallback"""
    if not landmarks:
        return -1, -1
    
    index_tip = None
    wrist = None
    middle_mcp = None
    
    for lm in landmarks:
        if isinstance(lm, dict) and 'id' in lm:
            if lm['id'] == 8:
                index_tip = lm
            elif lm['id'] == 0:
                wrist = lm
            elif lm['id'] == 9:
                middle_mcp = lm
    
    if index_tip:
        return index_tip['x'], index_tip['y']
    
    if wrist and middle_mcp:
        dx = middle_mcp['x'] - wrist['x']
        dy = middle_mcp['y'] - wrist['y']
        estimated_x = middle_mcp['x'] + dx * 1.2
        estimated_y = middle_mcp['y'] + dy * 1.2
        return estimated_x, estimated_y
    
    if wrist:
        return wrist['x'], wrist['y']
    
    return -1, -1

def draw_apple_style_pointer(frame, all_hands_data, hand_states, show_labels=True):
    """Draw pointer for ALL hands using the processed/scaled coordinates"""
    if frame is None or not hand_states:
        return frame
    
    for hs in hand_states:
        # Use VALID, SMOOTHED, and SCALED coordinates from hand_states
        # This ensures the visual pointer matches the interaction logic exactly
        x, y = hs['x'], hs['y']
        
        if x > 0 and y > 0:
            state = hs.get('state', 'NONE')
            is_fist = hs.get('is_fist', False)
            
            # Color based on gesture state
            if is_fist:
                color = (0, 200, 255)   # Orange for grab/fist
                label = "GRAB"
            elif state in ['START', 'HOLD']:
                color = (0, 255, 255)   # Yellow for pinch
                label = "PINCH"
            elif state == 'RELEASE':
                color = (255, 0, 255)   # Magenta for release
                label = "DROP"
            else:
                color = (255, 180, 80)  # Light blue for idle pointer
                label = ""
            
            # Draw pointer ring
            cv2.circle(frame, (int(x), int(y)), 22, color, 3, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 8, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
            
            # Show gesture label
            if label and show_labels:
                cv2.putText(frame, label, (int(x) + 25, int(y) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    return frame

def build_hand_mask(frame, all_hands_data, frame_w, frame_h, padding=40):
    """Build a binary mask that EXCLUDES hand regions from SLAM feature detection.
    Returns mask where 255 = track, 0 = ignore (hand region)."""
    mask = np.full((frame_h, frame_w), 255, dtype=np.uint8)
    
    for hand_data in all_hands_data:
        if not hand_data or 'landmarks' not in hand_data:
            continue
        landmarks = hand_data['landmarks']
        if not landmarks:
            continue
        
        # Get bounding box of hand landmarks
        xs = [lm['x'] for lm in landmarks]
        ys = [lm['y'] for lm in landmarks]
        
        x_min = max(0, int(min(xs)) - padding)
        y_min = max(0, int(min(ys)) - padding)
        x_max = min(frame_w, int(max(xs)) + padding)
        y_max = min(frame_h, int(max(ys)) + padding)
        
        mask[y_min:y_max, x_min:x_max] = 0
    
    return mask

def draw_help_overlay(frame, show_hint, is_fullscreen):
    """Draw help overlay"""
    if frame is None or not show_hint:
        return frame
    
    h, w = frame.shape[:2]
    box_h = 120
    box_w = 650
    box_x = 20
    box_y = h - box_h - 20
    
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                 (240, 240, 240), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                 (100, 100, 100), 2)
    
    y_offset = box_y + 25
    cv2.putText(frame, "Q - Quit  |  M - Menu  |  F - Fullscreen  |  G - Debug  |  H - Help", 
               (box_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    
    y_offset += 30
    cv2.putText(frame, "One Hand: Point + Pinch  |  Two Hands: H1=Point, H2=Pinch", 
               (box_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    
    y_offset += 30
    cv2.putText(frame, "Drag menu borders to resize | Click menu items to select", 
               (box_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    
    return frame

def main():
    """Main application loop"""
    print("=" * 70)
    print("AR INTERIOR DESIGN STUDIO - Apple Glass Menu")

    print("=" * 70)
    
    # 1. Camera Setup (Dual Camera Support with Multithreading)
    print(f"ðŸŽ¥ Initializing {CAMERA_BACKEND} backend with Threaded Capture...")
    
    # Hand Camera (Webcam) - Optimized for latency
    cap_hand = ThreadedCamera(CAMERA_INDEX_HAND, CAMERA_BACKEND, 
                             width=HAND_CAM_WIDTH, height=HAND_CAM_HEIGHT, 
                             fps=30.0, fourcc='MJPG').start()
    
    if not cap_hand.isOpened():
        print(f"âŒ Hand Camera (Index {CAMERA_INDEX_HAND}) failed to open!")
        return

    # World Camera (Mobile / AR View) - Optimized for high frame rate
    # Use a temporary capture to check if the camera exists
    temp_cap = cv2.VideoCapture(CAMERA_INDEX_WORLD, CAMERA_BACKEND)
    using_dual_camera = temp_cap.isOpened()
    temp_cap.release()
    
    if using_dual_camera:
        print(f"âœ… World Camera (Index {CAMERA_INDEX_WORLD}) found! Starting 60FPS thread...")
        cap_world = ThreadedCamera(CAMERA_INDEX_WORLD, CAMERA_BACKEND,
                                  width=FRAME_WIDTH, height=FRAME_HEIGHT, 
                                  fps=60.0, fourcc='MJPG').start()
    else:
        print(f"⚠️  World Camera not found. Opening Single Camera (Index {CAMERA_INDEX_HAND}) at HIGH RESOLUTION.")
        # Override hand camera settings to capture at WORLD resolution because it's shared
        cap_hand.release()
        cap_hand = ThreadedCamera(CAMERA_INDEX_HAND, CAMERA_BACKEND, 
                                 width=FRAME_WIDTH, height=FRAME_HEIGHT, 
                                 fps=30.0, fourcc='MJPG').start()
        cap_world = cap_hand # Shared

    print(f"âœ… Camera Threads Active")
    
    # Initialize components
    print("ðŸ‘‹ Initializing hand tracking...")
    # hand_tracker = HandTracker()
    from core.hand_tracker import HandTrackerHighPerf
    hand_tracker = HandTrackerHighPerf(internal_res=(HAND_CAM_WIDTH, HAND_CAM_HEIGHT)) # Hand tracker works on hand cam resolution
    print("âœ… Hand tracker ready")
    
    print("ðŸ¤Œ Initializing gesture engines...")
    # Per-hand gesture engines â€” each hand gets its own state machine
    gesture_engines = [GestureEngine(), GestureEngine()]
    # Per-hand smoothers
    hand_smoothers = [KalmanSmoother(), KalmanSmoother()]
    print("âœ… Gesture engines & smoothers ready (2x independent)")
    
    print("ðŸ“± Initializing Apple Glass Menu...")
    ar_menu = AppleGlassMenu(screen_width=FRAME_WIDTH, screen_height=FRAME_HEIGHT)
    print("âœ… Menu ready")

    print("ðŸ‘ï¸ Initializing Camera Tracker (Visual Odometry)...")
    camera_tracker = CameraTracker(
        width=FRAME_WIDTH, height=FRAME_HEIGHT, fov=CAMERA_FOV,
        orb_features=SLAM_ORB_FEATURES, match_ratio=SLAM_MATCH_RATIO,
        reproj_threshold=SLAM_REPROJ_THRESHOLD,
        assumed_height=SLAM_ASSUMED_HEIGHT, proc_width=SLAM_PROC_WIDTH,
        filter_min_cutoff=POSE_FILTER_MIN_CUTOFF, filter_beta=POSE_FILTER_BETA,
        filter_d_cutoff=POSE_FILTER_D_CUTOFF
    )
    print("âœ… Camera Tracker ready")
    
    # Performance profiler
    profiler = PerformanceProfiler()
    show_stats = True  # Toggle with 'P' key
    
    fps_counter = FPSCounter()
    
    # UI state
    show_fps = False     # Default: OFF
    show_debug = False   # Default: OFF
    mirror_frame = MIRROR_FRAME
    show_hint = False    # Default: OFF
    fullscreen_mode = True # Default: ON
    # FPS
    fps_counter = FPSCounter()
    frame_count = 0
    
    # Click Cooldown
    last_click_time = time.time()
    CLICK_COOLDOWN = 0.3
    
    window_name = "AR Design Studio"
    
    print("\n" + "=" * 70)
    print("ðŸš€ STARTING AR STUDIO")
    print("=" * 70)
    print("\nðŸ“‹ CONTROLS:")
    print("  Q or ESC     - Quit")
    print("  M            - Toggle Menu")
    print("  F            - Toggle Fullscreen")
    print("  G            - Toggle Debug")
    print("  H            - Toggle Help")
    print("\nðŸ¤Œ GESTURES:")
    print("  Pinch        - Click / Select / Drag menu")
    print("  Fist         - Grab and move objects")
    print("  Both hands   - Fully independent, symmetric")
    print("=" * 70 + "\n")
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Force fullscreen immediately
    if fullscreen_mode:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        profiler.start_frame()
        
        # â”€â”€â”€ 1. CAPTURE FRAMES â”€â”€â”€
        profiler.start_section("Camera")
        ret_hand, frame_hand = cap_hand.read()
        
        if using_dual_camera:
            ret_world, frame_world = cap_world.read()
        else:
            frame_world = frame_hand.copy() # Shared (Single Cam Mode)
            ret_world = ret_hand

        if not ret_hand or not ret_world:
            print("âŒ Frame capture failed")
            break

        # Resize Inputs if needed to match config (Critical for AR Mapping)
        if frame_world.shape[1] != FRAME_WIDTH or frame_world.shape[0] != FRAME_HEIGHT:
             frame_world = cv2.resize(frame_world, (FRAME_WIDTH, FRAME_HEIGHT))
             
        # Hand frame is already low-res (optimized), no resize needed usually.
        
        # MIRRORING
        if mirror_frame:
            # We mirror BOTH in single camera mode because it's usually front-facing
            frame_hand = cv2.flip(frame_hand, 1)
            if not using_dual_camera:
                 frame_world = cv2.flip(frame_world, 1)
        
        # ✅ FIX: For single camera mode, we MUST downscale frame_hand for performance
        # but keep frame_world for accuracy.
        if not using_dual_camera:
            frame_hand = cv2.resize(frame_world, (HAND_CAM_WIDTH, HAND_CAM_HEIGHT))
        
        # We work primarily on frame_world for rendering
        frame = frame_world.copy() 
        
        
        # ✅ STEP 1: Hand Tracking on HAND FRAME (Optimized)
        profiler.start_section("Hands")
        
        # Hand Tracker expects RGB
        # We process the smaller frame_hand
        hand_tracker.find_hands(frame_hand)
        
        # Coordinate Mapping: Hand landmarks are normalized (0-1).
        # We must scale them to the WORLD FRAME size for drawing.
        # HandTracker stores results internally. We need to override how we look them up?
        # No, HandTracker uses find_hands to update internal state.
        # But `gesture_engine` uses `hand_tracker.pixel_landmarks`.
        # `pixel_landmarks` are calculated based on the image size passed to `find_hands`.
        # So right now, they are scaled to 640x480.
        # We need to RESCALE them to 1920x1080 for the Menu System to work on the big screen!
        
        # Patching HandTracker results to match World Resolution
        # The `extract_landmarks` method returns landmarks in the coordinate space of the frame
        # passed to `find_hands`. So, `all_hands_data` will contain landmarks scaled to `HAND_CAM_WIDTH`x`HAND_CAM_HEIGHT`.
        # We need to scale these to `FRAME_WIDTH`x`FRAME_HEIGHT` for UI interaction.
        
        all_hands_data_hand_cam_res = hand_tracker.extract_landmarks(frame_hand) # Landmarks in hand cam resolution
        
        # Calculate scaling factors from hand cam resolution to world cam resolution
        scale_x = FRAME_WIDTH / HAND_CAM_WIDTH
        scale_y = FRAME_HEIGHT / HAND_CAM_HEIGHT

        all_hands_data = []
        for hand_data_hc in all_hands_data_hand_cam_res:
            if hand_data_hc and 'landmarks' in hand_data_hc:
                # Optimized scaling using list comprehension or direct mapping (minor gain, but cleaner)
                scaled_landmarks = [
                    {
                        'id': lm['id'],
                        'x': lm['x'] * scale_x,
                        'y': lm['y'] * scale_y,
                        'z': lm['z']
                    } for lm in hand_data_hc['landmarks']
                ]
                scaled_hand_data = hand_data_hc.copy()
                scaled_hand_data['landmarks'] = scaled_landmarks
                all_hands_data.append(scaled_hand_data)
        
        profiler.end_section()  # End "Hands" section
        
        # ✅ STEP 2: SLAM Camera Tracking on WORLD FRAME (after hands, so we can mask them)
        profiler.start_section("SLAM")
        if SLAM_ENABLED:
            # Build hand exclusion mask (hands must not be tracked as world features)
            hand_mask = build_hand_mask(frame, all_hands_data, FRAME_WIDTH, FRAME_HEIGHT)
            camera_tracker.track(frame, hand_mask=hand_mask)
            view_matrix = camera_tracker.get_view_matrix()
            camera_pose = camera_tracker.get_camera_pose()
            tracking_state = camera_tracker.get_tracking_state()
        else:
            # STATIC VIEW MATRIX - Camera at origin looking down -Z (OpenGL convention)
            # Only flip Y for CV→GL coordinate conversion, keep Z as-is
            # so objects at negative Z are in front of the camera
            view_matrix = np.array([
                [1.0,  0.0,  0.0, 0.0],
                [0.0, -1.0,  0.0, 0.0],
                [0.0,  0.0,  1.0, 0.0],
                [0.0,  0.0,  0.0, 1.0]
            ], dtype=np.float32)
            camera_pose = (np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
            tracking_state = TrackingState.TRACKING
            camera_pose = (np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
            tracking_state = TrackingState.TRACKING
            
        # [VIRTUAL MODE OVERRIDE]
        if ar_menu.settings.get('virtual_mode', False):
            # Use Orbit Camera Matrix
            view_matrix = ar_menu.get_virtual_view_matrix()
            # Also override camera pose? 
            # CameraPose (R, t) is used for ray-casting.
            # We can extract it from view_matrix inverse, but for now just letting view_matrix drive rendering is key.
            # We must ensure interaction functions use this view_matrix (which they do, passed in update/draw).
            
        profiler.end_section()  # End "SLAM" section
        
        # âœ… VIRTUAL MODE: Radial Transition (Center Fade Out)
        vm_progress = getattr(ar_menu, 'virtual_mode_progress', 0.0)
        
        if vm_progress > 0.001:
            if vm_progress >= 0.99:
                # Fully Virtual - Gradient Background (Sky/Ground)
                # frame = np.full_like(frame, 255) # Old White
                
                # Create Vertical Gradient (Top: Light Blue, Bottom: White/Grey)
                h, w = frame.shape[:2]
                
                # Top Color: Light Blue (250, 230, 200) -> BGR: (200, 230, 250)
                # Bottom Color: White (240, 240, 240) -> BGR: (240, 240, 240)
                color_top = np.array([230, 240, 255]) # Light Sky
                color_bot = np.array([240, 240, 245]) # Horizon
                
                # Linear Interpolation
                # Create a column vector (h, 1, 3)
                v_gradient = np.linspace(color_top, color_bot, h, dtype=np.uint8)
                v_gradient = v_gradient.reshape((h, 1, 3))
                
                # Broadcast to full image (this is fast in numpy)
                frame[:] = v_gradient
                
                # Add a "Horizon" line or Ground plane color?
                # The Grid will handle the ground plane.
                # Just a nice sky gradient is enough.
            else:
                # Radial Wipe from Center
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                max_radius = int(np.hypot(w/2, h/2))
                current_radius = int(max_radius * vm_progress)
                
                # Draw white circle on top of camera frame
                cv2.circle(frame, center, current_radius, (255, 255, 255), -1, cv2.LINE_AA)
                
                # Optional: Soften the edge?
                # For high perf, pure circle is best.

        
        # âœ… STEP 2: Process each hand INDEPENDENTLY (symmetric)
        hand_states = []  # One state per detected hand
        
        for hand_idx, hand_data in enumerate(all_hands_data):
            if not hand_data or 'landmarks' not in hand_data:
                continue
            
            landmarks = hand_data['landmarks']
            if not landmarks or len(landmarks) == 0:
                continue
            
            # Get pointer position (already scaled to world frame resolution)
            raw_x, raw_y = get_pointer_position(landmarks)
            if raw_x < 0 or raw_y < 0:
                continue
            
            # Smooth pointer
            smoother = hand_smoothers[hand_idx % 2]
            pointer_x, pointer_y = smoother.update(raw_x, raw_y)
            
            # Sensitivity is now applied during interaction (rotation), not to the pointer itself.
            # This ensures the pointer stays 1:1 with the hand.
            
            # Clamp to screen
            pointer_x = max(0, min(pointer_x, ar_menu.screen_width))
            pointer_y = max(0, min(pointer_y, ar_menu.screen_height))
            
            # Process gesture with THIS hand's engine (no cross-talk)
            engine_idx = min(hand_idx, 1)  # Cap to 2 engines
            try:
                # Gesture engine needs landmarks in its original coordinate space for distance calculations
                # So we pass the original (unscaled) landmarks to the gesture engine
                original_hand_landmarks = all_hands_data_hand_cam_res[hand_idx]['landmarks']
                gesture_result = gesture_engines[engine_idx].process_gestures(original_hand_landmarks)
                hand_state = {
                    'hand_idx': hand_idx,
                    'x': pointer_x, # Scaled pointer position
                    'y': pointer_y, # Scaled pointer position
                    'state': gesture_result.get('state', 'NONE'),
                    'is_fist': gesture_result.get('is_fist', False),
                    'is_pinching': gesture_result.get('is_pinching', False),
                    'hand_detected': True,
                    'pinch_distance': gesture_result.get('distance', 0),
                    'threshold': gesture_result.get('threshold', 0),
                }
                hand_states.append(hand_state)
            except Exception as e:
                if show_debug:
                    print(f"Gesture error hand {hand_idx}: {e}")
        
        # âœ… STEP 3: Cross-hand gesture logic
        # Rules:
        #   - Hand A pinch â†’ action at Hand B's pointer (two hands)
        #   - Hand A pinch â†’ action at Hand A's position (one hand)
        #   - Fist â†’ grab at own hand's position
        
        action_x, action_y = -1, -1
        gesture_data = {
            'state': 'NONE',
            'is_fist': False,
            'hand_detected': len(hand_states) > 0,
            'hand_count': len(hand_states),
            'hand_distance': 0.0,  # Distance between two hands (pixels)
            'active_hand_id': -1,  # ID of the hand performing the action
        }
        
        fist_hand = None
        pinch_hand = None
        
        for hs in hand_states:
            if hs.get('is_fist', False) and fist_hand is None:
                fist_hand = hs
            if hs.get('state', 'NONE') in ['START', 'HOLD', 'RELEASE'] and pinch_hand is None:
                pinch_hand = hs
        
        # Calculate distance between two hands (for two-hand scaling)
        # Only count if BOTH hands are actively pinching
        if len(hand_states) >= 2:
            h0 = hand_states[0]
            h1 = hand_states[1]
            h0_pinching = h0.get('state', 'NONE') in ['START', 'HOLD']
            h1_pinching = h1.get('state', 'NONE') in ['START', 'HOLD']
            gesture_data['hand_distance'] = np.hypot(h0['x'] - h1['x'], h0['y'] - h1['y'])
            gesture_data['both_pinching'] = h0_pinching and h1_pinching
        
        if fist_hand:
            # Fist always uses OWN hand position (grabbing at your fist)
            action_x, action_y = fist_hand['x'], fist_hand['y']
            # Fist always uses OWN hand position (grabbing at your fist)
            action_x, action_y = fist_hand['x'], fist_hand['y']
            gesture_data['is_fist'] = True
            gesture_data['active_hand_id'] = fist_hand['hand_idx']
            
        elif pinch_hand:
            # Pinch: use the OTHER hand's pointer if two hands are visible
            pinch_idx = pinch_hand['hand_idx']
            other_hand = None
            for hs in hand_states:
                if hs['hand_idx'] != pinch_idx:
                    other_hand = hs
                    break
            

            if other_hand:
                # Two-hand mode: pinch triggers action at the OTHER hand's pointer
                action_x, action_y = other_hand['x'], other_hand['y']
            else:
                # Single hand: action at same hand
                action_x, action_y = pinch_hand['x'], pinch_hand['y']
            
            gesture_data['state'] = pinch_hand['state']
            gesture_data['pinch_distance'] = pinch_hand['pinch_distance']
            gesture_data['active_hand_id'] = pinch_hand['hand_idx']
        else:
            # No gesture â€” just pointing with first hand
            if hand_states:
                action_x, action_y = hand_states[0]['x'], hand_states[0]['y']
                gesture_data['active_hand_id'] = hand_states[0]['hand_idx']
        
        # âœ… FIX: Sticky Hand Logic (Prevent teleporting on release)
        # If we are holding an object, force action_x/y to follow that specific hand
        held_hand_id = getattr(ar_menu, 'holding_hand_id', -1)
        if held_hand_id != -1:
            # Find the hand with this ID
            sticky_hand = next((h for h in hand_states if h['hand_idx'] == held_hand_id), None)
            if sticky_hand:
                action_x, action_y = sticky_hand['x'], sticky_hand['y']
                gesture_data['active_hand_id'] = held_hand_id
                # Even if no gesture is detected (hand open), we keep the context of this hand
                # This ensures the 'RELEASE' event happens at THIS hand's position, not Hand 0's.
        
        
        # Send unified gesture to menu
        # Process UI / Menu Logic
        ar_menu.update(action_x, action_y, gesture_data, camera_pose=camera_pose, view_matrix=view_matrix)
        
        # âœ… STEP 4: Draw Placed Objects (Behind menu)
        profiler.start_section("3D Render")
        # Pass dynamic view matrix to renderer
        # We need to update render function in menu_renderer too? 
        # Actually draw_placed_objects_and_held calls renderer.render
        # Let's check menu_renderer.py... It currently doesn't accept view_matrix.
        # Wait, I should pass it to draw_placed_objects_and_held.
        # But first let's see where draw_placed_objects_and_held is defined.
        
        # It's in ui/menu_renderer.py. I need to update that signature too.
        # For now, let's inject it via side-channel or update argument?
        # Better to update `draw_placed_objects_and_held` to accept view_matrix.
        
        # Process any pending model imports (from file dialog background thread)
        ar_menu.process_pending_import()
        
        frame = draw_placed_objects_and_held(frame, ar_menu, action_x, action_y, view_matrix=view_matrix)
        if frame is None:
            continue
            
        # [DEBUG] Log position of first placed object every 60 frames
        if ar_menu.placed_objects and (frame_count % 60 == 0):
             obj = ar_menu.placed_objects[0]
             print(f"📌 [DEBUG] Object 0 ({obj.name}) Position: {obj.position}")
             
             # Also log View Matrix translation to check for Camera Drift
             if view_matrix is not None:
                 trans = view_matrix[3, :3]
                 print(f"📷 [DEBUG] Camera Trans: [{trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}]")
                 
             # Log Hand Data
             print(f"🖐️ [DEBUG] Hand Action: ({action_x:.1f}, {action_y:.1f}) | Gesture: {gesture_data.get('state', 'NONE')}")
        
        frame_count += 1
        
        # âœ… STEP 5: Draw Menu UI
        profiler.start_section("UI")
        frame = draw_menu(frame, ar_menu, action_x, action_y)
        if frame is None:
            continue
        
        # âœ… STEP 5b: Draw Color Palette (when object selected)
        swatch_positions = draw_color_palette(frame, ar_menu, action_x, action_y)
        ar_menu._swatch_positions = swatch_positions  # Store for hit-testing
        
        profiler.end_section()
        profiler.end_frame()
        
        # Draw performance stats
        if show_stats:
            frame = profiler.draw_stats(frame)
        
        if (ar_menu.settings.get('skeleton', False) or show_debug):
             for hand_data in all_hands_data:
                if hand_data and 'landmarks' in hand_data:
                    landmarks = hand_data['landmarks']
                    if landmarks and len(landmarks) >= 21:
                        # Draw connections
                        connections = [
                            (0,1),(1,2),(2,3),(3,4),     # Thumb
                            (0,5),(5,6),(6,7),(7,8),     # Index
                            (5,9),(9,10),(10,11),(11,12), # Middle
                            (9,13),(13,14),(14,15),(15,16), # Ring
                            (13,17),(17,18),(18,19),(19,20), # Pinky
                            (0,17),                       # Palm base
                        ]
                        lm_dict = {lm['id']: lm for lm in landmarks}
                        for a, b in connections:
                            if a in lm_dict and b in lm_dict:
                                pt1 = (int(lm_dict[a]['x']), int(lm_dict[a]['y']))
                                pt2 = (int(lm_dict[b]['x']), int(lm_dict[b]['y']))
                                cv2.line(frame, pt1, pt2, (0, 255, 200), 2, cv2.LINE_AA)
                        for lm in landmarks:
                            cv2.circle(frame, (int(lm['x']), int(lm['y'])), 4,
                                      (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(frame, (int(lm['x']), int(lm['y'])), 2,
                                      (0, 200, 150), -1, cv2.LINE_AA)
        
        # Draw all hand pointers (with gesture labels if enabled)
        show_gesture_labels = ar_menu.settings.get('gestures', True)
        frame = draw_apple_style_pointer(frame, all_hands_data, hand_states, show_gesture_labels)
        if frame is None:
            continue
        

        
        # === Hitbox / Debug Visualization ===
        if ar_menu.settings.get('hit_boxes', False):
             # Draw Settings Button Box
             set_x, set_y = ar_menu.get_settings_button_position()
             btn_sz = ar_menu.minimized_button_size + 10
             cv2.rectangle(frame, (set_x - 20, set_y - 20), (set_x + btn_sz + 20, set_y + btn_sz + 20), (0, 0, 255), 2)
             
             # Draw Minimized Menu Button Box
             min_x, min_y = ar_menu.get_minimized_button_position()
             cv2.rectangle(frame, (min_x, min_y), (min_x + ar_menu.minimized_button_size, min_y + ar_menu.minimized_button_size), (0, 255, 0), 2)
             
             # Draw 3D Object Hitboxes (Body Detection Visualization)
             f = ar_menu._focal_length()
             cx, cy = ar_menu.screen_width / 2, ar_menu.screen_height / 2
             
             # Placed Objects
             for obj in ar_menu.placed_objects:
                 obj.draw_hitbox_debug(frame, ar_menu.screen_width, ar_menu.screen_height, f, view_matrix=view_matrix, color=(255, 0, 255))
             
             # Held Object
             if ar_menu.held_object:
                 ar_menu.held_object.draw_hitbox_debug(frame, ar_menu.screen_width, ar_menu.screen_height, f, view_matrix=view_matrix, color=(0, 255, 255))
        
        # === Pointer Location ===
        if ar_menu.settings.get('pointers', False):
            for hs in hand_states:
                 px, py = int(hs['x']), int(hs['y'])
                 cv2.putText(frame, f"({px}, {py})", (px + 30, py + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # === Simple FPS ===
        if ar_menu.settings.get('fps', False):
             fps_val = fps_counter.update()
             cv2.putText(frame, f"FPS: {int(fps_val)}", (FRAME_WIDTH - 120, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            

        # === SLAM Tracking State HUD ===
        if SLAM_ENABLED:
            state_colors = {
                TrackingState.NOT_INITIALIZED: (128, 128, 128),
                TrackingState.INITIALIZING: (0, 200, 255),
                TrackingState.TRACKING: (0, 255, 100),
                TrackingState.ROTATION_ONLY: (255, 200, 0),
                TrackingState.LOST: (0, 0, 255),
            }
            slam_color = state_colors.get(tracking_state, (128, 128, 128))
            slam_label = f"TRACK: {tracking_state.name}"
            if tracking_state == TrackingState.ROTATION_ONLY or tracking_state == TrackingState.TRACKING:
                 cv2.putText(frame, "TRACKING ACTIVE", (FRAME_WIDTH // 2 - 130, FRAME_HEIGHT - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2, cv2.LINE_AA)

        if show_debug:
            h, w = frame.shape[:2]
            y_pos = 30
            line_height = 28
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (450, 280), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, f"Menu: {ar_menu.state.name}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y_pos += line_height
            
            cv2.putText(frame, f"Category: {ar_menu.selected_category}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y_pos += line_height
            
            cv2.putText(frame, f"Hands: {len(hand_states)}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y_pos += line_height

        # === INTERACTION LOGIC ===
        if gesture_data['state'] == 'START':
            # 1. Main Menu Button (when minimized)
            if ar_menu.state == MenuState.MINIMIZED:
                btn_x, btn_y = ar_menu.get_minimized_button_position()
                dist = np.hypot(action_x - (btn_x + ar_menu.minimized_button_size/2), 
                               action_y - (btn_y + ar_menu.minimized_button_size/2))
                if dist < 60:
                    ar_menu.toggle()
                    print("ðŸ“– Menu toggled via pinch")
                    # No continue here, let other checks run if needed (though toggling usually consumes input)
                
            # 4. Object Toolbar (Delete & Duplicate)
            if ar_menu.selected_object_index != -1:
                # Button Size
                btn_sz = ar_menu.minimized_button_size + 10
                
                # DELETE
                dx, dy = ar_menu.get_delete_button_position()
                if (dx <= action_x <= dx + btn_sz and dy <= action_y <= dy + btn_sz):
                    if time.time() - last_click_time > CLICK_COOLDOWN:
                        print(f"ðŸ—‘ï¸ Deleted object {ar_menu.selected_object_index}")
                        del ar_menu.placed_objects[ar_menu.selected_object_index]
                        ar_menu.selected_object_index = -1
                        last_click_time = time.time()
                
                # DUPLICATE
                # Check duplicate INDEPENDENTLY of delete
                if ar_menu.selected_object_index != -1: # Double check in case just deleted
                    dup_x, dup_y = ar_menu.get_duplicate_button_position()
                    # print(f"Checking Dup: {action_x},{action_y} vs {dup_x},{dup_y}") # DEBUG
                    if (dup_x <= action_x <= dup_x + btn_sz and dup_y <= action_y <= dup_y + btn_sz):
                         print("ðŸŸ¢ Duplicate Button Clicked!") 
                         if time.time() - last_click_time > CLICK_COOLDOWN:
                             # Clone
                             orig_obj = ar_menu.placed_objects[ar_menu.selected_object_index]
                             new_obj = orig_obj.clone()
                             
                             # Offset slightly so it's not directly inside
                             new_obj.position += np.array([20, 0, 20], dtype=np.float32)
                             
                             ar_menu.placed_objects.append(new_obj)
                             
                             # Select the new object
                             ar_menu.selected_object_index = len(ar_menu.placed_objects) - 1
                             print(f"ðŸ‘¯ Duplicated object! New count: {len(ar_menu.placed_objects)}")
                             
                             last_click_time = time.time()
            
            # 2. Settings Button (Floating)
            set_x, set_y = ar_menu.get_settings_button_position()
            btn_sz = ar_menu.minimized_button_size + 10
            
            # Button Click (Expanded Hitbox)
            if (set_x - 20 <= action_x <= set_x + btn_sz + 20 and 
                set_y - 20 <= action_y <= set_y + btn_sz + 20):
                
                # Debounce
                if time.time() - ar_menu._last_settings_click_time > 0.5:
                    ar_menu._last_settings_click_time = time.time()
                    ar_menu.show_settings = not ar_menu.show_settings
                    print(f"⚙️ Settings Panel: {'OPEN' if ar_menu.show_settings else 'CLOSED'}")
            
            # 5. Import Model Button
            if ar_menu._is_over_import_button(action_x, action_y):
                if time.time() - last_click_time > CLICK_COOLDOWN:
                    last_click_time = time.time()
                    ar_menu.import_model_from_file()
                    print("📂 Import button clicked")
            
            # Settings Panel Clicks/Drags
            if ar_menu.show_settings:
                # Handle interaction (START for toggles, START/HOLD for sliders)
                if gesture_data['state'] in ['START', 'HOLD']:
                    ar_menu._handle_settings_interaction(action_x, action_y, gesture_data['state'])
                    
                # Apply Immediate Effects via Settings Dict
                # Performance Stats
                show_stats = ar_menu.settings.get('performance', False)
                # FPS
                show_fps = ar_menu.settings.get('fps', False)
                # Shutdown
                if ar_menu.settings.get('shutdown', False):
                    print("ðŸ‘‹ Shutdown requested from Settings")
                    # Cleanup
                    cap_hand.release()
                    if using_dual_camera:
                        cap_world.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

        # === GESTURE DETAILS (Settings Dependent) ===
        if (ar_menu.settings.get('gestures', False) or show_debug):
             y_pos_dbg = 150 # Start below other debug info
             for hs in hand_states:
                state = hs.get('state', 'NONE')
                is_fist = hs.get('is_fist', False)
                gesture_label = "FIST" if is_fist else state
                
                if state in ['START', 'HOLD']:
                    gesture_color = (0, 255, 255)
                elif is_fist:
                    gesture_color = (0, 200, 255)
                else:
                    gesture_color = (128, 128, 128)
                
                cv2.putText(frame, f"H{hs['hand_idx']+1}: {gesture_label} ({int(hs['x'])},{int(hs['y'])})", 
                           (15, y_pos_dbg), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 1, cv2.LINE_AA)
                y_pos_dbg += 25
                
                # Also force drawn coordinates if debug is ON
                if show_debug:
                    px, py = int(hs['x']), int(hs['y'])
                    cv2.putText(frame, f"({px}, {py})", (px + 30, py + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        

        
        
        
        # === DEBUG VISUALIZATION ===
        if show_debug:
             # Draw Hitboxes for all placed objects
             if hasattr(ar_menu, 'placed_objects'):
                 for obj in ar_menu.placed_objects:
                     # Use the same View Matrix as the Renderer
                     obj.draw_hitbox_debug(frame, FRAME_WIDTH, FRAME_HEIGHT, 
                                          camera_tracker.K[0,0], # Focal length from K matrix
                                          view_matrix=view_matrix)
             if ar_menu:
                # ar_menu.update expects single hand data. Loop through hands.
                for hs in hand_states:
                    # Inject global gesture data (like hand_distance for zoom)
                    hs['hand_distance'] = gesture_data.get('hand_distance', 0)
                    hs['hand_count'] = gesture_data.get('hand_count', 0)
                    ar_menu.update(hs['x'], hs['y'], hs, view_matrix=view_matrix)
                
                # If no hands, update once to handle animations/timers?
                if not hand_states:
                    # Pass dummy data to keep animations running
                    dummy_gesture = {'hand_detected': False, 'state': 'NONE'}
                    ar_menu.update(0, 0, dummy_gesture, view_matrix=view_matrix)
                     
             # Draw Held Object Hitbox
             if ar_menu.held_object:
                 ar_menu.held_object.draw_hitbox_debug(frame, FRAME_WIDTH, FRAME_HEIGHT, 
                                                      camera_tracker.K[0,0], 
                                                      view_matrix=view_matrix,
                                                      color=(0, 255, 0))

        # === DISPLAY ===
        cv2.imshow(window_name, frame)
        
        # === KEYBOARD ===
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            print("\nðŸ‘‹ Shutting down...")
            break
        elif key == ord('m'):
            ar_menu.toggle()
            print(f"ðŸ“± Menu: {ar_menu.state.name}")
        elif key == ord('f'):
            fullscreen_mode = not fullscreen_mode
            if fullscreen_mode:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("ðŸ–¥ï¸  Fullscreen: ON")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("ðŸ–¥ï¸  Fullscreen: OFF")
        elif key == ord('g'):
            show_debug = not show_debug
            print(f"ðŸ› Debug: {'ON' if show_debug else 'OFF'}")
        elif key == ord('r'):
            mirror_frame = not mirror_frame
            print(f"ðŸªž Mirror Mode: {'ON' if mirror_frame else 'OFF'}")
        elif key == ord('h'):
            show_hint = not show_hint
            print(f"â“ Help Overlay: {'ON' if show_hint else 'OFF'}")
        elif key == ord('c'): # 'c' for camera reset
            camera_tracker.reset()
            print("ðŸ”„ Camera Tracker Reset")
        elif key == ord('p') or key == ord('P'):
            show_fps = not show_fps
            print(f" FPS: {'ON' if show_fps else 'OFF'}")
    
    # === CLEANUP ===
    hand_tracker.close()
    # === CLEANUP ===
    cap_hand.release()
    if using_dual_camera:
        cap_world.release()
    cv2.destroyAllWindows()
    print("\nâœ… Application closed successfully")

if __name__ == "__main__":
    main()