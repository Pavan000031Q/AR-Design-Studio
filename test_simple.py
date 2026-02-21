"""
MINIMAL TEST - FIXED landmark access
"""

import cv2
import numpy as np
from core.hand_tracker import HandTracker
from core.gesture_engine import GestureEngine
from config import *

from ui.menu_system import AppleGlassMenu

def main():
    print("=" * 60)
    print("MINIMAL TEST")
    print("=" * 60)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera OK")
    
    hand_tracker = HandTracker()
    gesture_engine = GestureEngine()
    ar_menu = AppleGlassMenu(screen_width=FRAME_WIDTH, screen_height=FRAME_HEIGHT)
    
    print("✅ All systems ready")
    print("Press M to toggle menu, Q to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        if frame_count == 1:
            print(f"✅ Frame shape: {frame.shape}")
        
        if MIRROR_FRAME:
            frame = cv2.flip(frame, 1)
        
        # Hand tracking
        hand_tracker.find_hands(frame)
        hands = hand_tracker.extract_landmarks(frame)
        
        # Debug landmarks structure (first frame only)
        if frame_count == 1 and hands:
            print(f"✅ Hands detected: {len(hands)}")
            if hands[0]:
                print(f"✅ First landmark type: {type(hands[0])}")
                if len(hands[0]) > 0:
                    print(f"✅ Landmark[0] type: {type(hands[0][0])}")
                    print(f"✅ Landmark[0] value: {hands[0][0]}")
        
        # Get pointer position
        pointer_x, pointer_y = -1, -1
        
        if hands and len(hands) > 0:
            landmarks = hands[0]  # First hand
            
            if landmarks and len(landmarks) > 0:
                # Check if landmarks is a list of dicts or something else
                if isinstance(landmarks, list):
                    for lm in landmarks:
                        if isinstance(lm, dict) and 'id' in lm and lm['id'] == INDEX_TIP:
                            pointer_x = lm['x']
                            pointer_y = lm['y']
                            break
                        elif isinstance(lm, dict) and 'id' in lm and lm['id'] == 8:  # INDEX_TIP = 8
                            pointer_x = lm['x']
                            pointer_y = lm['y']
                            break
        
        # Gesture detection
        gesture = {'hand_detected': False, 'state': 'NONE'}
        if hands and len(hands) > 0 and hands[0]:
            try:
                gesture = gesture_engine.detect_gesture(hands[0], frame.shape)
            except Exception as e:
                if frame_count == 1:
                    print(f"⚠️ Gesture detection error: {e}")
        
        # Update menu
        ar_menu.update(pointer_x, pointer_y, gesture)
        
        # === DRAW EVERYTHING ===
        
        # 1. Draw simple menu
        if ar_menu.state.value == 0:  # Minimized
            btn_x, btn_y = ar_menu.get_minimized_button_position()
            btn_size = ar_menu.minimized_button_size
            
            # Simple button
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x+btn_size, btn_y+btn_size),
                         (220, 220, 220), -1)
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x+btn_size, btn_y+btn_size),
                         (100, 100, 255), 3)
            cv2.putText(frame, "MENU", (btn_x+8, btn_y+45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        else:  # Expanded
            menu_x, menu_y = ar_menu.get_screen_position()
            menu_w = ar_menu.menu_width
            menu_h = ar_menu.menu_height
            
            # Background
            cv2.rectangle(frame, (menu_x, menu_y), (menu_x+menu_w, menu_y+menu_h),
                         (240, 240, 240), -1)
            cv2.rectangle(frame, (menu_x, menu_y), (menu_x+menu_w, menu_y+menu_h),
                         (100, 100, 255), 3)
            
            # Title
            cv2.putText(frame, "Design Studio", (menu_x+20, menu_y+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 2)
            
            # Sidebar
            sidebar_w = int(menu_w * 0.25)
            cv2.rectangle(frame, (menu_x, menu_y+70), (menu_x+sidebar_w, menu_y+menu_h),
                         (220, 220, 220), -1)
            cv2.line(frame, (menu_x+sidebar_w, menu_y+70), (menu_x+sidebar_w, menu_y+menu_h),
                    (150, 150, 150), 2)
            
            # Nav items
            nav_items = ["Walls", "Furniture", "Doors", "Windows"]
            item_y = menu_y + 100
            for i, item in enumerate(nav_items):
                is_selected = (i == 0)  # First item selected
                color = (100, 100, 255) if is_selected else (180, 180, 180)
                cv2.rectangle(frame, (menu_x+10, item_y), (menu_x+sidebar_w-10, item_y+50),
                            color, 2)
                cv2.putText(frame, item, (menu_x+20, item_y+32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
                item_y += 65
        
        # 2. Draw pointer
        if pointer_x > 0 and pointer_y > 0:
            cv2.circle(frame, (int(pointer_x), int(pointer_y)), 20, (255, 180, 80), 3, cv2.LINE_AA)
            cv2.circle(frame, (int(pointer_x), int(pointer_y)), 8, (255, 180, 80), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(pointer_x), int(pointer_y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        
        # 3. Draw help
        help_y = frame.shape[0] - 90
        cv2.rectangle(frame, (10, help_y), (320, frame.shape[0]-10),
                     (240, 240, 240), -1)
        cv2.rectangle(frame, (10, help_y), (320, frame.shape[0]-10),
                     (100, 100, 100), 2)
        cv2.putText(frame, "M - Toggle Menu", (20, help_y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        cv2.putText(frame, "Q - Quit", (20, help_y+60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        
        # 4. Draw state info
        state_text = f"Menu: {ar_menu.state.name}"
        cv2.putText(frame, state_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        # Display
        cv2.imshow("AR Design Studio - SIMPLE TEST", frame)
        
        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("👋 Quitting...")
            break
        elif key == ord('m'):
            ar_menu.toggle()
            print(f"📱 Menu: {ar_menu.state.name}")
    
    hand_tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done")

if __name__ == "__main__":
    main()
