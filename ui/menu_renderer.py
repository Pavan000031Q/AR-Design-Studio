"""
Apple Glass Menu Renderer - OPTIMIZED
High Performance version using Background Caching and OpenCV Text
"""

import cv2
import numpy as np
import math
import os
import time

# Custom Icon Manager
class IconManager:
    def __init__(self, base_path="assets/icons"):
        self.base_path = base_path
        self.cache = {}
        # Ensure directory exists
        if not os.path.exists(base_path):
            try:
                os.makedirs(base_path)
                print(f"ðŸ“ Created icon directory: {base_path}")
                print(f"â„¹ï¸  Place 64x64 .png icons here to customize!")
            except OSError:
                pass

    def get_icon(self, name, size=None):
        """Load and cache icon. Returns None if not found."""
        key = (name, size)
        if key in self.cache:
            return self.cache[key]
        
        # Try extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            path = os.path.join(self.base_path, name + ext)
            if os.path.exists(path):
                try:
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is None: continue
                    
                    if size:
                        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                    
                    self.cache[key] = img
                    return img
                except Exception as e:
                    print(f"âš ï¸ Error loading icon {path}: {e}")
        
        # Cache failure to avoid repeated filesystem checks
        self.cache[key] = None
        return None

# Global instance
icon_manager = IconManager()

# Global Cache
_MENU_BG_CACHE = None
_LAST_MENU_STATE = None
_LAST_MENU_POS = None
_LAST_SCREEN_SIZE = None

def get_cached_background(menu):
    """
    Check if we can use cached background.
    Invalidate if:
    - Menu state changed
    - Menu position changed
    - Screen size resized
    """
    global _MENU_BG_CACHE, _LAST_MENU_STATE, _LAST_MENU_POS, _LAST_SCREEN_SIZE
    
    current_state = menu.state.value
    current_pos = menu.get_screen_position()
    current_size = (menu.menu_width, menu.menu_height)
    
    if (_MENU_BG_CACHE is not None and 
        _LAST_MENU_STATE == current_state and 
        _LAST_MENU_POS == current_pos and
        _LAST_SCREEN_SIZE == current_size):
        return _MENU_BG_CACHE
    
    return None

def update_background_cache(bg_image, menu):
    global _MENU_BG_CACHE, _LAST_MENU_STATE, _LAST_MENU_POS, _LAST_SCREEN_SIZE
    _MENU_BG_CACHE = bg_image.copy()
    _LAST_MENU_STATE = menu.state.value
    _LAST_MENU_POS = menu.get_screen_position()
    _LAST_SCREEN_SIZE = (menu.menu_width, menu.menu_height)

def apply_glass_blur(frame, region_rect, blur_amount=15):
    """
    Skipping CPU blur because we now use GPU-accelerated blur in the rendering stage.
    This saves massive CPU cycles on 1080p frames.
    """
    return frame # No-op, GPU handles this now!

def draw_smooth_glass_rect(frame, x, y, w, h, radius, color=(255, 255, 255), alpha=0.6, blur=True):
    """High-fidelity glass rectangle with rounded corners and blur"""
    x, y, w, h = int(x), int(y), int(w), int(h)
    if w <= 0 or h <= 0: return frame
    
    # 1. Blur the background region
    if blur:
        apply_glass_blur(frame, (x, y, w, h), blur_amount=15)
        
    # 2. Draw the tinted overlay with rounded corners
    overlay = frame.copy()
    
    # Draw filled rounded rect on overlay
    # Cross shape
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1, cv2.LINE_AA)
    
    # Corners
    cv2.circle(overlay, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + w - radius, y + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + radius, y + h - radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, -1, cv2.LINE_AA)
    
    # Blend overlay with original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 3. Add a subtle white border/highlight
    cv2.line(frame, (x+radius, y), (x+w-radius, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (x, y+radius), (x, y+h-radius), (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def overlay_icon(frame, icon, cx, cy):
    """
    Overlay an RGBA icon onto the frame centered at (cx, cy).
    Handles alpha blending manually.
    """
    if icon is None: return
    
    h, w = icon.shape[:2]
    x = int(cx - w//2)
    y = int(cy - h//2)
    
    # Clip to frame bounds
    fh, fw = frame.shape[:2]
    
    # Calculate intersection
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    
    if x2 <= x1 or y2 <= y1: return
    
    # Dimensions of the visible icon area
    icon_x1 = x1 - x
    icon_y1 = y1 - y
    icon_x2 = icon_x1 + (x2 - x1)
    icon_y2 = icon_y1 + (y2 - y1)
    
    # Extract source and destination
    icon_crop = icon[icon_y1:icon_y2, icon_x1:icon_x2]
    frame_crop = frame[y1:y2, x1:x2]
    
    # Alpha blending
    if icon_crop.shape[2] == 4:
        alpha = icon_crop[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]
        
        # Invert colors? No, assume icon is BGR/RGB correctly. 
        # OpenCV imread UNCHANGED loads BGRA.
        img_rgb = icon_crop[:, :, :3]
        
        # Blend
        frame_crop[:] = (1.0 - alpha) * frame_crop + alpha * img_rgb
    else:
        frame_crop[:] = icon_crop

def draw_vector_icon(frame, x, y, size, name, color=(80, 80, 80), thickness=2):
    """Draw vector style icons using OpenCV primitives"""
    cx, cy = x + size//2, y + size//2
    
    name = name.lower()
    
    if "wall" in name:
        cv2.rectangle(frame, (cx-15, cy-15), (cx+15, cy+15), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx-15, cy), (cx+15, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy-15), (cx, cy+15), color, 1, cv2.LINE_AA)
    elif "sofa" in name or "chair" in name:
        cv2.rectangle(frame, (cx-15, cy), (cx+15, cy+15), color, thickness, cv2.LINE_AA)
        cv2.rectangle(frame, (cx-15, cy-15), (cx+15, cy), color, thickness, cv2.LINE_AA)
    elif "table" in name:
        cv2.rectangle(frame, (cx-20, cy-5), (cx+20, cy+5), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx-15, cy+5), (cx-15, cy+20), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx+15, cy+5), (cx+15, cy+20), color, thickness, cv2.LINE_AA)
    elif "door" in name:
        cv2.rectangle(frame, (cx-12, cy-20), (cx+12, cy+20), color, thickness, cv2.LINE_AA)
        cv2.circle(frame, (cx+5, cy), 2, color, -1, cv2.LINE_AA)
    elif "window" in name:
        cv2.rectangle(frame, (cx-15, cy-15), (cx+15, cy+15), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy-15), (cx, cy+15), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx-15, cy), (cx+15, cy), color, 1, cv2.LINE_AA)
    elif "light" in name or "lamp" in name:
        pts = np.array([[cx, cy-20], [cx-15, cy], [cx+15, cy]], np.int32)
        cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy+20), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx-10, cy+20), (cx+10, cy+20), color, thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (cx-10, cy-10), (cx+10, cy+10), color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "?", (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius):
    """Draw smooth rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    if x2 <= x1 or y2 <= y1: return
    
    max_radius = min((x2 - x1) // 2, (y2 - y1) // 2)
    radius = min(radius, max_radius)
    
    if thickness < 0:
        # Filled
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    else:
        # Outlined
        # 4 lines
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        
        # 4 arcs
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)

def draw_settings_icon(frame, menu, pointer_x, pointer_y):
    """Draw floating settings button below main menu button"""
    settings_x, settings_y = menu.get_settings_button_position()
    s_size = menu.minimized_button_size + 10
    
    is_hovered = (settings_x <= pointer_x <= settings_x + s_size and
                  settings_y <= pointer_y <= settings_y + s_size)

    # Debug every 60 frames roughly (assuming 30fps)

    
    # Glass Capsule (Circle)
    draw_smooth_glass_rect(frame, settings_x, settings_y, s_size, s_size,
                          radius=s_size//2, # Perfect circle
                          color=(255, 255, 255),
                          alpha=0.9 if is_hovered else 0.7,
                          blur=True)
    
    # Custom Icon
    cx, cy = settings_x + s_size // 2, settings_y + s_size // 2
    icon = icon_manager.get_icon("setting", size=32) # Load 'setting.png'
    
    if icon is not None:
        overlay_icon(frame, icon, cx, cy)
    else:
        # Fallback Vector Gear
        cv2.circle(frame, (cx, cy), 12, (80, 80, 90), 2, cv2.LINE_AA) 
        cv2.circle(frame, (cx, cy), 4, (80, 80, 90), -1, cv2.LINE_AA)
        for i in range(0, 360, 45):
            rad = math.radians(i)
            ox = int(cx + math.cos(rad) * 16)
            oy = int(cy + math.sin(rad) * 16)
            ix = int(cx + math.cos(rad) * 12)
            iy = int(cy + math.sin(rad) * 12)
            cv2.line(frame, (ix, iy), (ox, oy), (80, 80, 90), 3, cv2.LINE_AA)

def draw_settings_panel(frame, menu, pointer_x, pointer_y):
    """Draw settings toggles"""
    btn_x, btn_y = menu.get_settings_button_position()
    
    panel_w = 280
    row_h = 50
    header_h = 50
    panel_h = header_h + len(menu.settings) * row_h + 10
    
    # Position to the left of the button
    panel_x = btn_x - panel_w - 15
    panel_y = btn_y
    
    # Background
    draw_smooth_glass_rect(frame, panel_x, panel_y, panel_w, panel_h,
                          radius=18, color=(245, 245, 250), alpha=0.96, blur=True)
    
    # Header
    cv2.line(frame, (panel_x, panel_y + header_h), 
            (panel_x + panel_w, panel_y + header_h), (200, 200, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, "Settings", (panel_x + 20, panel_y + 33),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
               
    # Rows
    for i, key in enumerate(menu.settings.keys()):
        y = panel_y + header_h + i * row_h
        label = menu.settings_labels.get(key, key)
        is_on = menu.settings[key]
        
        # Divider
        if i > 0:
            cv2.line(frame, (panel_x + 15, y), (panel_x + panel_w - 15, y), (220, 220, 230), 1, cv2.LINE_AA)
        
        # Row Hover
        if panel_x <= pointer_x <= panel_x + panel_w and y <= pointer_y <= y + row_h:
            cv2.rectangle(frame, (panel_x+4, y+2), (panel_x+panel_w-4, y+row_h-2), (200, 215, 255), -1)
            
        # Label
        cv2.putText(frame, label, (panel_x + 20, y + 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 70), 1, cv2.LINE_AA)
        
        # Toggle Switch or Slider Display
        if isinstance(is_on, float) and key != "shutdown":
            # Slider Track
            slider_w = 120
            slider_x = panel_x + panel_w - slider_w - 20
            slider_y = y + row_h // 2
            
            # Value percentage (0.5 to 4.0)
            pct = (is_on - 0.5) / 3.5
            pct = max(0, min(pct, 1))
            
            # Track
            cv2.line(frame, (slider_x, slider_y), (slider_x + slider_w, slider_y), (200, 200, 210), 4, cv2.LINE_AA)
            # Active part
            knob_x = int(slider_x + pct * slider_w)
            cv2.line(frame, (slider_x, slider_y), (knob_x, slider_y), (100, 150, 255), 4, cv2.LINE_AA)
            # Knob
            cv2.circle(frame, (knob_x, slider_y), 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (knob_x, slider_y), 8, (100, 150, 255), 2, cv2.LINE_AA)
            
        else:
            toggle_w, toggle_h = 40, 22
            tx = panel_x + panel_w - toggle_w - 20
            ty = y + row_h // 2
            
            # On/Off colors
            if key == "shutdown":
                 t_color = (80, 80, 255) # Red for shutdown
            else:
                 t_color = (60, 200, 100) if is_on else (180, 180, 190)
                 
            # Pill
            cv2.rectangle(frame, (tx, ty - toggle_h//2), (tx + toggle_w, ty + toggle_h//2), t_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), toggle_h//2, t_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (tx + toggle_w, ty), toggle_h//2, t_color, -1, cv2.LINE_AA)
            
            # Knob
            if key == "shutdown":
                kx = tx + toggle_w // 2
            else:
                kx = tx + toggle_w if is_on else tx
                
            cv2.circle(frame, (kx, ty), toggle_h//2 - 2, (255, 255, 255), -1, cv2.LINE_AA)

def draw_menu(frame, menu, pointer_x, pointer_y):
    """Main render function"""
    if frame is None:
        return frame
    
    if menu.state.value == 0:  # Minimized
        draw_minimized_button(frame, menu, pointer_x, pointer_y)
    else:
        draw_expanded_menu(frame, menu, pointer_x, pointer_y)
    
    # Settings Button
    draw_settings_icon(frame, menu, pointer_x, pointer_y)
    
    # NEW: Virtual Mode Button
    draw_virtual_mode_button(frame, menu, pointer_x, pointer_y)
    
    # NEW: Import Model Button
    draw_import_button(frame, menu, pointer_x, pointer_y)
    
    if menu.selected_object_index != -1:
        draw_delete_button(frame, menu, pointer_x, pointer_y)
        draw_duplicate_button(frame, menu, pointer_x, pointer_y)
        draw_lock_button(frame, menu, pointer_x, pointer_y)
    
    if menu.show_settings:
        draw_settings_panel(frame, menu, pointer_x, pointer_y)
    
    return frame

def draw_minimized_button(frame, menu, pointer_x, pointer_y):
    """Draw floating minimized button"""
    btn_x, btn_y = menu.get_minimized_button_position()
    btn_size = menu.minimized_button_size + 10 
    
    is_hovered = (btn_x <= pointer_x <= btn_x + btn_size and
                  btn_y <= pointer_y <= btn_y + btn_size)
    
    # Glass Capsule (Circle)
    draw_smooth_glass_rect(frame, btn_x, btn_y, btn_size, btn_size, 
                          radius=btn_size//2, # Perfect circle
                          color=(255, 255, 255), 
                          alpha=0.9 if is_hovered else 0.7, 
                          blur=True)
                          
    # Custom Icon
    cx, cy = btn_x + btn_size//2, btn_y + btn_size//2
    icon = icon_manager.get_icon("hamburger", size=32) # Load 'hamburger.png'
    
    if icon is not None:
        overlay_icon(frame, icon, cx, cy)
    else:
        # Fallback Hamburger
        icon_color = (40, 40, 40)
        for i in range(-1, 2):
            y = cy + i * 9
            cv2.line(frame, (cx - 13, y), (cx + 13, y), icon_color, 4, cv2.LINE_AA)
    

    
    return frame



def draw_expanded_menu(frame, menu, pointer_x, pointer_y):
    """Draw full expanded menu"""
    menu_x, menu_y = menu.get_screen_position()
    menu_w, menu_h = menu.menu_width, menu.menu_height
    
    if menu_w <= 0 or menu_h <= 0: return frame
    
    # Draw Glass Background (Always fresh because of transparency over camera)
    draw_smooth_glass_rect(frame, menu_x, menu_y, menu_w, menu_h, radius=20, 
                          color=(250, 250, 255), alpha=0.85, blur=True)
    
    draw_rounded_rectangle(frame, (menu_x, menu_y), (menu_x+menu_w, menu_y+menu_h),
                          (255, 255, 255), 1, radius=20)
    
    # Component Rendering
    draw_header(frame, menu, menu_x, menu_y, menu_w)
    draw_sidebar(frame, menu, menu_x, menu_y, menu_w, menu_h, pointer_x, pointer_y)
    draw_content_area(frame, menu, menu_x, menu_y, menu_w, menu_h, pointer_x, pointer_y)
    
    # Resize handles
    if menu.state.value in [1, 3]:
        draw_resize_handles(frame, menu, menu_x, menu_y, menu_w, menu_h, pointer_x, pointer_y)
        
    return frame

def draw_header(frame, menu, menu_x, menu_y, menu_w):
    header_h = 60
    # Header bg
    cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_w, menu_y + header_h), (250, 250, 250, 100), -1)
    
    # Divider
    cv2.line(frame, (menu_x, menu_y+header_h), (menu_x+menu_w, menu_y+header_h), (200, 200, 200), 1, cv2.LINE_AA)
    
    # Title
    cv2.putText(frame, "Design Studio", (menu_x + 30, menu_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
               
    # Window controls
    btn_y = menu_y + 30
    cx, mx, gx = menu_x + menu_w - 90, menu_x + menu_w - 55, menu_x + menu_w - 20
    # Close
    cv2.circle(frame, (cx, btn_y), 10, (86, 95, 255), -1, cv2.LINE_AA)
    cv2.line(frame, (cx-4, btn_y-4), (cx+4, btn_y+4), (255,255,255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx+4, btn_y-4), (cx-4, btn_y+4), (255,255,255), 2, cv2.LINE_AA)
    # Min
    cv2.circle(frame, (mx, btn_y), 10, (46, 189, 255), -1, cv2.LINE_AA)
    cv2.line(frame, (mx-5, btn_y), (mx+5, btn_y), (255,255,255), 2, cv2.LINE_AA)
    # Max
    cv2.circle(frame, (gx, btn_y), 10, (63, 201, 39), -1, cv2.LINE_AA)

def draw_sidebar(frame, menu, menu_x, menu_y, menu_w, menu_h, pointer_x, pointer_y):
    sidebar_w = int(menu_w * menu.sidebar_ratio)
    sidebar_x = menu_x
    sidebar_y = menu_y + 60
    sidebar_h = menu_h - 60
    
    # Items
    item_h, gap = 65, 8
    # Calculate Scroll Limits (Sidebar)
    total_h = len(menu.nav_items) * (item_h + gap)
    menu.max_sidebar_scroll = max(0, total_h - sidebar_h + 20)
    
    # â”€â”€â”€ ROI RENDERING (For Clipping) â”€â”€â”€
    # Create a local buffer for the sidebar content
    # We will draw everything relative to (0,0) of this buffer
    # and then copy it to the main frame.
    sidebar_img = np.full((sidebar_h, sidebar_w, 3), (230, 230, 235), dtype=np.uint8)
    
    # Draw Divider on the buffer (Right side)
    cv2.line(sidebar_img, (sidebar_w-1, 0), (sidebar_w-1, sidebar_h), (200, 200, 200), 2)
    
    # Content Start Y relative to buffer
    # Global start was `sidebar_y + 20`. In buffer, it is `20`.
    # Apply scroll offset.
    local_start_y = 20 - int(menu.sidebar_scroll_y)
    
    for i, item in enumerate(menu.nav_items):
        # Coordinates in the BUFFER
        local_item_y = local_start_y + i * (item_h + gap)
        local_item_x = 10
        item_w = sidebar_w - 20
        
        # Optimization: Skip if fully out of buffer
        if local_item_y + item_h < 0: continue
        if local_item_y > sidebar_h: continue
        
        # Update Item Bounds (Must be GLOBAL for hit testing)
        global_item_y = sidebar_y + local_item_y
        global_item_x = sidebar_x + 10
        item.set_bounds(global_item_x, global_item_y, item_w, item_h)
        
        # Interactions (Use global pointer vs global bounds)
        is_sel = (menu.selected_category == item.id)
        # Check if hovering AND pointer is within the visible sidebar ROI
        pointer_in_roi = (sidebar_y <= pointer_y < sidebar_y + sidebar_h) and (sidebar_x <= pointer_x < sidebar_x + sidebar_w)
        is_hov = item.is_clicked(pointer_x, pointer_y) and pointer_in_roi
        
        # Draw on BUFFER using LOCAL coordinates
        if is_sel:
            draw_rounded_rectangle(sidebar_img, (local_item_x, local_item_y), 
                                  (local_item_x+item_w, local_item_y+item_h),
                                  (220, 230, 255), -1, radius=12)
            draw_rounded_rectangle(sidebar_img, (local_item_x, local_item_y), 
                                  (local_item_x+item_w, local_item_y+item_h),
                                  (100, 150, 255), 2, radius=12)
        elif is_hov:
            draw_rounded_rectangle(sidebar_img, (local_item_x, local_item_y), 
                                  (local_item_x+item_w, local_item_y+item_h),
                                  (240, 240, 245), -1, radius=12)
        
        color = (100, 150, 255) if is_sel else (80, 80, 80)
        # Text also needs local coordinates
        cv2.putText(sidebar_img, item.label, (local_item_x + 15, int(local_item_y + 42)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1 if not is_sel else 2, cv2.LINE_AA)

    # Copy buffer to main frame
    # (sidebar_x, sidebar_y) is top-left in global frame
    frame[sidebar_y:sidebar_y+sidebar_h, sidebar_x:sidebar_x+sidebar_w] = sidebar_img

def draw_content_area(frame, menu, menu_x, menu_y, menu_w, menu_h, pointer_x, pointer_y):
    sidebar_w = int(menu_w * menu.sidebar_ratio)
    content_x = menu_x + sidebar_w
    content_y = menu_y + 60
    content_w = menu_w - sidebar_w
    content_h = menu_h - 60
    
    cards = menu.get_current_cards()
    if not cards:
        cv2.putText(frame, "No items", (content_x + 50, content_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        return

    card_w, card_h = 160, 180
    gap = 20
    cards_per_row = max(1, (content_w - 40) // (card_w + gap))
    

    
    # Calculate Scroll Limits (Content)
    num_rows = (len(cards) + cards_per_row - 1) // cards_per_row
    total_h = num_rows * (card_h + gap)
    menu.max_content_scroll = max(0, total_h - content_h + 20)

    # â”€â”€â”€ ROI RENDERING â”€â”€â”€
    # Local buffer for content area
    content_img = np.full((content_h, content_w, 3), (255, 255, 255), dtype=np.uint8) # White background? Default is usually transparent-ish or gray
    # Let's match previous look: it was transparent on top of glass blur? 
    # The previous code didn't draw a background rect for content area, only sidebar.
    # But for ROI clipping we need a surface. 
    # Let's use a very light gray/white to match the cards look, or Keep it transparent?
    # If we want transparent background but clipped items, we can:
    # 1. Start with copy of frame region (bg) -> NO, that double-blurs or looks weird if frame changes.
    # 2. Start with transparent (0,0,0,0) image -> Draw items -> Overlay using alpha mask.
    # But `frame` is BGR.
    # Simple fix: Use a solid background color for the content area. 
    # Sidebar uses (230...). Let's use (245, 245, 250) for content area slightly lighter.
    content_img[:] = (245, 245, 250) 
    
    # Local Start Y
    local_start_y = 20 - int(menu.content_scroll_y)
    local_side_margin = 20

    for i, card in enumerate(cards):
        row, col = i // cards_per_row, i % cards_per_row
        
        # Local Coordinates
        local_card_x = local_side_margin + col * (card_w + gap)
        local_card_y = local_start_y + row * (card_h + gap)
        
        # Skip if OOB
        if local_card_y + card_h < 0: continue
        if local_card_y > content_h: continue
        
        # Update Global Bounds
        global_card_x = content_x + local_card_x
        global_card_y = content_y + local_card_y
        card.set_bounds(global_card_x, global_card_y, card_w, card_h)
        
        # Interactions
        pointer_in_roi = (content_y <= pointer_y < content_y + content_h) and (content_x <= pointer_x < content_x + content_w)
        is_hov = card.is_clicked(pointer_x, pointer_y) and pointer_in_roi
        
        # Draw on BUFFER
        # Bg
        draw_rounded_rectangle(content_img, (local_card_x, local_card_y), 
                              (local_card_x+card_w, local_card_y+card_h),
                              (255, 255, 255), -1, radius=15)
        
        # Border
        b_col = (100, 150, 255) if is_hov else (220, 220, 220)
        draw_rounded_rectangle(content_img, (local_card_x, local_card_y), 
                              (local_card_x+card_w, local_card_y+card_h),
                              b_col, 3 if is_hov else 2, radius=15)
        
        # Icon
        draw_vector_icon(content_img, local_card_x, local_card_y+10, card_w, card.name, 
                        (80, 120, 200) if is_hov else (100, 100, 100))
        
        # Text
        cv2.putText(content_img, card.name[:14], (local_card_x + 10, int(local_card_y + card_h - 20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    # Copy to frame
    frame[content_y:content_y+content_h, content_x:content_x+content_w] = content_img

def draw_resize_handles(frame, menu, menu_x, menu_y, menu_w, menu_h, pointer_x, pointer_y):
    # Simplest: 4 corners
    corners = [(menu_x, menu_y), (menu_x+menu_w, menu_y), (menu_x, menu_y+menu_h), (menu_x+menu_w, menu_y+menu_h)]
    hov_handle = menu._get_resize_handle(pointer_x, pointer_y)
    
    for i, (cx, cy) in enumerate(corners):
        is_hov = (hov_handle.value == i + 1)
        col = (100, 150, 255) if is_hov else (180, 180, 180)
        cv2.circle(frame, (cx, cy), 8, col, -1, cv2.LINE_AA)



from core.gpu_renderer import GPURenderer as Renderer

# Initialize renderer lazily or globally? 
# We should probably pass it or reuse it. 
# For now, let's create a global one or attach to menu? 
# Attaching to local function for now, but better to be persistent.
_RENDERER = None

def get_renderer(w, h):
    global _RENDERER
    if _RENDERER is None:
        try:
            _RENDERER = Renderer(w, h)
        except Exception:
            print("âš ï¸ GPU Renderer failed, falling back or exiting 3D")
            _RENDERER = None # Handled in caller
    return _RENDERER



def draw_placed_objects_and_held(frame, menu, pointer_x, pointer_y, view_matrix=None):
    """
    Renders 3D objects using the GPU renderer.
    """
    if frame is None: return frame
    
    h, w = frame.shape[:2]
    
    # Collect all objects to render
    objects_to_render = []
    objects_to_render.extend(menu.placed_objects)
    
    if menu.held_object:
        objects_to_render.append(menu.held_object)
        
    if not objects_to_render:
        return frame
        
    # Render with GPU (dynamic size via get_renderer)
    renderer = get_renderer(w, h)
    
    # Calculate blur regions (e.g. behind menu panels)
    blur_regions = []
    if menu.state.value != 0: # Not minimized
         # Menu background
         mx, my = menu.get_screen_position()
         blur_regions.append((mx, my, menu.menu_width, menu.menu_height))
         
    # Settings panel blur
    if menu.show_settings:
        bx, by = menu.get_settings_button_position()
        # approximate panel rect
        blur_regions.append((bx - 300, by, 280, 500)) 
    
    rendered_frame = renderer.render(
        frame, 
        objects_to_render, 
        camera_fov=60.0, 
        blur_regions=blur_regions,
        view_matrix=view_matrix,
        draw_grid=menu.settings.get('virtual_mode', False)
    )
    
    
    # Draw overlays (selection, drop hint) separately on top of 2D frame
    # Camera Intrinsics Approx
    fov_rad = math.radians(60.0)
    f = h / (2.0 * math.tan(fov_rad / 2.0))
    cx, cy = w / 2, h / 2
    
    # Ensure view matrix is consistently a numpy array
    if view_matrix is None:
        # Match GPU renderer fallback: camera pushed back 500 units on Z
        from pyrr import Matrix44 as M44
        view_idx = np.array(M44.from_translation([0.0, 0.0, -500.0]), dtype=np.float32)
    else:
        view_idx = np.array(view_matrix, dtype=np.float32)
        
    def project_point(pos_world):
        # Convert to homogeneous
        p_world = np.array([pos_world[0], pos_world[1], pos_world[2], 1.0])
        
        # Transform World -> Camera Space
        # p_cam = View * P_world
        p_cam = view_idx @ p_world
        
        x, y, z = p_cam[0], p_cam[1], p_cam[2]
        
        # Check if behind camera (OpenGL Camera looks down -Z)
        if z > -0.1: 
            return None
            
        # Project Camera -> Screen
        # x_screen = cx + (x / -z) * f
        # y_screen = cy - (y / -z) * f  (Flip Y because Screen Y is down)
        
        sx = int(cx + (x / -z) * f)
        sy = int(cy - (y / -z) * f)
        return (sx, sy)

    for i, obj in enumerate(menu.placed_objects):
        # Highlight if selected
        if i == menu.selected_object_index:
            screen_pos = project_point(obj.position)
            
            if screen_pos:
                sx, sy = screen_pos
                cv2.circle(rendered_frame, (sx, sy), 10, (0, 255, 0), 2)
                cv2.putText(rendered_frame, "SELECTED", (sx+15, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if menu.held_object:
        screen_pos = project_point(menu.held_object.position)
        if screen_pos:
            sx, sy = screen_pos
            cv2.putText(rendered_frame, "DROP", (sx-20, sy-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    if menu.held_object and menu.settings.get('auto_snap', True):
        # Visualize Snap
        held_corners = menu.held_object.get_world_corners()
        snap_pt_screen = None
        
        for other in menu.placed_objects:
             if other is menu.held_object: continue
             if menu.held_object.group_id and menu.held_object.group_id == other.group_id: continue
             
             # Optimization
             if np.linalg.norm(menu.held_object.position - other.position) > 300: continue

             other_corners = other.get_world_corners()
             for hc in held_corners:
                 for oc in other_corners:
                     # Check if vertices are effectively identical (snapped)
                     if np.linalg.norm(hc - oc) < 1.0: 
                         snap_pt_screen = project_point(oc)
                         if snap_pt_screen:
                             sx, sy = snap_pt_screen
                             # Draw Green Crosshair
                             cv2.circle(rendered_frame, (sx, sy), 6, (0, 255, 0), 2, cv2.LINE_AA)
                             cv2.line(rendered_frame, (sx-10, sy), (sx+10, sy), (0, 255, 0), 2, cv2.LINE_AA)
                             cv2.line(rendered_frame, (sx, sy-10), (sx, sy+10), (0, 255, 0), 2, cv2.LINE_AA)
                             cv2.putText(rendered_frame, "CONNECTED", (sx+15, sy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                             break
                 if snap_pt_screen: break
             if snap_pt_screen: break

    return rendered_frame



def draw_menu_hint(frame, show_hint):
    if not show_hint: return frame
    
    # Simple help box
    h = frame.shape[0]
    cv2.rectangle(frame, (20, h-100), (400, h-20), (240, 240, 240), -1)
    lines = ["Click button to expand", "Drag header to move", "Drag corners to resize"]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (30, h-80 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
        
    return frame


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  COLOR PALETTE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Preset colors: (name, R, G, B)
COLOR_PALETTE = [
    ("White",   255, 255, 255),
    ("Black",    30,  30,  30),
    ("Red",     220,  50,  50),
    ("Blue",     50,  90, 220),
    ("Green",    50, 180,  70),
    ("Yellow",  240, 210,  50),
    ("Orange",  240, 140,  40),
    ("Purple",  150,  60, 200),
    ("Pink",    240, 130, 170),
    ("Brown",   140,  90,  50),
    ("Teal",     50, 180, 180),
    ("Gray",    150, 150, 150),
]

SWATCH_RADIUS = 18
SWATCH_SPACING = 48


def draw_color_palette(frame, menu, pointer_x=-1, pointer_y=-1):
    """Draw floating color palette at bottom of screen when an object is selected.
    Returns list of (cx, cy, (r, g, b)) for hit-testing.
    """
    if menu.selected_object_index == -1:
        return []
    
    h, w = frame.shape[:2]
    
    # Palette bar position
    num_colors = len(COLOR_PALETTE)
    total_width = num_colors * SWATCH_SPACING
    start_x = (w - total_width) // 2 + SWATCH_SPACING // 2
    bar_y = 50  # Top of screen
    
    # Semi-transparent background bar
    bar_x1 = start_x - SWATCH_RADIUS - 12
    bar_x2 = start_x + total_width - SWATCH_SPACING + SWATCH_RADIUS + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x1, bar_y - SWATCH_RADIUS - 12),
                  (bar_x2, bar_y + SWATCH_RADIUS + 12),
                  (30, 30, 30), -1)
    cv2.rectangle(overlay, (bar_x1, bar_y - SWATCH_RADIUS - 12),
                  (bar_x2, bar_y + SWATCH_RADIUS + 12),
                  (80, 80, 80), 2)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # "Color" label
    cv2.putText(frame, "COLOR", (bar_x1 + 8, bar_y - SWATCH_RADIUS - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    swatch_positions = []
    
    for i, (name, r, g, b) in enumerate(COLOR_PALETTE):
        cx = start_x + i * SWATCH_SPACING
        cy = bar_y
        
        # BGR for OpenCV
        bgr = (b, g, r)
        
        # Check if pointer is hovering
        dist = np.hypot(pointer_x - cx, pointer_y - cy) if pointer_x > 0 else 999
        hover = dist < SWATCH_RADIUS + 5
        
        # Draw swatch
        cv2.circle(frame, (cx, cy), SWATCH_RADIUS, bgr, -1)
        
        # Border (white, thicker if hovered)
        border_color = (255, 255, 255) if hover else (180, 180, 180)
        border_thick = 3 if hover else 1
        cv2.circle(frame, (cx, cy), SWATCH_RADIUS, border_color, border_thick)
        
        # Show name on hover
        if hover:
            cv2.putText(frame, name, (cx - 20, cy - SWATCH_RADIUS - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        swatch_positions.append((cx, cy, (r, g, b)))
    
    return swatch_positions

def draw_virtual_mode_button(frame, menu, pointer_x, pointer_y):
    """Draw Virtual Mode toggle (Top-Left)"""
    vx, vy = menu.get_virtual_mode_button_position()
    size = menu.minimized_button_size
    
    is_hovered = menu._is_over_virtual_mode_button(pointer_x, pointer_y)
    is_active = menu.settings.get('virtual_mode', False)
    

    
    # Glass Capsule (Circle)
    color = (200, 255, 200) if is_active else (255, 255, 255)
    draw_smooth_glass_rect(frame, vx, vy, size, size,
                          radius=size//2, # Perfect circle
                          color=color,
                          alpha=0.9 if is_hovered else 0.7,
                          blur=True)
    
    # Custom Icon
    cx, cy = vx + size//2, vy + size//2
    icon = icon_manager.get_icon("virtual-tour", size=32) # Load 'virtual-tour.png'
    
    if icon is not None:
        overlay_icon(frame, icon, cx, cy)
    else:
        # Fallback VR Headset Icon
        # Goggles
        cv2.rectangle(frame, (cx-18, cy-10), (cx+18, cy+10), (40, 40, 40), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx-18, cy-10), (cx+18, cy+10), (80, 80, 80), 2, cv2.LINE_AA)
        # Strap
        cv2.line(frame, (cx-18, cy), (cx-25, cy), (80, 80, 80), 2, cv2.LINE_AA)
        cv2.line(frame, (cx+18, cy), (cx+25, cy), (80, 80, 80), 2, cv2.LINE_AA)
    
    # Label for clarity (optional)
    if is_hovered:
        cv2.putText(frame, "Virtual Mode", (vx + size + 10, cy + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_import_button(frame, menu, pointer_x, pointer_y):
    """Draw Import Model button (left of Virtual Mode)"""
    ix, iy = menu.get_import_button_position()
    size = menu.minimized_button_size
    
    is_hovered = menu._is_over_import_button(pointer_x, pointer_y)
    is_busy = getattr(menu, '_import_dialog_open', False)
    
    # Glass Capsule (Circle)
    color = (255, 220, 180) if is_busy else (255, 255, 255)
    draw_smooth_glass_rect(frame, ix, iy, size, size,
                          radius=size//2,
                          color=color,
                          alpha=0.9 if is_hovered else 0.7,
                          blur=True)
    
    # Custom Icon
    cx, cy = ix + size//2, iy + size//2
    icon = icon_manager.get_icon("import", size=32)  # Load 'import.png'
    
    if icon is not None:
        overlay_icon(frame, icon, cx, cy)
    else:
        # Fallback "+" Plus Icon (import/add)
        icon_color = (80, 80, 80)
        cv2.line(frame, (cx - 14, cy), (cx + 14, cy), icon_color, 4, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 14), (cx, cy + 14), icon_color, 4, cv2.LINE_AA)
    
    # Label on hover
    if is_hovered:
        cv2.putText(frame, "Import Model", (ix + size + 10, cy + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_delete_button(frame, menu, pointer_x, pointer_y):
    """Draw Delete Button (Bottom-Center)"""
    dx, dy = menu.get_delete_button_position()
    w = menu.minimized_button_size + 10
    h = w # Square/Circle
    
    is_hovered = menu._is_over_delete_button(pointer_x, pointer_y)
    
    # Red Glass Circle
    cx, cy = dx + w//2, dy + h//2
    radius = 35
    
    # Pulse effect if hovered
    if is_hovered:
        cv2.circle(frame, (cx, cy), radius + 4, (100, 100, 255), 4, cv2.LINE_AA)
        
    draw_smooth_glass_rect(frame, dx, dy, w, h, radius=w//2,
                          color=(100, 100, 255), # Red tint
                          alpha=0.9,
                          blur=True)
                          
    # Custom Icon
    cx, cy = dx + w // 2, dy + h // 2
    icon = icon_manager.get_icon("delete", size=32)
    if icon is not None:
        overlay_icon(frame, icon, cx, cy)
    else:
        # Fallback Trash Can Icon
        # Lid
        cv2.line(frame, (cx-12, cy-12), (cx+12, cy-12), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (cx-4, cy-16), (cx+4, cy-16), (255, 255, 255), 2, cv2.LINE_AA) # Handle
        # Bin
        pts = np.array([[cx-10, cy-12], [cx+10, cy-12], [cx+8, cy+15], [cx-8, cy+15]], np.int32)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
    # Label
    if is_hovered:
        cv2.putText(frame, "DELETE", (dx, dy + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

def draw_duplicate_button(frame, menu, pointer_x, pointer_y):
    """Draw Duplicate Button (Below Virtual Mode)"""
    dx, dy = menu.get_duplicate_button_position()
    w = menu.minimized_button_size + 10
    h = w
    
    is_hovered = menu._is_over_duplicate_button(pointer_x, pointer_y)
    
    # Blue/Green Glass Circle
    cx, cy = dx + w//2, dy + h//2
    radius = 35
    
    # Pulse effect if hovered
    if is_hovered:
        cv2.circle(frame, (cx, cy), radius + 4, (100, 255, 100), 4, cv2.LINE_AA)
        
    draw_smooth_glass_rect(frame, dx, dy, w, h, radius=w//2,
                          color=(100, 255, 100), # Green tint
                          alpha=0.9,
                          blur=True)
                          
    # Custom Icon
    cx, cy = dx + w // 2, dy + h // 2
    icon = icon_manager.get_icon("duplicate", size=32)
    if icon is not None:
        overlay_icon(frame, icon, cx, cy)
    else:
        # Fallback Duplicate Icon (Two overlapping rects/pages)
        # Background Page
        cv2.rectangle(frame, (cx-5, cy-10), (cx+12, cy+8), (220, 220, 220), 2, cv2.LINE_AA)
        # Foreground Page (Filled)
        cv2.rectangle(frame, (cx-12, cy-6), (cx+5, cy+12), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx-12, cy-6), (cx+5, cy+12), (50, 50, 50), 1, cv2.LINE_AA) # Border
    
    # Label
    # Label
    if is_hovered:
        cv2.putText(frame, "Duplicate", (dx + 0, dy + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_lock_button(frame, menu, pointer_x, pointer_y):
    """Draw Lock/Unlock Button (Next to Duplicate)"""
    lx, ly = menu.get_lock_button_position()
    w = menu.minimized_button_size + 10
    h = w
    
    is_hovered = menu._is_over_lock_button(pointer_x, pointer_y)
    
    # Check if currently locked
    obj = menu.placed_objects[menu.selected_object_index]
    is_locked = (obj.group_id is not None)
    
    # Gold (locked) or White/Gray (unlocked)
    if is_locked:
        color = (0, 215, 255) # Gold
    else:
        color = (200, 200, 200) # Gray
    
    # Pulse effect if hovered
    cx, cy = lx + w//2, ly + h//2
    radius = 35
    if is_hovered:
        cv2.circle(frame, (cx, cy), radius + 4, color, 4, cv2.LINE_AA)
        
    draw_smooth_glass_rect(frame, lx, ly, w, h, radius=w//2,
                          color=color,
                          alpha=0.9,
                          blur=True)
                          
    # Icon
    # Lock body
    cv2.rectangle(frame, (cx-10, cy-5), (cx+10, cy+12), (255, 255, 255), -1, cv2.LINE_AA)
    # Shackle
    if is_locked:
        # Closed
        cv2.ellipse(frame, (cx, cy-5), (8, 8), 0, 180, 360, (255, 255, 255), 3, cv2.LINE_AA)
    else:
        # Open
        cv2.ellipse(frame, (cx+4, cy-5), (8, 8), 0, 180, 330, (255, 255, 255), 3, cv2.LINE_AA)

    # Label
    if is_hovered:
        label = "UNLOCK" if is_locked else "LOCK"
        cv2.putText(frame, label, (lx + 5, ly + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)