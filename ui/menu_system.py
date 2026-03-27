
import numpy as np
from enum import Enum
import time
import threading

from config import CLICK_COOLDOWN, GRID_SIZE, MAGNETIC_BORDER_PX
from core.graphics_engine import Object3D, Mesh
from core.constraint_manager import ConstraintManager
from meshes import primitive_meshes
from meshes import furniture_meshes
from loaders.obj_loader import load_obj
from loaders.glb_loader import load_glb
import os

# Pointer radius — matches the visual ring drawn in main.py (cv2.circle radius=22)
POINTER_RADIUS = 22

# Camera FOV must match gpu_renderer.py default (60 degrees)
CAMERA_FOV = 60.0

class MenuState(Enum):
    MINIMIZED = 0  # Floating button
    OPEN = 1       # Full menu
    DRAGGING = 2   # Being moved
    RESIZING = 3   # Being resized
    PLACING_OBJECT = 4 # Placing an object
    ROTATING_OBJECT = 5 # Rotating an object

class ResizeHandle(Enum):
    NONE = 0
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4
    LEFT = 5
    RIGHT = 6
    TOP = 7
    BOTTOM = 8

class MenuItem:
    """Navigation item (sidebar)"""
    def __init__(self, id, label, icon="", image_path=None):
        self.id = id
        self.label = label
        self.icon = icon
        self.image_path = image_path
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        
    def set_bounds(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    def is_clicked(self, px, py):
        # Expand hit-area by POINTER_RADIUS so edge of pointer ring triggers
        return (self.x - POINTER_RADIUS <= px <= self.x + self.width + POINTER_RADIUS and
                self.y - POINTER_RADIUS <= py <= self.y + self.height + POINTER_RADIUS)

class ItemCard:
    """Content card (right side)"""
    def __init__(self, id, name, category, thumbnail=None, mesh_type="cube", model_path=None):
        self.id = id
        self.name = name
        self.category = category
        self.thumbnail = thumbnail
        self.mesh_type = mesh_type # "cube", "pyramid", "obj"
        self.model_path = model_path
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        
    def set_bounds(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    def is_clicked(self, px, py):
        # Expand hit-area by POINTER_RADIUS so edge of pointer ring triggers
        return (self.x - POINTER_RADIUS <= px <= self.x + self.width + POINTER_RADIUS and
                self.y - POINTER_RADIUS <= py <= self.y + self.height + POINTER_RADIUS)

class AppleGlassMenu:
    """
    Apple Vision Pro Style Menu
    Features: Minimize, Resize, Split Layout, Scrolling
    """
    def __init__(self, screen_width=1920, screen_height=1080):
        self.state = MenuState.MINIMIZED  # Start minimized
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Menu dimensions (when expanded)
        self.pos_x = 0.15  # Center-left
        self.pos_y = 0.1
        self.menu_width = 900   # Large menu
        self.menu_height = 700
        
        # Minimum sizes
        self.min_width = 600
        self.min_height = 400
        self.max_width = 1400
        self.max_height = 900
        

        
        # Minimized button
        self.minimized_button_size = 70
        self.minimized_pos_x = 0.93  # Bottom-left corner
        self.minimized_pos_y = 0.02
        
        # Layout proportions
        self.sidebar_ratio = 0.25  # 25% sidebar, 75% content
        
        # Dragging & Resizing
        self.is_selected = False
        self.resize_handle = ResizeHandle.NONE
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.resize_start_width = 0
        self.resize_start_height = 0
        
        # Debouncing
        self.last_click_time = 0
        
        # Object Placement (3D)
        self.held_object = None  # Currently holding this Object3D
        self.holding_hand_id = -1 # ID of the hand holding the object
        self.placed_objects = [] # List of Object3D instances
        self.selected_object_index = -1 # Index of currently selected object for editing
        
        # Manipulation vars
        self.initial_pinch_dist = 0
        self.initial_scale = 1.0
        self.initial_rotation_y = 0
        self.initial_rotation_x = 0
        self.manipulation_start_x = 0
        self.manipulation_start_y = 0
        
        # Virtual Orbit Camera (Yaw, Pitch, Distance)
        self.virtual_cam_yaw = 0.0   # Degrees
        self.virtual_cam_pitch = 20.0 # Degrees (slightly looking down)
        self.virtual_cam_dist = 300.0 # Distance from target
        self.virtual_cam_target = np.array([0.0, 0.0, 500.0], dtype=np.float32)
        
        # Tap-based selection
        self._tap_start_time = 0       # When pinch START happened
        self._tap_start_pos = (0, 0)   # Where pinch started
        self._tap_consumed = False     # Prevents double-fire within one pinch cycle
        self._rotating = False         # True while pinch-hold rotation is active
        self._rotating_hand_index = -1 # ID of the hand controlling rotation
        self._last_selection_time = 0  # Cooldown timer for selection changes
        self._last_settings_click_time = 0 # Cooldown for settings toggles

        self._initial_hand_distance = 0  # Baseline for two-hand scaling
        
        # Mesh cache to avoid reloading the same model
        self._mesh_cache = {}  # path -> (mesh, material_groups_tuples)
        
        # Constraint Manager (replaces legacy group_id logic)
        self.constraint_manager = ConstraintManager()
        
        # Snap state for visual feedback
        self._last_snap_info = None  # (snap_point, snap_face_a, snap_face_b, other_obj)
        
        # Scrolling
        # Scrolling
        self.sidebar_scroll_y = 0
        self.max_sidebar_scroll = 0
        
        self.content_scroll_y = 0
        self.max_content_scroll = 0
        
        # Settings
        self.show_settings = False
        self.settings = {
            'performance': False,
            'hit_boxes': False,
            'fps': False,
            'skeleton': False,
            'gestures': True,
            'pointers': False,
            'sensitivity_x': 0.5,
            'sensitivity_y': 0.5,
            'virtual_mode': False,
            'auto_snap': True,
            'grid_snap': False,
            'shutdown': False
        }
        self.content_scroll_y = 0
        self.max_content_scroll = 0 # Will be calculated by renderer or known item count
        self.virtual_mode_progress = 0.0 # 0.0 (Off) to 1.0 (On)
        self.last_update_time = time.time()
        
        self.settings_labels = {
            'performance': 'Performance Stats',
            'hit_boxes': 'Show Hitboxes',
            'fps': 'Show FPS',
            'skeleton': 'Hand Skeleton',
            'gestures': 'Gesture Details',
            'pointers': 'Pointer Loc',
            'sensitivity_x': 'Sensitivity X',
            'sensitivity_y': 'Sensitivity Y',
            'virtual_mode': 'Virtual Mode',
            'auto_snap': 'Auto Snapping',
            'grid_snap': 'Grid Snapping',
            'shutdown': 'Shutdown App'
        }
        self.selected_category = "walls"
        
        # Navigation items (sidebar)
        self.nav_items = [
            MenuItem("walls", "Walls" ),
            MenuItem("furniture", "Furniture"),
            MenuItem("doors", "Doors & Gates" ),
            MenuItem("windows", "Windows"),
            MenuItem("lighting", "Lighting"),
            MenuItem("decor", "Decorations",),
            MenuItem("Completed Models", "Completed Models"),
            MenuItem("colors", "Colors"),

        ]
        
        # Item database (cards shown in content area)
        self.item_database = {
            "walls": [
                ItemCard("wall_1", "Wall Panel", "walls", mesh_type="furniture_mesh", model_path="assets/models/walls/Wall_Base.obj"),
                ItemCard("wall_2", "Brick Wall", "walls", mesh_type="cube"),
                ItemCard("wall_3", "Concrete Wall", "walls", mesh_type="cube"),
            ],
            "furniture": [
                ItemCard("sofa_1", "Modern Sofa", "furniture", mesh_type="furniture_mesh", model_path="sofa"),
                ItemCard("sofa_2", "L-Shape Sofa", "furniture", mesh_type="furniture_mesh", model_path="l_sofa"),
                ItemCard("chair_1", "Chair", "furniture", mesh_type="furniture_mesh", model_path="chair"),
                ItemCard("table_1", "Coffee Table", "furniture", mesh_type="furniture_mesh", model_path="table"),
                ItemCard("bed_1", "King Bed", "furniture", mesh_type="furniture_mesh", model_path="bed"),
                ItemCard("car_1", "Cartoon Car", "furniture", mesh_type="glb", model_path="assets/models/cartoon_car.glb"),
            ],
            "doors": [
                ItemCard("door_1", "Door", "doors", mesh_type="furniture_mesh", model_path="door"),
            ],
            "windows": [
                ItemCard("window_1", "Window", "windows", mesh_type="furniture_mesh", model_path="window"),
            ],
            "lighting": [
                ItemCard("lamp_1", "Standing Lamp", "lighting", mesh_type="furniture_mesh", model_path="lamp"),
            ],
            "decor": [
                ItemCard("shelf_1", "Bookshelf", "decor", mesh_type="furniture_mesh", model_path="shelf"),
                ItemCard("plant_1", "Plant Pot", "decor", mesh_type="pyramid"),
            ],
            "Completed Models": [
                ItemCard("cottage_1", "Cottage", "Completed Models", mesh_type="furniture_mesh", model_path="assets/models/cottage_obj.obj"),
                ItemCard("floor_1", "Concrete Floor", "Completed Models", mesh_type="furniture_mesh", model_path="assets/models/Floor/Floor_Concrete.obj"),
                ItemCard("wall_base_1", "Wall Base", "Completed Models", mesh_type="furniture_mesh", model_path="assets/models/walls/Wall_Base.obj"),
                ItemCard("car_2", "Cartoon Car", "Completed Models", mesh_type="glb", model_path="assets/models/cartoon_car.glb"),
            ],

        }
    

    def resize(self, new_width, new_height):
        """Update screen dimensions and constrain menu size"""
        self.screen_width = new_width
        self.screen_height = new_height
        
        # Ensure menu doesn't exceed screen size
        max_w = int(new_width * 0.9)
        max_h = int(new_height * 0.9)
        
        if self.menu_width > max_w:
            self.menu_width = max_w
        if self.menu_height > max_h:
            self.menu_height = max_h
            
        # Ensure it's not too small either (unless screen is tiny)
        self.menu_width = max(min(self.menu_width, max_w), 300)
        self.menu_height = max(min(self.menu_height, max_h), 200)
        
        # Keep position within bounds
        current_x = int(self.pos_x * new_width)
        current_y = int(self.pos_y * new_height)
        
        # Clamp to screen
        current_x = max(0, min(current_x, new_width - self.menu_width))
        current_y = max(0, min(current_y, new_height - self.menu_height))
        
        self.pos_x = current_x / new_width
        self.pos_y = current_y / new_height
        
        # Re-calculate button positions just in case
        print(f"📱 Resized UI to: {new_width}x{new_height}")

    def _focal_length(self):
        """Compute focal length matching the GPU renderer's perspective projection.
        Must use screen_height to match CameraTracker's vertical FOV mapping.
        """
        import math
        fov_rad = math.radians(CAMERA_FOV)
        return self.screen_height / (2.0 * math.tan(fov_rad / 2.0))

    def get_virtual_view_matrix(self):
        """
        Calculate View Matrix for Virtual Orbit Camera
        Converts Spherical Coordinates (Yaw, Pitch, Dist) to Cartesian Eye Position
        Target is (0,0,0) (World Center)
        """
        from pyrr import Matrix44, Vector3
        import math
        
        # Convert to radians
        yaw_rad = math.radians(self.virtual_cam_yaw)
        pitch_rad = math.radians(self.virtual_cam_pitch)
        
        # Clamp pitch to avoid gimbal lock (vertical flip)
        pitch_rad = max(math.radians(-89), min(math.radians(89), pitch_rad))
        
        # Spherical to Cartesian
        # Y is Up in our world (OpenGL convention often uses Y-up)
        # But wait, our objects are on XZ plane usually?
        # Let's assume typical OpenGL: Y is UP.
        # x = r * cos(pitch) * sin(yaw)
        # y = r * sin(pitch)
        # z = r * cos(pitch) * cos(yaw)
        
        # Distance logic
        r = self.virtual_cam_dist
        
        eye_x = self.virtual_cam_target[0] + r * math.cos(pitch_rad) * math.sin(yaw_rad)
        eye_y = self.virtual_cam_target[1] + r * math.sin(pitch_rad)
        eye_z = self.virtual_cam_target[2] + r * math.cos(pitch_rad) * math.cos(yaw_rad)
        
        eye = Vector3([eye_x, eye_y, eye_z])
        target = Vector3(self.virtual_cam_target) # Look at target
        up = Vector3([0.0, 1.0, 0.0])
        
        # Create LookAt Matrix
        view = Matrix44.look_at(eye, target, up)
            # pyrr stores matrices column-major; transpose to row-major for numpy @ operations
        return np.array(view, dtype=np.float32).T

    def _create_3d_object_from_card(self, card):
        """Factory to create Object3D from ItemCard"""
        mesh = None
        material_groups = None
        
        # Check cache first
        cache_key = card.model_path if card.model_path else None
        if cache_key and cache_key in self._mesh_cache:
            cached_mesh, cached_mat_groups = self._mesh_cache[cache_key]
            # Clone the mesh (share VBO data but allow independent VAO)
            mesh = Mesh(
                cached_mesh.vertices.copy(),
                cached_mesh.faces.copy(),
                normals=cached_mesh.normals.copy() if cached_mesh.normals is not None else None,
                indices=cached_mesh.indices.copy() if cached_mesh.indices is not None else None,
                material_groups=[tuple(g) for g in cached_mat_groups] if cached_mat_groups else None
            )
            print(f"⚡ Using cached model: {cache_key}")
        
        # Try loading GLB first
        elif card.model_path and card.model_path.lower().endswith('.glb') and os.path.exists(card.model_path):
            print(f"📥 Loading GLB model: {card.model_path}")
            try:
                v, n, idx, mg = load_glb(card.model_path)
                if v is not None and len(v) > 0:
                    faces = idx.reshape(-1, 3).astype(np.int32)
                    mat_groups = [(g.material_id, g.start_index, g.index_count, g.color) for g in mg]
                    mesh = Mesh(v, faces, normals=n, indices=idx.astype(np.int32), material_groups=mat_groups)
                    # Cache it
                    self._mesh_cache[card.model_path] = (mesh, mat_groups)
                else:
                    print(f"⚠️ GLB loaded but empty: {card.model_path}")
            except Exception as e:
                print(f"❌ Failed to create Mesh from GLB {card.model_path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Try loading OBJ
        elif card.model_path and os.path.exists(card.model_path):
            print(f"📥 Loading OBJ model: {card.model_path}")
            try:
                # Updated load_obj returns 5 values + enhanced material groups
                v, n, t, i, mg = load_obj(card.model_path)
                if v is not None and len(v) > 0:
                    faces = i.reshape(-1, 3)
                    
                    # Convert obj_loader groups (5 items) to Mesh groups (5 items)
                    # obj_loader: (name, start, count, color, texture_path)
                    # mesh: same
                    final_mg = []
                    if mg:
                        base_dir = os.path.dirname(card.model_path)
                        for mat_name, start, count, color, tex_path in mg:
                            
                            # Resolve texture path if present
                            final_tex_path = None
                            if tex_path:
                                # Check if absolute
                                if os.path.exists(tex_path):
                                    final_tex_path = tex_path
                                else:
                                    # Check relative to OBJ
                                    potential = os.path.join(base_dir, tex_path)
                                    if os.path.exists(potential):
                                        final_tex_path = potential
                                        
                            final_mg.append((mat_name, start, count, color, final_tex_path))
                            
                    mesh = Mesh(v, faces, normals=n, tex_coords=t, indices=i, material_groups=final_mg)
                    # Cache it
                    self._mesh_cache[card.model_path] = (mesh, final_mg)
                else:
                    print(f"⚠️ Model loaded but empty: {card.model_path}")
            except Exception as e:
                print(f"❌ Failed to create Mesh from {card.model_path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Furniture mesh (procedural multi-part models)
        if mesh is None and card.mesh_type == "furniture_mesh" and card.model_path in furniture_meshes.MESH_REGISTRY:
            mesh = furniture_meshes.MESH_REGISTRY[card.model_path]()
        
        # Fallback to primitives
        if mesh is None:
            if card.mesh_type == "cube":
                mesh = primitive_meshes.create_cube(size=1.0)
            elif card.mesh_type == "pyramid":
                mesh = primitive_meshes.create_pyramid(size=1.0, height=1.5)
            else:
                mesh = primitive_meshes.create_cube(size=1.0)
            
                mesh = primitive_meshes.create_cube(size=1.0)
            
        obj = Object3D(mesh, category=card.category)
        obj.name = card.name
        # Default color by category
        if "wall" in card.id: obj.color = (150, 150, 160)
        elif "sofa" in card.id: obj.color = (100, 100, 200)
        elif "plant" in card.id: obj.color = (50, 180, 50)
        
        # Populate material overrides
        if mesh.material_groups:
            # Handle both 4-item and 5-item tuples for compatibility
            for group in mesh.material_groups:
                if len(group) == 5:
                    mat_id, _, _, mat_color, _ = group
                else:
                    mat_id, _, _, mat_color = group
                obj.materials[mat_id] = mat_color
        
        # Auto-scale: normalize largest bbox axis to TARGET_OBJECT_SIZE
        from config import TARGET_OBJECT_SIZE
        bbox_extent = mesh.bbox_max - mesh.bbox_min
        max_extent = float(np.max(bbox_extent))
        if max_extent > 1e-6:
            uniform_scale = TARGET_OBJECT_SIZE / max_extent
        else:
            uniform_scale = TARGET_OBJECT_SIZE  # degenerate mesh fallback
        obj.scale = np.array([uniform_scale, uniform_scale, uniform_scale])
        # Fix: Rotate 180 degrees to face the camera (which looks down +Z)
        # Also apply Camera Pose if available to spawn in front of user
        
        # Default local spawn (500 units in front)
        # Camera looks down -Z in some conventions, or +Z in others.
        # Based on CameraTracker, we look down -Z. So spawn at (0, 0, -500).
        # Wait, CameraTracker initial view looks at -Z.
        spawn_pos = np.array([0.0, 0.0, -500.0])
        
        if hasattr(self, 'last_camera_pose') and self.last_camera_pose:
            R_cam_world, t_cam_world = self.last_camera_pose
            # P_world = R * P_local + t
            # spawn_pos is (3,) -> (3,1)
            p_local = spawn_pos.reshape(3, 1)
            p_world = R_cam_world @ p_local + t_cam_world
            obj.position = p_world.flatten()
            
            # Also align rotation? Maybe face the camera?
            # For now, just keep global alignment or align with camera rotation.
            # If we want it to face the user, we might need to apply R_cam_world to orientation too.
            pass
        else:
            obj.position = spawn_pos # Fallback to origin-relative
        
        # Face the object toward the camera (which looks down -Z)
        obj.rotation = np.array([0.0, 180.0, 0.0])
        return obj



    def _start_resize(self, x, y):
        """Start resizing - supports edges and corners"""
        edge_type = self._get_edge_resize_type(x, y)
        
        if edge_type:
            self.state = MenuState.RESIZING
            self.resize_start_x = x
            self.resize_start_y = y
            self.resize_start_w = self.menu_width
            self.resize_start_h = self.menu_height
            self.resize_edge_type = edge_type  # Store which edge
            return True
        
        return False

     
    def _get_edge_resize_type(self, x, y):
        """
        Detect which edge/corner user is hovering over
        Returns: ('left', 'right', 'top', 'bottom', 'corner', or None)
        """
        if self.state not in [MenuState.OPEN, MenuState.RESIZING]:
            return None
        
        menu_x, menu_y = self.get_screen_position()
        menu_w = self.menu_width
        menu_h = self.menu_height
        
        edge_threshold = 15  # Pixels from edge to detect
        
        # Check if near edges
        near_left = abs(x - menu_x) < edge_threshold
        near_right = abs(x - (menu_x + menu_w)) < edge_threshold
        near_top = abs(y - menu_y) < edge_threshold
        near_bottom = abs(y - (menu_y + menu_h)) < edge_threshold
        
        # Check if within menu bounds (vertically/horizontally)
        in_vertical_range = menu_y < y < menu_y + menu_h
        in_horizontal_range = menu_x < x < menu_x + menu_w
        
        # Detect corners first (priority)
        if near_left and near_top:
            return 'corner_tl'  # Top-left
        if near_right and near_top:
            return 'corner_tr'  # Top-right
        if near_left and near_bottom:
            return 'corner_bl'  # Bottom-left
        if near_right and near_bottom:
            return 'corner_br'  # Bottom-right
        
        # Then detect edges
        if near_left and in_vertical_range:
            return 'left'
        if near_right and in_vertical_range:
            return 'right'
        if near_top and in_horizontal_range:
            return 'top'
        if near_bottom and in_horizontal_range:
            return 'bottom'
        
        return None

    def _calculate_snap(self, held_obj, other_objects, threshold=50.0):
        """
        Face-priority snap system.
        Priority: 1) Face-to-Face  2) Edge-to-Edge  3) Vertex-to-Vertex
        Returns: (delta_vec, snap_type, face_a, face_b, other_obj) or (None, ...)
        """
        best_dist = float('inf')
        best_delta = None
        best_type = None
        best_face_a = ""
        best_face_b = ""
        best_other = None
        
        cm = self.constraint_manager
        my_group_ids = cm.get_all_group_member_ids(held_obj)
        
        for other in other_objects:
            if other is held_obj: continue
            if other.obj_id in my_group_ids: continue
            
            # Skip far objects
            if np.linalg.norm(held_obj.position - other.position) > 500:
                continue
            
            # === TIER 1: Face-to-Face Snap ===
            held_faces = held_obj.get_faces()
            other_faces = other.get_faces()
            
            for h_name, h_center, h_normal in held_faces:
                for o_name, o_center, o_normal in other_faces:
                    # Faces must be facing each other (opposite normals)
                    dot = np.dot(h_normal, o_normal)
                    if dot > -0.7:  # Not facing each other
                        continue
                    
                    # Distance between face centers
                    dist = np.linalg.norm(h_center - o_center)
                    
                    # Face snap gets a bonus (lower effective distance)
                    effective_dist = dist * 0.5  # Prioritize faces
                    
                    if dist < threshold and effective_dist < best_dist:
                        # Snap held face center to other face center
                        best_delta = o_center - h_center
                        best_dist = effective_dist
                        best_type = "FACE_ALIGN"
                        best_face_a = h_name
                        best_face_b = o_name
                        best_other = other
            
            # === TIER 2: Edge Midpoint Snap ===
            held_edges = held_obj.get_edge_midpoints()
            other_edges = other.get_edge_midpoints()
            
            for he in held_edges:
                for oe in other_edges:
                    dist = np.linalg.norm(he - oe)
                    effective_dist = dist * 0.7  # Slight priority over vertex
                    
                    if dist < threshold * 0.8 and effective_dist < best_dist:
                        best_delta = oe - he
                        best_dist = effective_dist
                        best_type = "EDGE_ALIGN"
                        best_face_a = ""
                        best_face_b = ""
                        best_other = other
            
            # === TIER 3: Vertex-to-Vertex Snap ===
            held_corners = held_obj.get_world_corners()
            other_corners = other.get_world_corners()
            
            for hc in held_corners:
                for oc in other_corners:
                    dist = np.linalg.norm(hc - oc)
                    if dist < threshold * 0.6 and dist < best_dist:
                        best_delta = oc - hc
                        best_dist = dist
                        best_type = "VERTEX"
                        best_face_a = ""
                        best_face_b = ""
                        best_other = other
        
        return best_delta, best_type, best_face_a, best_face_b, best_other

    def toggle(self):
        """Minimize/Expand menu"""
        if self.state == MenuState.MINIMIZED:
            self.state = MenuState.OPEN
            print("📖 Menu expanded")
        else:
            self.state = MenuState.MINIMIZED
            print("📘 Menu minimized")
    
    # Helper for hit testing
    def _raycast_for_object(self, px, py, view_matrix):
        """Returns True if the pointer is over any placed object."""
        f = self._focal_length()
        for i in range(len(self.placed_objects)-1, -1, -1):
            obj = self.placed_objects[i]
            if obj.is_pointer_inside(px, py, self.screen_width, self.screen_height, f, view_matrix=view_matrix):
                return True
        return False

    def begin_frame(self, num_hands):
        """Called once per frame before update_hand calls.
        Handles cleanup when hands disappear."""
        if num_hands == 0:
            # No hands visible - release any grabs
            if self.held_object:
                print(f"💨 Lost hand — released: {self.held_object.name}")
                self.held_object = None
                self.holding_hand_id = -1
            # Reset drag/resize state
            if self.state in [MenuState.DRAGGING, MenuState.RESIZING]:
                self.state = MenuState.OPEN
                self.is_selected = False
                self.resize_handle = ResizeHandle.NONE
    
    
    def update(self, pointer_x, pointer_y, gesture_data, camera_pose=None, view_matrix=None):
        """
        Process unified gesture input per frame.
        """
        if camera_pose:
            self.last_camera_pose = camera_pose
            
        self.pointer_x = pointer_x
        self.pointer_y = pointer_y
        
        # Update Timers
        now = time.time()
        dt = now - getattr(self, 'last_update_time', now)
        self.last_update_time = now
        
        # Animate Virtual Mode Transition
        target = 1.0 if self.settings.get('virtual_mode', False) else 0.0
        speed = 2.0 
        if self.virtual_mode_progress < target:
            self.virtual_mode_progress = min(target, self.virtual_mode_progress + speed * dt)
        elif self.virtual_mode_progress > target:
            self.virtual_mode_progress = max(target, self.virtual_mode_progress - speed * dt)
            
        if not gesture_data.get('hand_detected', False):
             return

        # Calculate focal length for 3D hits
        f = self._focal_length() 
        
        # Unpack
        px, py = pointer_x, pointer_y
        state = gesture_data.get('state', 'NONE')
        # Map pinch states
        is_pinching = (state in ['START', 'HOLD'])
        is_fist = gesture_data.get('is_fist', False)
        pinch_dist = gesture_data.get('pinch_distance', 0)
        
        # ... (Priority 1 Logic skipped here for brevity, assumed unchanged) ...
        
        # ZOOM LOGIC (2 Hands)
        hand_count = gesture_data.get('hand_count', 1)
        hand_distance = gesture_data.get('hand_distance', 0)
        
        # Check if hitting object to distinguish Zoom (Background) vs Resize (Object)
        is_hitting_object_zoom = self._raycast_for_object(px, py, view_matrix)
        
        if self.settings.get('virtual_mode', False) and hand_count == 2 and gesture_data.get('both_pinching', False) and not is_hitting_object_zoom:
             # If pinching with both hands in EMPTY SPACE -> Zoom
             if hand_distance > 10: # Noise threshold
                 if self._initial_hand_distance == 0:
                     self._initial_hand_distance = hand_distance
                     self._initial_cam_dist = self.virtual_cam_dist
                 else:
                     # Calculate scale factor
                     scale = self._initial_hand_distance / hand_distance
                     # Apply to distance
                     new_dist = self._initial_cam_dist * scale
                     self.virtual_cam_dist = max(100.0, min(2000.0, new_dist))
             else:
                 self._initial_hand_distance = 0
        else:
             # Only reset if NOT doing object scaling (otherwise clobbers baseline)
             if self.selected_object_index == -1:
                 self._initial_hand_distance = 0

        # ROTATION LOGIC (1 Hand)
        if self.settings.get('virtual_mode', False) and not self.held_object and hand_count == 1 and self.selected_object_index == -1:
             # Check for start condition
             # PRIORITY: Menu > Hits/Gizmos > Background (Rotation)
             is_hitting_object = self._raycast_for_object(px, py, view_matrix)
             
             can_start_rotate = (state != "NONE" and 
                                 not self._is_over_menu(px, py) and 
                                 not self._is_over_minimized_button(px, py) and
                                 not is_hitting_object) # Don't rotate if clicking object
             
             # If already rotating, check if THIS hand is the one rotating
             is_rotating_hand = (self._rotating and 
                                 gesture_data.get('hand_idx', -1) == self._rotating_hand_index)
             
             if can_start_rotate or is_rotating_hand:
                 # Check delay to start rotating
                 elapsed = time.time() - self._tap_start_time
                 if elapsed > 0.1: # Small threshold
                     
                     # START ROTATION
                     if not self._rotating and state in ['START', 'HOLD']:
                          self._rotating = True
                          self._rotating_hand_index = gesture_data.get('hand_idx', -1)
                          self.manipulation_start_x = px
                          self.manipulation_start_y = py
                          self._cam_start_yaw = self.virtual_cam_yaw
                          self._cam_start_pitch = self.virtual_cam_pitch
                     
                     # CONTINUE ROTATION
                     if self._rotating and is_rotating_hand:
                         if state in ['START', 'HOLD']:
                             # Delta
                             dx = px - self.manipulation_start_x
                             dy = py - self.manipulation_start_y
                             
                             # Sensitivity from Settings
                             sens_x = self.settings.get('sensitivity_x', 0.5) * 0.5
                             sens_y = self.settings.get('sensitivity_y', 0.5) * 0.5
                             
                             self.virtual_cam_yaw = self._cam_start_yaw - dx * sens_x
                             self.virtual_cam_pitch = self._cam_start_pitch + dy * sens_y
                             
                             # Clamp Pitch
                             self.virtual_cam_pitch = max(-89.0, min(89.0, self.virtual_cam_pitch))
                         else:
                             # RELEASE -> STOP ROTATION
                             self._rotating = False
                             self._rotating_hand_index = -1
                             print("🔄 Stopped Camera Rotation")
                         
                         # Let release fall through to selection logic
                         pass

        # ============================================================
        # PRIORITY 1: Continue active menu drag/resize (highest priority)
        # ============================================================
        
        # FIST DRAG continuation for menu
        if self.state == MenuState.DRAGGING:
            if is_fist or is_pinching:
                self._handle_drag(px, py)
                return
            else:
                self.state = MenuState.OPEN
                self.is_selected = False
                return
        
        # PINCH RESIZE continuation for menu
        if self.state == MenuState.RESIZING:
            if is_pinching:
                self._handle_resize(px, py)
                return
            else:
                self.state = MenuState.OPEN
                self.is_selected = False
                self.resize_handle = ResizeHandle.NONE
                return
        
        hand_count = gesture_data.get('hand_count', 1)
        hand_distance = gesture_data.get('hand_distance', 0.0)
        TAP_MAX_DURATION = 0.3
        
        # Helper: find closest object to pointer
        def _find_closest_object():
            f = self._focal_length()
            best_idx = -1
            min_dist_to_center = float('inf')
            
            # Use View Matrix if available for accurate projection
            if view_matrix is None:
                from pyrr import Matrix44 as M44
                vm = np.array(M44.from_translation([0.0, 0.0, -500.0]), dtype=np.float32)
            else:
                vm = np.array(view_matrix, dtype=np.float32)
            
            for i, obj in enumerate(self.placed_objects):
                # Pass view_matrix to Ray-Cast
                if obj.is_pointer_inside(px, py, self.screen_width, self.screen_height, f, view_matrix=vm):
                    
                    # Project Center to Screen for Distance Check
                    # P_cam = V * P_world
                    p_world = np.array([obj.position[0], obj.position[1], obj.position[2], 1.0])
                    p_cam = vm @ p_world
                    
                    x, y, z = p_cam[0], p_cam[1], p_cam[2]
                    
                    # ✅ Z-Clipping: Reject objects behind camera
                    if z > -0.1:
                        continue
                    
                    # Project
                    cx, cy = self.screen_width / 2, self.screen_height / 2
                    if abs(z) > 0.001:
                        sx = cx + (x / -z) * f
                        sy = cy - (y / -z) * f
                        
                        # ✅ Screen Bounds Check (Roughly)
                        # Only consider objects whose center is somewhat near the screen
                        # (Prevents selecting things remotely far to the side)
                        if not (-200 < sx < self.screen_width + 200 and -200 < sy < self.screen_height + 200):
                             continue
                             
                        dist = np.hypot(px - sx, py - sy)
                    else:
                        dist = float('inf')
                    
                    if dist < min_dist_to_center:
                        min_dist_to_center = dist
                        best_idx = i
            return best_idx, min_dist_to_center

        # ── 4. FIST GRAB (pointer must be on the object) ──
        if is_fist and not self.held_object and not self._is_over_menu(px, py):
            target, dist = _find_closest_object()
            
            # User request: "object is comming to that place it should not work"
            # So we enforce that the pointer must be reasonably close to the object's center
            # even if RayCast says "Inside" (which can happen if camera is INSIDE the object box)
            if target != -1 and target < len(self.placed_objects) and dist < 150:
                self.held_object = self.placed_objects.pop(target)
                self.holding_hand_id = gesture_data.get('active_hand_id', -1)
                print(f"✊ Grabbed 3D: {self.held_object.name} (Hand {self.holding_hand_id}) at {self.held_object.position}")
                self.selected_object_index = -1
                return
            
            # If we are here, we FISTED but didn't grab an object.
            # If we represent a "Grab in Empty Space", we should DESELECT the current selection.
            if self.selected_object_index != -1 and self.selected_object_index < len(self.placed_objects):
                # Don't deselect if hovering over a toolbar button (Delete, Lock, etc.)
                if (self._is_over_delete_button(px, py) or 
                    self._is_over_lock_button(px, py) or 
                    self._is_over_duplicate_button(px, py)):
                    pass 
                else:
                    print(f"❌ Deselected: {self.placed_objects[self.selected_object_index].name}")
                    self.selected_object_index = -1
        
        # ── 1. MOVE HELD OBJECT ──
        if self.held_object:
            # ✅ CHECK RELEASE FIRST — place at LAST KNOWN position, not current pointer
            # This prevents teleporting when the pointer jumps on the release frame
            if not is_fist and not is_pinching:
                # Re-add to list at its CURRENT (last-frame) position
                self.placed_objects.append(self.held_object)
                print(f"📍 Placed 3D: {self.held_object.name} at {self.held_object.position}")
                
                # AUTO-LOCK: If snapped, create constraint
                if self._last_snap_info is not None:
                    _, snap_type, face_a, face_b, snap_other = self._last_snap_info
                    if snap_other is not None:
                        gid = self.constraint_manager.create_snap_lock(
                            self.held_object, snap_other, face_a, face_b,
                            all_objects=self.placed_objects
                        )
                        print(f"🧲 Auto-locked {self.held_object.name} ↔ {snap_other.name} ({snap_type}, Group {gid})")
                
                self._last_snap_info = None
                self.held_object = None
                self.holding_hand_id = -1
                return
            
            # Still holding — update position
            f = self._focal_length()
            cx, cy = self.screen_width / 2, self.screen_height / 2
            
            # === VIRTUAL MODE: Unproject through inverse view matrix ===
            if self.settings.get('virtual_mode', False) and view_matrix is not None:
                vm = np.array(view_matrix, dtype=np.float32)
                
                # Ray direction in camera space
                dir_x = (px - cx) / f
                dir_y = -(py - cy) / f
                dir_z = -1.0
                ray_dir_cam = np.array([dir_x, dir_y, dir_z], dtype=np.float32)
                ray_dir_cam /= np.linalg.norm(ray_dir_cam)
                
                # Camera position in world space (from inverse view matrix)
                try:
                    inv_vm = np.linalg.inv(vm)
                except np.linalg.LinAlgError:
                    inv_vm = np.eye(4, dtype=np.float32)
                
                cam_pos_world = inv_vm[:3, 3]
                
                # Transform ray direction to world space (rotation only)
                ray_dir_world = inv_vm[:3, :3] @ ray_dir_cam
                ray_dir_world /= np.linalg.norm(ray_dir_world)
                
                # Find the point on the ray at the same distance from camera
                # as the object's current position
                obj_to_cam = self.held_object.position - cam_pos_world
                current_dist = np.linalg.norm(obj_to_cam)
                if current_dist < 1.0:
                    current_dist = 500.0
                
                # Project along ray to that distance
                new_pos = cam_pos_world + ray_dir_world * current_dist
                
                new_x = new_pos[0]
                new_y = new_pos[1]
                z_depth = new_pos[2]
            else:
                # === AR MODE: Simple screen-to-world ===
                # Objects are at negative z (e.g. -500 = in front of camera)
                # Use absolute depth for projection math, keep original z for position
                z_depth = self.held_object.position[2]  # preserve original z
                proj_depth = abs(z_depth)
                if proj_depth < 100: proj_depth = 500.0  # Fallback for near-zero depth
                
                # Proposed position (use positive proj_depth for correct direction)
                new_x = (px - cx) * proj_depth / f
                new_y = (py - cy) * proj_depth / f
            
            # --- GRID SNAP ---
            if self.settings.get('grid_snap', False):
                new_x = round(new_x / GRID_SIZE) * GRID_SIZE
                new_y = round(new_y / GRID_SIZE) * GRID_SIZE
                z_depth = round(z_depth / GRID_SIZE) * GRID_SIZE
            
            # --- MAGNETIC BORDER SNAP ---
            # Calculate world limits at this depth (use absolute depth for correct snap bounds)
            snap_depth = abs(z_depth) if abs(z_depth) >= 100 else 500.0
            world_w = (self.screen_width / 2) * snap_depth / f
            world_h = (self.screen_height / 2) * snap_depth / f
            
            # Convert border threshold from pixels to world units at this depth
            border_thresh_world = MAGNETIC_BORDER_PX * (snap_depth / f)
            
            # Snap to left/right
            if abs(new_x - (-world_w)) < border_thresh_world: new_x = -world_w
            elif abs(new_x - world_w) < border_thresh_world: new_x = world_w
            
            # Snap to top/bottom
            if abs(new_y - (-world_h)) < border_thresh_world: new_y = -world_h
            elif abs(new_y - world_h) < border_thresh_world: new_y = world_h
            
            # ✅ PER-FRAME JUMP CAP — prevent teleporting from pointer glitches
            MAX_MOVE_PER_FRAME = 150.0  # world units
            move_vec = np.array([new_x - self.held_object.position[0],
                                 new_y - self.held_object.position[1],
                                 z_depth - self.held_object.position[2]], dtype=np.float32)
            move_mag = np.linalg.norm(move_vec)
            if move_mag > MAX_MOVE_PER_FRAME:
                move_vec = move_vec / move_mag * MAX_MOVE_PER_FRAME
                new_x = self.held_object.position[0] + move_vec[0]
                new_y = self.held_object.position[1] + move_vec[1]
                z_depth = self.held_object.position[2] + move_vec[2]
            
            # --- AUTO SNAP LOGIC (Face-Priority) ---
            delta_pos = np.array([new_x - self.held_object.position[0],
                                  new_y - self.held_object.position[1],
                                  z_depth - self.held_object.position[2]], dtype=np.float32)

            self.held_object.position[0] = new_x
            self.held_object.position[1] = new_y
            self.held_object.position[2] = z_depth
            
            self._last_snap_info = None  # Reset each frame
            
            if self.settings.get('auto_snap', True):
                snap_delta, snap_type, face_a, face_b, snap_other = self._calculate_snap(
                    self.held_object, self.placed_objects, threshold=50.0
                )
                
                if snap_delta is not None:
                    self.held_object.position += snap_delta
                    delta_pos += snap_delta
                    self._last_snap_info = (self.held_object.position.copy(), snap_type, face_a, face_b, snap_other)
            
            # Move Group Mates via ConstraintManager (Matrix Update)
            self.constraint_manager.update_children_transforms(self.held_object, self.placed_objects)
            return
        
        # ── 2. PINCH STATE TRACKING ──
        # On pinch START: record time (but don't reset if we're rotating)
        if state == "START" and not self._rotating:
            self._tap_start_time = time.time()
            self._tap_start_pos = (px, py)
            self._tap_consumed = False
        
        # ── 2a. COLOR SWATCH HIT-TEST ──
        # Check if tap hit a color swatch (from palette drawn by main.py)
        swatch_positions = getattr(self, '_swatch_positions', [])
        if state == "RELEASE" and self.selected_object_index != -1 and swatch_positions:
            elapsed = time.time() - self._tap_start_time
            if elapsed < 0.5:  # Quick or medium tap on swatch
                for sx, sy, color_rgb in swatch_positions:
                    dist = np.hypot(px - sx, py - sy)
                    if dist < 25:  # Hit a swatch
                        obj = self.placed_objects[self.selected_object_index]
                        obj.color = color_rgb
                        # Also update all material overrides
                        for mat_id in list(obj.materials.keys()):
                            obj.materials[mat_id] = color_rgb
                        # Force GPU to re-render with new color
                        print(f"🎨 Color changed: {obj.name} → RGB{color_rgb}")
                        return
        
        # On pinch RELEASE: check gesture type
        SELECTION_COOLDOWN = 0.5  # seconds between selection changes
        DESELECT_HOLD_MIN = 0.3   # Must hold at least this long to deselect
        DESELECT_HOLD_MAX = 1.0   # But not longer than this
        ROTATION_MOVE_THRESHOLD = 20  # px movement needed to count as rotation vs hold-still
        
        if state == "RELEASE" and not self._is_over_menu(px, py):
            elapsed = time.time() - self._tap_start_time
            time_since_last_select = time.time() - getattr(self, '_last_selection_time', 0)
            
            # Calculate how much the pointer moved since pinch start
            move_dist = np.hypot(px - self._tap_start_pos[0], py - self._tap_start_pos[1])
            was_rotating = self._rotating
            
            # Reset rotating on release
            self._rotating = False
            
            if time_since_last_select > SELECTION_COOLDOWN:
                candidate, candidate_dist = _find_closest_object()
                
                # move_dist already calculated above (line ~1023)
                # Reuse it for the tap-vs-drag check
                
                # TAP — select (only if moved less than 30px and wasn't rotating camera)
                if candidate != -1 and elapsed < TAP_MAX_DURATION and move_dist < 30 and not was_rotating:
                    if self.selected_object_index != candidate:
                        self.selected_object_index = candidate
                        obj = self.placed_objects[candidate]
                        print(f"✅ Selected: {obj.name}")
                        self.initial_rotation_y = obj.rotation[1]
                        self.initial_rotation_x = obj.rotation[0]
                        self.initial_scale = obj.scale[0]
                        self._initial_hand_distance = 0
                        self._last_selection_time = time.time()
                    return
                
                # Fall through to menu interaction if no object selected/tapped
                pass
            
            # Continue to priority 3 (Menu checks)
        
        # ============================================================
        # PRIORITY 3: Menu Interaction (start new drag/resize/click)
        # ============================================================
        
        # SCROLLING (BOTH PANELS)
        menu_x, menu_y = self.get_screen_position()
        sidebar_w = int(self.menu_width * self.sidebar_ratio)
        content_start_x = menu_x + sidebar_w
        
        if (state == "START" or state == "HOLD"):
            if hasattr(self, 'last_pointer_pos'):
                lpx, lpy = self.last_pointer_pos
                dy = py - lpy
                
                # Check which panel we are scrolling
                is_sidebar = (menu_x <= px < content_start_x)
                is_content = (content_start_x <= px <= menu_x + self.menu_width)
                
                # Only scroll if inside menu bounds vertically too
                if menu_y <= py <= menu_y + self.menu_height:
                    
                    if is_sidebar:
                        # Scroll Sidebar
                        old_s = self.sidebar_scroll_y
                        self.sidebar_scroll_y -= dy * 1.5
                        self.sidebar_scroll_y = max(0, min(self.sidebar_scroll_y, self.max_sidebar_scroll))
                        # Debug
                        if abs(old_s - self.sidebar_scroll_y) > 0.1:
                            print(f"📜 Sidebar Scroll: {self.sidebar_scroll_y:.1f} (dy={dy:.1f})")
                        
                    elif is_content:
                        # Scroll Content
                        old_c = self.content_scroll_y
                        self.content_scroll_y -= dy * 1.5
                        self.content_scroll_y = max(0, min(self.content_scroll_y, self.max_content_scroll))
                        # Debug
                        if abs(old_c - self.content_scroll_y) > 0.1:
                            print(f"📜 Content Scroll: {self.content_scroll_y:.1f} (dy={dy:.1f})")

        # At VERY END of update:
        self.last_pointer_pos = (px, py)

        if self.selected_object_index != -1 and self.selected_object_index < len(self.placed_objects):
            obj = self.placed_objects[self.selected_object_index]
            
            # TWO-HAND SCALING (requires both hands pinching)
            both_pinching = gesture_data.get('both_pinching', False)
            if both_pinching and hand_count >= 2 and hand_distance > 10:
                if self._initial_hand_distance < 10:
                    self._initial_hand_distance = hand_distance
                    self.initial_scale = obj.scale[0]
                else:
                    # Amplify scale ratio for more sensitivity
                    ratio = hand_distance / self._initial_hand_distance
                    amplified = 1.0 + (ratio - 1.0) * 2.0  # 2x sensitivity
                    new_scale = max(10.0, min(self.initial_scale * amplified, 500.0))
                    
                    # Apply to Object OR Group
                    scale_vec = np.array([new_scale, new_scale, new_scale])
                    obj.scale = scale_vec
                    
                    # Propagate scale to group via ConstraintManager
                    self.constraint_manager.update_children_transforms(obj, self.placed_objects)
            else:
                self._initial_hand_distance = 0
            
            # PINCH-HOLD ROTATION (requires movement from start position)
            if is_pinching and not gesture_data.get('both_pinching', False):
                elapsed = time.time() - self._tap_start_time
                move_dist = np.hypot(px - self._tap_start_pos[0], py - self._tap_start_pos[1])
                
                if (elapsed >= TAP_MAX_DURATION and move_dist >= ROTATION_MOVE_THRESHOLD) or self._rotating:
                    if not self._rotating:
                        # First frame of rotation — record baselines
                        self._rotating = True
                        self.manipulation_start_x = px
                        self.manipulation_start_y = py
                        self.initial_rotation_y = obj.rotation[1]
                        self.initial_rotation_x = obj.rotation[0]
                        self._last_rotation_px = px
                        self._last_rotation_py = py
                    
                    # ✅ JUMP DETECTION — if pointer jumps too far, reset baseline
                    frame_jump = np.hypot(px - self._last_rotation_px, py - self._last_rotation_py)
                    if frame_jump > 100:
                        # Reset baseline to current position to absorb the jump
                        self.manipulation_start_x = px
                        self.manipulation_start_y = py
                        self.initial_rotation_y = obj.rotation[1]
                        self.initial_rotation_x = obj.rotation[0]
                    self._last_rotation_px = px
                    self._last_rotation_py = py
                    
                    delta_x = px - self.manipulation_start_x
                    delta_y = py - self.manipulation_start_y
                    
                    # Apply Sensitivity Mutipliers
                    sens_x = self.settings.get('sensitivity_x', 1.0)
                    sens_y = self.settings.get('sensitivity_y', 1.0)
                    
                    new_ry = self.initial_rotation_y + (delta_x / 100.0) * 90.0 * sens_x
                    new_rx = self.initial_rotation_x + (delta_y / 100.0) * 90.0 * sens_y
                    
                    # Apply to object
                    obj.rotation[1] = new_ry
                    obj.rotation[0] = new_rx
                    
                    # Propagate rotation around group pivot via ConstraintManager
                    self.constraint_manager.update_children_transforms(obj, self.placed_objects)
                    return
        
        # FIST GRAB moved to top of update loop to handle tuple return from _find_closest_object
        
        # VIRTUAL MODE TOGGLE (Floating Button)
        if state == "START":
            if self._is_over_virtual_mode_button(px, py):
                self.settings['virtual_mode'] = not self.settings.get('virtual_mode', False)
                # Auto-center camera on objects when entering virtual mode
                if self.settings['virtual_mode'] and self.placed_objects:
                    positions = np.array([obj.position for obj in self.placed_objects])
                    centroid = positions.mean(axis=0)
                    self.virtual_cam_target = centroid.astype(np.float32)
                    self.virtual_cam_dist = 300.0
                    self.virtual_cam_yaw = 0.0
                    self.virtual_cam_pitch = 20.0
                print(f"🌌 Virtual Mode: {'ON' if self.settings['virtual_mode'] else 'OFF'}")
                return

            # 4. Object Toolbar Buttons (Delete, Duplicate, LOCK)
            if self.selected_object_index != -1:
                # DELETE
                if self._is_over_delete_button(px, py):
                    obj = self.placed_objects[self.selected_object_index]
                    
                    # Delete group or single object
                    group = self.constraint_manager.get_group_for_object(obj)
                    if group:
                        group_objs = self.constraint_manager.get_group_objects(group, self.placed_objects)
                        for o in group_objs:
                            if o in self.placed_objects:
                                self.placed_objects.remove(o)
                        # Clean up group from manager
                        if group.group_id in self.constraint_manager.groups:
                            del self.constraint_manager.groups[group.group_id]
                        print(f"🗑️ Deleted locked group ({len(group_objs)} objects)")
                    else:
                        self.placed_objects.remove(obj)
                        print(f"🗑️ Deleted object: {obj.name}")
                        
                    self.selected_object_index = -1
                    self.held_object = None 
                    return
                
                # LOCK / UNLOCK (via ConstraintManager)
                if self._is_over_lock_button(px, py):
                     if time.time() - self.last_click_time > CLICK_COOLDOWN:
                         obj = self.placed_objects[self.selected_object_index]
                         cm = self.constraint_manager
                         
                         if cm.is_locked(obj):
                             # BREAK LOCK — detach this object from its group
                             cm.break_lock(obj, self.placed_objects)
                             print(f"🔓 Unlocked object: {obj.name}")
                         else:
                             # MANUAL LOCK — find closest objects to group with
                             candidates = []
                             for other in self.placed_objects:
                                 if other is obj: continue
                                 dist = np.linalg.norm(obj.position - other.position)
                                 if dist < 200:  # Generous lock radius
                                     candidates.append((dist, other))
                             
                             candidates.sort(key=lambda x: x[0])
                             
                             if candidates:
                                 lock_targets = [obj] + [c[1] for c in candidates[:3]]  # Lock up to 3 nearby
                                 gid = cm.create_manual_lock(lock_targets, all_objects=self.placed_objects)
                                 names = ', '.join(o.name for o in lock_targets)
                                 print(f"🔒 Locked: {names} (Group {gid})")
                             else:
                                 print("⚠️ No objects nearby to lock with!")
                                 
                         self.last_click_time = time.time()
                     return

            # 5. Object Selection (Raycast)
            if not self.held_object and state == "START":
                 # Guard: Don't select objects behind UI buttons
                 if self._is_over_minimized_button(px, py) or self._is_over_virtual_mode_button(px, py):
                     pass  # Don't raycast through UI buttons
                 else:
                     # Raycast
                     clicked_obj_index = -1
                     f = self._focal_length()
                     for i in range(len(self.placed_objects)-1, -1, -1):
                         obj = self.placed_objects[i]
                         # PASS VIEW MATRIX HERE
                         if obj.is_pointer_inside(px, py, self.screen_width, self.screen_height, f, view_matrix=view_matrix):
                             clicked_obj_index = i
                             break
                     
                     if clicked_obj_index != -1:
                         if self.selected_object_index != clicked_obj_index:
                             self.selected_object_index = clicked_obj_index
                             print(f"✅ Selected: {self.placed_objects[clicked_obj_index].name}")
                             self._last_selection_time = time.time()
                     else:
                         # Deselect if clicking empty space
                         if self.selected_object_index != -1 and time.time() - self._last_selection_time > 0.3:
                             self.selected_object_index = -1
                             print("❌ Deselected object")
        
        # FIST initiate drag for menu
        # FIST initiate drag for menu
        if is_fist and self.state == MenuState.OPEN:
             menu_x, menu_y = self.get_screen_position()
             if (menu_x - POINTER_RADIUS <= px <= menu_x + self.menu_width + POINTER_RADIUS and
                 menu_y - POINTER_RADIUS <= py <= menu_y + 60 + POINTER_RADIUS):
                 self.state = MenuState.DRAGGING
                 self.is_selected = True
                 self.drag_offset_x = px - menu_x
                 self.drag_offset_y = py - menu_y
                 return

        # PINCH INTERACTION
        if self.state == MenuState.MINIMIZED:
            self._handle_minimized_button(px, py, state)
            return
        
        if state == "START":
            if self._is_over_menu(px, py):
                # Standard menu clicks
                if self._is_over_minimize_button(px, py):
                    self.toggle()
                    return
                
                if self._is_over_close_button(px, py):
                    self.state = MenuState.MINIMIZED
                    print("❌ Menu closed")
                    return
                
                handle = self._get_resize_handle(px, py)
                if handle != ResizeHandle.NONE:
                    self.state = MenuState.RESIZING
                    self.resize_handle = handle
                    self.resize_start_width = self.menu_width
                    self.resize_start_height = self.menu_height
                    self.drag_offset_x = px
                    self.drag_offset_y = py
                    return
                
                # Drag header (pinch)
                menu_x, menu_y = self.get_screen_position()
                if menu_y - POINTER_RADIUS <= py <= menu_y + 60 + POINTER_RADIUS:
                    self.state = MenuState.DRAGGING
                    self.is_selected = True
                    self.drag_offset_x = px - menu_x
                    self.drag_offset_y = py - menu_y
                    return
                
                self._handle_sidebar_click(px, py)
                self._handle_card_click(px, py)
                
        elif state == "HOLD":
            if self.state == MenuState.DRAGGING:
                self._handle_drag(px, py)
            elif self.state == MenuState.RESIZING:
                self._handle_resize(px, py)
                
        elif state == "RELEASE":
            if self.state in [MenuState.DRAGGING, MenuState.RESIZING]:
                self.state = MenuState.OPEN
                self.is_selected = False
                self.resize_handle = ResizeHandle.NONE
    
    def _is_over_menu(self, px, py):
        if self.state not in [MenuState.OPEN, MenuState.DRAGGING, MenuState.RESIZING]:
            return False
        menu_x, menu_y = self.get_screen_position()
        return (menu_x - POINTER_RADIUS <= px <= menu_x + self.menu_width + POINTER_RADIUS and
                menu_y - POINTER_RADIUS <= py <= menu_y + self.menu_height + POINTER_RADIUS)

    def _handle_minimized_button(self, px, py, state):
        """Handle clicks on minimized floating buttons (menu + settings)"""
        btn_x = int(self.minimized_pos_x * self.screen_width)
        btn_y = int(self.minimized_pos_y * self.screen_height)
        btn_size = self.minimized_button_size
        r = POINTER_RADIUS
        
        if state == "START":
            # Menu button (expanded hit area)
            if (btn_x - r <= px <= btn_x + btn_size + r and
                btn_y - r <= py <= btn_y + btn_size + r):
                self.toggle()
                return
            

    
    
    def _is_over_minimize_button(self, px, py):
        """Check if pointer is over minimize button"""
        btn_x, btn_y = self.get_minimized_button_position()
        return (btn_x <= px <= btn_x + self.minimized_button_size and 
                btn_y <= py <= btn_y + self.minimized_button_size)

    def get_settings_button_position(self):
        """Get settings button position (below menu button)"""
        btn_x = int(self.minimized_pos_x * self.screen_width)
        btn_y = int(self.minimized_pos_y * self.screen_height)
        # 15px gap below the main menu button
        settings_y = btn_y + self.minimized_button_size + 15
        return btn_x, settings_y

    def get_virtual_mode_button_position(self):
        """Return (x, y) of the Virtual Mode button"""
        sx, sy = self.get_settings_button_position()
        # Left of settings
        vx = sx - self.minimized_button_size - 15
        vy = sy
        return vx, vy

    def get_import_button_position(self):
        """Return (x, y) of the Import Model button (left of Virtual Mode)"""
        vx, vy = self.get_virtual_mode_button_position()
        ix = vx - self.minimized_button_size - 15
        iy = vy
        return ix, iy

    def _is_over_import_button(self, px, py):
        ix, iy = self.get_import_button_position()
        size = self.minimized_button_size
        return (ix <= px <= ix + size and iy <= py <= iy + size)

    def import_model_from_file(self):
        """Open a file dialog in a background thread (only picks the file path)."""
        if getattr(self, '_import_dialog_open', False):
            print("⚠️ Import dialog already open")
            return

        self._import_dialog_open = True

        def _dialog_thread():
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()  # Hide the root window
                root.attributes('-topmost', True)
                file_path = filedialog.askopenfilename(
                    title="Import 3D Model",
                    filetypes=[
                        ("3D Models", "*.obj *.glb"),
                        ("OBJ Files", "*.obj"),
                        ("GLB Files", "*.glb"),
                        ("All Files", "*.*")
                    ]
                )
                root.destroy()

                if file_path:
                    print(f"📂 Selected model: {file_path}")
                    # Store path for main thread to process
                    self._pending_import_path = file_path
                else:
                    print("ℹ️ Import cancelled")
            except Exception as e:
                print(f"❌ Import dialog error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._import_dialog_open = False

        thread = threading.Thread(target=_dialog_thread, daemon=True)
        thread.start()

    def process_pending_import(self):
        """Called from the main loop to create imported objects on the main thread."""
        file_path = getattr(self, '_pending_import_path', None)
        if file_path is None:
            return

        self._pending_import_path = None  # Clear immediately

        try:
            ext = os.path.splitext(file_path)[1].lower()
            mesh_type = "glb" if ext == ".glb" else "obj"
            name = os.path.splitext(os.path.basename(file_path))[0]

            card = ItemCard(
                id=f"imported_{name}_{int(time.time())}",
                name=name,
                category="imported",
                mesh_type=mesh_type,
                model_path=file_path
            )
            obj = self._create_3d_object_from_card(card)
            self.placed_objects.append(obj)
            print(f"✅ Imported and placed: {name}")
        except Exception as e:
            print(f"❌ Import error: {e}")
            import traceback
            traceback.print_exc()

    def _is_over_virtual_mode_button(self, px, py):
        vx, vy = self.get_virtual_mode_button_position()
        size = self.minimized_button_size
        return (vx <= px <= vx + size and vy <= py <= vy + size)

    def get_delete_button_position(self):
        """Return (x, y) of the Delete button (Below Duplicate)"""
        dup_x, dup_y = self.get_duplicate_button_position()
        spacing = 20
        size = self.minimized_button_size + 10
        
        # Stack BELOW Duplicate
        return (dup_x, dup_y + size + spacing)
        
    def get_duplicate_button_position(self):
        """Return (x, y) of the Duplicate button (Below Virtual Mode)"""
        vx, vy = self.get_virtual_mode_button_position()
        spacing = 20
        size = self.minimized_button_size + 10
        
        # Stack BELOW Virtual Mode
        return (vx, vy + size + spacing)

    def get_lock_button_position(self):
        """Return (x, y) of the Lock button (Below Delete)"""
        dx, dy = self.get_delete_button_position()
        spacing = 20
        size = self.minimized_button_size + 10
        
        # Stack BELOW Delete
        return (dx, dy + size + spacing)

    def _is_over_delete_button(self, x, y):
        dx, dy = self.get_delete_button_position()
        w, h = 80, 80
        return (dx <= x <= dx + w and dy <= y <= dy + h)

    def _is_over_duplicate_button(self, x, y):
        dx, dy = self.get_duplicate_button_position()
        w, h = 80, 80
        return (dx <= x <= dx + w and dy <= y <= dy + h)

    def _is_over_lock_button(self, x, y):
        lx, ly = self.get_lock_button_position()
        w, h = 80, 80
        return (lx <= x <= lx + w and ly <= y <= ly + h)

    def _handle_settings_interaction(self, px, py, state):
        """Handle interactions (click/drag) on settings panel"""
        btn_x, btn_y = self.get_settings_button_position()
        
        # Panel position (must match renderer)
        panel_w = 280
        row_h = 50
        header_h = 50
        panel_x = btn_x - panel_w - 15
        panel_y = btn_y
        
        # Check if interaction is inside the panel area
        keys = list(self.settings.keys())
        for i, key in enumerate(keys):
            row_y = panel_y + header_h + i * row_h
            if (panel_x <= px <= panel_x + panel_w and
                row_y <= py <= row_y + row_h):
                
                # Special Handle for Sliders (Sensitivity)
                if key in ['sensitivity_x', 'sensitivity_y']:
                    # Slider area: from panel_x + 100 to panel_x + panel_w - 20
                    slider_start = panel_x + 100
                    slider_end = panel_x + panel_w - 20
                    slider_w = slider_end - slider_start
                    
                    if state in ["START", "HOLD"]:
                        # Calculate value (0.5 to 4.0)
                        pct = (px - slider_start) / slider_w
                        pct = max(0, min(pct, 1))
                        val = 0.5 + pct * 3.5 # Range: 0.5 to 4.0
                        self.settings[key] = round(val, 1)
                    return True
                
                # Toggle Buttons (Standard)
                elif state == "START":
                    # DEBOUNCE: Prevent rapid toggling if user's pinch flickers
                    if time.time() - self._last_settings_click_time < 0.3:
                        return True
                        
                    self._last_settings_click_time = time.time()
                    self.settings[key] = not self.settings[key]
                    print(f"⚙️ {self.settings_labels[key]}: {'ON' if self.settings[key] else 'OFF'}")
                    return True
        return False
    
    def _is_over_close_button(self, px, py):
        """Check if pointer is over close button (red dot with X)"""
        menu_x, menu_y = self.get_screen_position()
        btn_y = menu_y + 30
        btn_x = menu_x + self.menu_width - 90
        return abs(px - btn_x) < 14 + POINTER_RADIUS and abs(py - btn_y) < 14 + POINTER_RADIUS
    

    

    
    def _get_resize_handle(self, px, py):
        """Detect which resize handle is being grabbed"""
        menu_x, menu_y = self.get_screen_position()
        menu_w = self.menu_width
        menu_h = self.menu_height
        
        handle_size = 20
        
        # Corners
        if (menu_x <= px <= menu_x + handle_size and
            menu_y <= py <= menu_y + handle_size):
            return ResizeHandle.TOP_LEFT
        elif (menu_x + menu_w - handle_size <= px <= menu_x + menu_w and
              menu_y <= py <= menu_y + handle_size):
            return ResizeHandle.TOP_RIGHT
        elif (menu_x <= px <= menu_x + handle_size and
              menu_y + menu_h - handle_size <= py <= menu_y + menu_h):
            return ResizeHandle.BOTTOM_LEFT
        elif (menu_x + menu_w - handle_size <= px <= menu_x + menu_w and
              menu_y + menu_h - handle_size <= py <= menu_y + menu_h):
            return ResizeHandle.BOTTOM_RIGHT
        
        # Edges
        elif (menu_x <= px <= menu_x + handle_size and
              menu_y + handle_size <= py <= menu_y + menu_h - handle_size):
            return ResizeHandle.LEFT
        elif (menu_x + menu_w - handle_size <= px <= menu_x + menu_w and
              menu_y + handle_size <= py <= menu_y + menu_h - handle_size):
            return ResizeHandle.RIGHT
        elif (menu_x + handle_size <= px <= menu_x + menu_w - handle_size and
              menu_y <= py <= menu_y + handle_size):
            return ResizeHandle.TOP
        elif (menu_x + handle_size <= px <= menu_x + menu_w - handle_size and
              menu_y + menu_h - handle_size <= py <= menu_y + menu_h):
            return ResizeHandle.BOTTOM
        
        return ResizeHandle.NONE
    
    def _handle_drag(self, px, py):
        """Move menu"""
        new_x = px - self.drag_offset_x
        new_y = py - self.drag_offset_y
        
        new_x = max(0, min(new_x, self.screen_width - self.menu_width))
        new_y = max(0, min(new_y, self.screen_height - self.menu_height))
        
        self.pos_x = new_x / self.screen_width
        self.pos_y = new_y / self.screen_height
    
    def _handle_resize(self, px, py):
        """Resize menu from corners/edges"""
        menu_x, menu_y = self.get_screen_position()
        
        delta_x = px - self.drag_offset_x
        delta_y = py - self.drag_offset_y
        
        new_w = self.menu_width
        new_h = self.menu_height
        
        if self.resize_handle in [ResizeHandle.RIGHT, ResizeHandle.TOP_RIGHT, ResizeHandle.BOTTOM_RIGHT]:
            new_w = self.resize_start_width + delta_x
        elif self.resize_handle in [ResizeHandle.LEFT, ResizeHandle.TOP_LEFT, ResizeHandle.BOTTOM_LEFT]:
            new_w = self.resize_start_width - delta_x
        
        if self.resize_handle in [ResizeHandle.BOTTOM, ResizeHandle.BOTTOM_LEFT, ResizeHandle.BOTTOM_RIGHT]:
            new_h = self.resize_start_height + delta_y
        elif self.resize_handle in [ResizeHandle.TOP, ResizeHandle.TOP_LEFT, ResizeHandle.TOP_RIGHT]:
            new_h = self.resize_start_height - delta_y
        
        # Clamp to min/max
        new_w = max(self.min_width, min(new_w, self.max_width))
        new_h = max(self.min_height, min(new_h, self.max_height))
        
        self.menu_width = int(new_w)
        self.menu_height = int(new_h)
    
    def _handle_sidebar_click(self, px, py):
        """Handle navigation item clicks"""
        for item in self.nav_items:
            if item.is_clicked(px, py):
                self.selected_category = item.id
                self.content_scroll = 0  # Reset scroll
                print(f"📂 Selected: {item.label}")
                break
    

    def _handle_card_click(self, px, py):
        """Handle item card clicks"""
        # Debounce check
        if time.time() - self.last_click_time < CLICK_COOLDOWN:
            return

        cards = self.get_current_cards()
        for card in cards:
            if card.is_clicked(px, py):
                self.last_click_time = time.time()
                
                # SPECIAL CASE: Colors
                if self.selected_category == "colors":
                    if self.selected_object_index != -1 and self.selected_object_index < len(self.placed_objects):
                        obj = self.placed_objects[self.selected_object_index]
                        # Determine the color from the card
                        new_color = None
                        if "white" in card.id: new_color = (255, 255, 255)
                        elif "gray" in card.id: new_color = (128, 128, 128)
                        elif "beige" in card.id: new_color = (245, 245, 220)
                        elif "blue" in card.id: new_color = (100, 149, 237)
                        elif "red" in card.id: new_color = (220, 60, 60)
                        elif "green" in card.id: new_color = (60, 180, 80)
                        elif "black" in card.id: new_color = (40, 40, 40)
                        elif "wood" in card.id: new_color = (160, 110, 60)
                        
                        if new_color:
                            # Apply to all material groups (selective per-group later)
                            if obj.materials:
                                # Apply to all material groups
                                for mat_id in list(obj.materials.keys()):
                                    obj.materials[mat_id] = new_color
                            else:
                                obj.color = new_color
                            print(f"🎨 Painted {obj.name} → {card.name}")
                    else:
                        print("⚠️ select an object first to paint it")
                    return

                # Normal object creation
                # Create NEW Object3D instead of holding the card
                new_obj = self._create_3d_object_from_card(card)
                self.held_object = new_obj
                self.toggle() # Minimize menu to see world
                print(f"✊ Picking up 3D: {new_obj.name}")
                break

    
    def get_current_cards(self):
        """Get cards for currently selected category"""
        return self.item_database.get(self.selected_category, [])
    
    def get_screen_position(self):
        """Get menu position in pixels"""
        return (int(self.pos_x * self.screen_width), 
                int(self.pos_y * self.screen_height))
    
    def get_minimized_button_position(self):
        """Return (x, y) of the floating button in pixels"""
        return (int(self.minimized_pos_x * self.screen_width),
                int(self.minimized_pos_y * self.screen_height))

    def get_settings_button_position(self):
        """Return (x, y) of the floating settings button in pixels"""
        main_x, main_y = self.get_minimized_button_position()
        spacing = 20
        size = self.minimized_button_size + 10 
        
        # Adaptive Stacking: Check if we are in top half or bottom half
        if self.minimized_pos_y < 0.5:
            # Top half -> Stack BELOW
            return (main_x, main_y + size + spacing)
        else:
            # Bottom half -> Stack ABOVE
            return (main_x, main_y - size - spacing)

    def get_virtual_mode_button_position(self):
        """Return (x, y) of the Virtual Mode button"""
        set_x, set_y = self.get_settings_button_position()
        spacing = 20
        size = self.minimized_button_size + 10
        
        # Follow same direction as settings button
        if self.minimized_pos_y < 0.5:
            # Top half -> Stack BELOW settings
            return (set_x, set_y + size + spacing)
        else:
            # Bottom half -> Stack ABOVE settings
            return (set_x, set_y - size - spacing)

    def _is_over_minimized_button(self, px, py):
        """Helper to check if over ANY minimized button (Menu, Settings, etc)"""
        if self.state != MenuState.MINIMIZED: return False
        
        # Check Main Button
        mx, my = self.get_minimized_button_position()
        sz = self.minimized_button_size
        if mx <= px <= mx+sz and my <= py <= my+sz: return True
        
        # Check Settings
        sx, sy = self.get_settings_button_position()
        if sx <= px <= sx+sz and sy <= py <= sy+sz: return True
        
        # Check Virtual Mode
        vx, vy = self.get_virtual_mode_button_position()
        if vx <= px <= vx+sz and vy <= py <= vy+sz: return True
        
        # Check Import Button
        ix, iy = self.get_import_button_position()
        if ix <= px <= ix+sz and iy <= py <= iy+sz: return True
        
        return False
