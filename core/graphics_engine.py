
from pyrr import Matrix44
import numpy as np
import cv2
import math

class Mesh:
    def __init__(self, vertices, faces, normals=None, indices=None, material_groups=None, tex_coords=None):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        self.normals = np.array(normals, dtype=np.float32) if normals is not None else None
        self.tex_coords = np.array(tex_coords, dtype=np.float32) if tex_coords is not None else None
        self.indices = np.array(indices, dtype=np.int32) if indices is not None else None
        self.material_groups = material_groups
        
        self.vao = None
        self.vbo_data = None
        self.ibo_data = None
        
        if len(self.vertices) > 0:
            self.bbox_min = np.min(self.vertices, axis=0)
            self.bbox_max = np.max(self.vertices, axis=0)
            self.tri_vertices = self.vertices[self.faces]
            self.tri_v0 = self.tri_vertices[:, 0]
            self.tri_v1 = self.tri_vertices[:, 1]
            self.tri_v2 = self.tri_vertices[:, 2]
        else:
            self.bbox_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
            self.bbox_max = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.tri_v0 = None
            self.tri_v1 = None
            self.tri_v2 = None
        
        self.prepare_gpu_data()
    
    def prepare_gpu_data(self):
        if self.normals is None:
            self.normals = np.zeros_like(self.vertices)
            for face in self.faces:
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]
                n = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(n)
                if norm > 0:
                    n /= norm
                self.normals[face[0]] = n
                self.normals[face[1]] = n
                self.normals[face[2]] = n
            
            norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.normals /= norms
        
        if self.tex_coords is None:
            self.tex_coords = np.zeros((len(self.vertices), 2), dtype=np.float32)
        
        data = np.hstack([self.vertices, self.normals, self.tex_coords])
        self.vbo_data = data.astype('f4').tobytes()
        
        if self.indices is None:
            self.indices = self.faces.flatten()
        self.ibo_data = self.indices.astype('i4').tobytes()
        
        if self.material_groups is None:
            self.material_groups = [("default", 0, len(self.indices), (200, 200, 200), None)]


class Object3D:
    _next_uid = 0
    
    def __init__(self, mesh, position=(0,0,0), rotation=(0,0,0), scale=(1,1,1), color=(200, 200, 200), category="misc"):
        self.mesh = mesh
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)
        self.color = color
        self.name = "Object"
        self.id = "obj_0"
        self.category = category
        self.group_id = None  # Legacy — now managed by ConstraintManager
        
        # Unique ID for constraint referencing
        Object3D._next_uid += 1
        self.obj_id = f"obj_{Object3D._next_uid}"
        
        # Per-material color overrides: material_id -> (R, G, B)
        self.materials = {}
    
    def get_model_matrix(self):
        # Match GPU Renderer's logic exactly using Pyrr
        t = Matrix44.from_translation(self.position)
        rx, ry, rz = np.radians(self.rotation)
        r = Matrix44.from_eulers((rx, ry, rz))
        s = Matrix44.from_scale(self.scale)
        model = t * r * s
        return np.array(model, dtype=np.float32)
    
    def get_local_axes(self):
        rx, ry, rz = np.radians(self.rotation)
        r = Matrix44.from_eulers((rx, ry, rz))
        
        v_right = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v_up = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        v_fwd = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        
        right = r @ v_right
        up = r @ v_up
        forward = r @ v_fwd
        
        return np.array(right[:3]), np.array(up[:3]), np.array(forward[:3])
    
    def get_world_corners(self):
        minv = self.mesh.bbox_min
        maxv = self.mesh.bbox_max
        
        lc = [
            (minv[0], minv[1], minv[2], 1.0),
            (maxv[0], minv[1], minv[2], 1.0),
            (minv[0], maxv[1], minv[2], 1.0),
            (maxv[0], maxv[1], minv[2], 1.0),
            (minv[0], minv[1], maxv[2], 1.0),
            (maxv[0], minv[1], maxv[2], 1.0),
            (minv[0], maxv[1], maxv[2], 1.0),
            (maxv[0], maxv[1], maxv[2], 1.0)
        ]
        
        M = self.get_model_matrix()
        world_corners = []
        for c in lc:
            wc = np.array(c) @ M
            world_corners.append(wc[:3])
        return world_corners
    
    def get_faces(self):
        """
        Returns 6 face descriptors: (name, center_world, normal_world)
        Face names: left, right, bottom, top, back, front
        """
        minv = self.mesh.bbox_min
        maxv = self.mesh.bbox_max
        mid = [(minv[i] + maxv[i]) / 2.0 for i in range(3)]
        
        # Local face centers (in model space before rotation/scale)
        local_faces = [
            ("left",   np.array([minv[0], mid[1], mid[2], 1.0]), np.array([-1, 0, 0, 0], dtype=np.float32)),
            ("right",  np.array([maxv[0], mid[1], mid[2], 1.0]), np.array([ 1, 0, 0, 0], dtype=np.float32)),
            ("bottom", np.array([mid[0], minv[1], mid[2], 1.0]), np.array([ 0,-1, 0, 0], dtype=np.float32)),
            ("top",    np.array([mid[0], maxv[1], mid[2], 1.0]), np.array([ 0, 1, 0, 0], dtype=np.float32)),
            ("back",   np.array([mid[0], mid[1], minv[2], 1.0]), np.array([ 0, 0,-1, 0], dtype=np.float32)),
            ("front",  np.array([mid[0], mid[1], maxv[2], 1.0]), np.array([ 0, 0, 1, 0], dtype=np.float32)),
        ]
        
        M = self.get_model_matrix()
        # Rotation matrix for normals (no translation)
        rx, ry, rz = np.radians(self.rotation)
        R = np.array(Matrix44.from_eulers((rx, ry, rz)), dtype=np.float32)
        
        faces = []
        for name, center_local, normal_local in local_faces:
            center_world = (np.array(center_local, dtype=np.float32) @ M)[:3]
            normal_world = (np.array(normal_local, dtype=np.float32) @ R)[:3]
            # Normalize
            n = np.linalg.norm(normal_world)
            if n > 0:
                normal_world /= n
            faces.append((name, center_world, normal_world))
        
        return faces
    
    def get_edge_midpoints(self):
        """
        Returns 12 edge midpoints in world space.
        Each edge is the midpoint between two adjacent corners.
        """
        corners = self.get_world_corners()
        # 12 edges of a box (indexed by corner pairs)
        edge_pairs = [
            (0,1), (2,3), (4,5), (6,7),  # X-axis edges
            (0,2), (1,3), (4,6), (5,7),  # Y-axis edges
            (0,4), (1,5), (2,6), (3,7),  # Z-axis edges
        ]
        midpoints = []
        for a, b in edge_pairs:
            mid = (corners[a] + corners[b]) / 2.0
            midpoints.append(mid)
        return midpoints
    
    def is_pointer_inside(self, px, py, screen_w, screen_h, focal_length, view_matrix=None):
        cx, cy = screen_w / 2, screen_h / 2
        
        dir_x = (px - cx) / focal_length
        dir_y = -(py - cy) / focal_length
        dir_z = -1.0
        
        ray_origin_cam = np.array([0, 0, 0, 1], dtype=np.float32)
        ray_dir_cam = np.array([dir_x, dir_y, dir_z, 0], dtype=np.float32)
        
        M = self.get_model_matrix()
        
        if view_matrix is None:
            from pyrr import Matrix44 as M44
            V = np.array(M44.from_translation([0.0, 0.0, -500.0]), dtype=np.float32)
        else:
            V = np.array(view_matrix, dtype=np.float32)
        
        MV = V @ M
        
        try:
            inv_MV = np.linalg.inv(MV)
        except np.linalg.LinAlgError:
            return False
        
        ray_origin_local = inv_MV @ ray_origin_cam
        ray_dir_local = inv_MV @ ray_dir_cam
        
        ray_dir_local = ray_dir_local[:3]
        ray_origin_local = ray_origin_local[:3]
        
        norm = np.linalg.norm(ray_dir_local)
        if norm > 0:
            ray_dir_local /= norm
        
        t_min = 0.0
        t_max = 100000.0
        b_min = self.mesh.bbox_min
        b_max = self.mesh.bbox_max
        
        for i in range(3):
            if abs(ray_dir_local[i]) < 1e-6:
                if ray_origin_local[i] < b_min[i] or ray_origin_local[i] > b_max[i]:
                    return False
            else:
                inv_d = 1.0 / ray_dir_local[i]
                t1 = (b_min[i] - ray_origin_local[i]) * inv_d
                t2 = (b_max[i] - ray_origin_local[i]) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_max < t_min:
                    return False
        
        return True
    
    def draw_hitbox_debug(self, frame, screen_w, screen_h, focal_length, view_matrix=None, color=(255, 0, 255)):
        """Draw the projected Convex Hull on the frame for debugging using proper View Matrix"""
        corners = self.get_world_corners()
        cx, cy = screen_w / 2, screen_h / 2
        f = focal_length
        
        if view_matrix is None:
            from pyrr import Matrix44 as M44
            V = np.array(M44.from_translation([0.0, 0.0, -500.0]), dtype=np.float32)
        else:
            V = np.array(view_matrix, dtype=np.float32)
        
        projected_points = []
        for c in corners:
            p_world = np.array([c[0], c[1], c[2], 1.0])
            p_cam = V @ p_world
            x, y, z = p_cam[0], p_cam[1], p_cam[2]
            
            if z < -0.1:  # Only project if in front of camera
                sx = int(cx + (x / -z) * f)
                sy = int(cy - (y / -z) * f)
                projected_points.append((sx, sy))
        
        if len(projected_points) < 3:
            return
        
        pts = np.array(projected_points, dtype=np.int32)
        
        try:
            hull = cv2.convexHull(pts)
            cv2.polylines(frame, [hull], True, color, 2, cv2.LINE_AA)
            
            # Draw center point
            center_world = np.append(self.position, 1.0)
            center_cam = V @ center_world
            
            if center_cam[2] < -0.1:
                csx = int(cx + (center_cam[0] / -center_cam[2]) * f)
                csy = int(cy - (center_cam[1] / -center_cam[2]) * f)
                cv2.circle(frame, (csx, csy), 5, color, -1)
                cv2.putText(frame, f"VISIBLE: TRUE", (csx + 10, csy - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception:
            pass
    
    def clone(self):
        new_obj = Object3D(
            self.mesh,
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            scale=self.scale.copy(),
            color=self.color
        )
        new_obj.materials = self.materials.copy()
        new_obj.name = f"{self.name} Copy"
        return new_obj
