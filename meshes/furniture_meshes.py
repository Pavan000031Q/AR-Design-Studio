"""
Procedural 3D furniture & structural meshes with named material groups.
Each model returns a Mesh with multiple material_groups so parts can be
colored independently via Object3D.materials dict.
"""

import numpy as np
from core.graphics_engine import Mesh


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _box(cx, cy, cz, sx, sy, sz, vert_offset=0):
    """Generate a box (cuboid) centered at (cx, cy, cz) with half-sizes (sx, sy, sz).
    Returns (vertices_list, faces_list) with indices offset by vert_offset.
    """
    verts = [
        [cx - sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz + sz],
    ]
    o = vert_offset
    faces = [
        [o+0, o+1, o+2], [o+0, o+2, o+3],  # Front
        [o+5, o+4, o+7], [o+5, o+7, o+6],  # Back
        [o+4, o+0, o+3], [o+4, o+3, o+7],  # Left
        [o+1, o+5, o+6], [o+1, o+6, o+2],  # Right
        [o+3, o+2, o+6], [o+3, o+6, o+7],  # Top
        [o+4, o+5, o+1], [o+4, o+1, o+0],  # Bottom
    ]
    return verts, faces


def _cylinder(cx, cy, cz, radius, height, segments=8, vert_offset=0):
    """Generate a simple cylinder along Y axis centered at (cx, cy, cz)."""
    verts = []
    faces = []
    hy = height / 2.0
    
    # Bottom center, top center
    verts.append([cx, cy - hy, cz])  # 0
    verts.append([cx, cy + hy, cz])  # 1
    
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = cx + radius * np.cos(angle)
        z = cz + radius * np.sin(angle)
        verts.append([x, cy - hy, z])  # bottom ring: 2 + i
        verts.append([x, cy + hy, z])  # top ring: 2 + i + segments (interleaved actually)
    
    # Re-organize: bottom ring at indices 2..segments+1, top ring at segments+2..2*segments+1
    verts_clean = []
    verts_clean.append([cx, cy - hy, cz])  # 0: bottom center
    verts_clean.append([cx, cy + hy, cz])  # 1: top center
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = cx + radius * np.cos(angle)
        z = cz + radius * np.sin(angle)
        verts_clean.append([x, cy - hy, z])  # bottom ring
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = cx + radius * np.cos(angle)
        z = cz + radius * np.sin(angle)
        verts_clean.append([x, cy + hy, z])  # top ring
    
    o = vert_offset
    faces = []
    for i in range(segments):
        n = (i + 1) % segments
        # Bottom fan
        faces.append([o + 0, o + 2 + n, o + 2 + i])
        # Top fan
        faces.append([o + 1, o + 2 + segments + i, o + 2 + segments + n])
        # Side quads (2 tris each)
        bi = o + 2 + i
        bn = o + 2 + n
        ti = o + 2 + segments + i
        tn = o + 2 + segments + n
        faces.append([bi, bn, tn])
        faces.append([bi, tn, ti])
    
    return verts_clean, faces


def _build_mesh(parts):
    """Build a Mesh from a list of (material_id, default_color, vertices, faces) tuples.
    Returns a Mesh with proper material_groups.
    """
    all_verts = []
    all_faces = []
    material_groups = []
    
    vert_offset = 0
    index_offset = 0
    
    for mat_id, color, verts, faces in parts:
        all_verts.extend(verts)
        all_faces.extend(faces)
        
        num_indices = len(faces) * 3
        material_groups.append((mat_id, index_offset, num_indices, color))
        index_offset += num_indices
        vert_offset += len(verts)
    
    return Mesh(
        np.array(all_verts, dtype=np.float32),
        np.array(all_faces, dtype=np.int32),
        material_groups=material_groups
    )


# ─────────────────────────────────────────────
#  FURNITURE
# ─────────────────────────────────────────────

def create_sofa():
    """Modern sofa with body, cushions, armrests, and legs."""
    parts = []
    o = 0  # running vertex offset
    
    # Body (wide box)
    v, f = _box(0, 0.25, 0, 0.8, 0.15, 0.4, o)
    parts.append(('body', (80, 80, 90), v, f)); o += len(v)
    
    # Back cushion
    v, f = _box(0, 0.55, -0.3, 0.7, 0.15, 0.08, o)
    parts.append(('cushions', (120, 60, 60), v, f)); o += len(v)
    
    # Seat cushions (two side by side)
    v1, f1 = _box(-0.35, 0.42, 0.05, 0.32, 0.04, 0.3, o)
    parts.append(('cushions', (120, 60, 60), v1, f1)); o += len(v1)
    v2, f2 = _box(0.35, 0.42, 0.05, 0.32, 0.04, 0.3, o)
    parts.append(('cushions', (120, 60, 60), v2, f2)); o += len(v2)
    
    # Left armrest
    v, f = _box(-0.85, 0.4, 0, 0.06, 0.2, 0.4, o)
    parts.append(('armrests', (90, 90, 100), v, f)); o += len(v)
    
    # Right armrest
    v, f = _box(0.85, 0.4, 0, 0.06, 0.2, 0.4, o)
    parts.append(('armrests', (90, 90, 100), v, f)); o += len(v)
    
    # 4 Legs
    for x, z in [(-0.7, -0.3), (0.7, -0.3), (-0.7, 0.3), (0.7, 0.3)]:
        v, f = _box(x, 0.04, z, 0.04, 0.06, 0.04, o)
        parts.append(('legs', (50, 40, 30), v, f)); o += len(v)
    
    return _build_mesh(parts)


def create_l_sofa():
    """L-shaped sectional sofa."""
    parts = []
    o = 0
    
    # Main section
    v, f = _box(0, 0.25, 0, 0.8, 0.15, 0.4, o)
    parts.append(('body', (80, 80, 90), v, f)); o += len(v)
    
    # Corner section (extends to the right-back)
    v, f = _box(0.85, 0.25, -0.55, 0.25, 0.15, 0.55, o)
    parts.append(('corner', (85, 85, 95), v, f)); o += len(v)
    
    # Main back
    v, f = _box(0, 0.55, -0.3, 0.7, 0.15, 0.08, o)
    parts.append(('cushions', (140, 70, 70), v, f)); o += len(v)
    
    # Corner back  
    v, f = _box(1.02, 0.55, -0.55, 0.08, 0.15, 0.55, o)
    parts.append(('cushions', (140, 70, 70), v, f)); o += len(v)
    
    # Seat cushion
    v, f = _box(0, 0.42, 0.05, 0.7, 0.04, 0.3, o)
    parts.append(('cushions', (140, 70, 70), v, f)); o += len(v)
    
    # 6 Legs
    for x, z in [(-0.7, -0.3), (0.7, -0.3), (-0.7, 0.3), (0.7, 0.3), (1.0, -0.9), (1.0, -0.1)]:
        v, f = _box(x, 0.04, z, 0.04, 0.06, 0.04, o)
        parts.append(('legs', (50, 40, 30), v, f)); o += len(v)
    
    return _build_mesh(parts)


def create_chair():
    """Office/dining chair with seat, backrest, and legs."""
    parts = []
    o = 0
    
    # Seat
    v, f = _box(0, 0.35, 0, 0.3, 0.03, 0.3, o)
    parts.append(('seat', (100, 80, 60), v, f)); o += len(v)
    
    # Backrest
    v, f = _box(0, 0.65, -0.27, 0.28, 0.2, 0.03, o)
    parts.append(('backrest', (110, 90, 70), v, f)); o += len(v)
    
    # 4 Legs
    for x, z in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]:
        v, f = _box(x, 0.16, z, 0.025, 0.16, 0.025, o)
        parts.append(('legs', (60, 50, 40), v, f)); o += len(v)
    
    return _build_mesh(parts)


def create_table():
    """Coffee/dining table with tabletop and legs."""
    parts = []
    o = 0
    
    # Tabletop
    v, f = _box(0, 0.4, 0, 0.6, 0.03, 0.4, o)
    parts.append(('tabletop', (160, 120, 80), v, f)); o += len(v)
    
    # 4 Legs
    for x, z in [(-0.5, -0.3), (0.5, -0.3), (-0.5, 0.3), (0.5, 0.3)]:
        v, f = _box(x, 0.19, z, 0.03, 0.19, 0.03, o)
        parts.append(('legs', (100, 70, 40), v, f)); o += len(v)
    
    return _build_mesh(parts)


def create_bed():
    """King bed with frame, mattress, headboard, and legs."""
    parts = []
    o = 0
    
    # Frame
    v, f = _box(0, 0.2, 0, 0.7, 0.05, 0.9, o)
    parts.append(('frame', (120, 90, 60), v, f)); o += len(v)
    
    # Mattress
    v, f = _box(0, 0.33, 0.05, 0.65, 0.08, 0.8, o)
    parts.append(('mattress', (220, 220, 230), v, f)); o += len(v)
    
    # Headboard
    v, f = _box(0, 0.55, -0.85, 0.7, 0.25, 0.04, o)
    parts.append(('headboard', (100, 70, 45), v, f)); o += len(v)
    
    # Pillow left
    v, f = _box(-0.3, 0.43, -0.6, 0.2, 0.04, 0.15, o)
    parts.append(('mattress', (240, 240, 245), v, f)); o += len(v)
    
    # Pillow right
    v, f = _box(0.3, 0.43, -0.6, 0.2, 0.04, 0.15, o)
    parts.append(('mattress', (240, 240, 245), v, f)); o += len(v)
    
    # 4 legs
    for x, z in [(-0.6, -0.8), (0.6, -0.8), (-0.6, 0.8), (0.6, 0.8)]:
        v, f = _box(x, 0.07, z, 0.04, 0.08, 0.04, o)
        parts.append(('legs', (80, 60, 40), v, f)); o += len(v)
    
    return _build_mesh(parts)


def create_lamp():
    """Standing lamp with base, pole, and shade."""
    parts = []
    o = 0
    
    # Base (flat disc approximated as short wide box)
    v, f = _box(0, 0.03, 0, 0.2, 0.03, 0.2, o)
    parts.append(('base', (40, 40, 45), v, f)); o += len(v)
    
    # Pole
    v, f = _box(0, 0.55, 0, 0.02, 0.5, 0.02, o)
    parts.append(('pole', (180, 180, 170), v, f)); o += len(v)
    
    # Shade (truncated pyramid approximated as box)
    v, f = _box(0, 1.0, 0, 0.2, 0.12, 0.2, o)
    parts.append(('shade', (255, 230, 180), v, f)); o += len(v)
    
    return _build_mesh(parts)


def create_shelf():
    """Bookshelf with frame and shelves."""
    parts = []
    o = 0
    
    # Left side
    v, f = _box(-0.45, 0.5, 0, 0.03, 0.5, 0.2, o)
    parts.append(('frame', (130, 95, 60), v, f)); o += len(v)
    
    # Right side
    v, f = _box(0.45, 0.5, 0, 0.03, 0.5, 0.2, o)
    parts.append(('frame', (130, 95, 60), v, f)); o += len(v)
    
    # Back panel
    v, f = _box(0, 0.5, -0.18, 0.42, 0.5, 0.02, o)
    parts.append(('frame', (130, 95, 60), v, f)); o += len(v)
    
    # 4 Shelves (including top and bottom)
    for y in [0.02, 0.28, 0.55, 0.82]:
        v, f = _box(0, y, 0, 0.42, 0.02, 0.2, o)
        parts.append(('shelves', (150, 115, 75), v, f)); o += len(v)
    
    # Top
    v, f = _box(0, 1.0, 0, 0.45, 0.02, 0.22, o)
    parts.append(('frame', (130, 95, 60), v, f)); o += len(v)
    
    return _build_mesh(parts)


# ─────────────────────────────────────────────
#  STRUCTURAL
# ─────────────────────────────────────────────

def create_wall_panel():
    """Simple wall panel."""
    parts = []
    o = 0
    
    v, f = _box(0, 0.5, 0, 0.8, 0.5, 0.04, o)
    parts.append(('wall_face', (200, 195, 185), v, f))
    
    return _build_mesh(parts)


def create_door():
    """Door with frame, panel, and handle."""
    parts = []
    o = 0
    
    # Frame (U-shape: left, right, top)
    v, f = _box(-0.4, 0.5, 0, 0.04, 0.5, 0.06, o)
    parts.append(('frame', (140, 110, 75), v, f)); o += len(v)
    v, f = _box(0.4, 0.5, 0, 0.04, 0.5, 0.06, o)
    parts.append(('frame', (140, 110, 75), v, f)); o += len(v)
    v, f = _box(0, 0.98, 0, 0.4, 0.04, 0.06, o)
    parts.append(('frame', (140, 110, 75), v, f)); o += len(v)
    
    # Door panel
    v, f = _box(0, 0.47, 0, 0.34, 0.45, 0.03, o)
    parts.append(('panel', (170, 140, 100), v, f)); o += len(v)
    
    # Handle
    v, f = _box(0.25, 0.45, 0.05, 0.02, 0.03, 0.02, o)
    parts.append(('handle', (200, 200, 210), v, f)); o += len(v)
    
    return _build_mesh(parts)


def create_window():
    """Window with frame and glass pane."""
    parts = []
    o = 0
    
    # Frame (outer border)
    # Left
    v, f = _box(-0.4, 0.5, 0, 0.03, 0.35, 0.03, o)
    parts.append(('frame', (220, 220, 225), v, f)); o += len(v)
    # Right
    v, f = _box(0.4, 0.5, 0, 0.03, 0.35, 0.03, o)
    parts.append(('frame', (220, 220, 225), v, f)); o += len(v)
    # Top
    v, f = _box(0, 0.85, 0, 0.4, 0.03, 0.03, o)
    parts.append(('frame', (220, 220, 225), v, f)); o += len(v)
    # Bottom
    v, f = _box(0, 0.15, 0, 0.4, 0.03, 0.03, o)
    parts.append(('frame', (220, 220, 225), v, f)); o += len(v)
    # Center divider
    v, f = _box(0, 0.5, 0, 0.015, 0.35, 0.03, o)
    parts.append(('frame', (220, 220, 225), v, f)); o += len(v)
    
    # Glass pane (thin, semi-opaque blue-ish)
    v, f = _box(0, 0.5, 0, 0.37, 0.32, 0.005, o)
    parts.append(('glass', (180, 210, 240), v, f)); o += len(v)
    
    return _build_mesh(parts)


# ─────────────────────────────────────────────
#  REGISTRY — for easy lookup
# ─────────────────────────────────────────────

MESH_REGISTRY = {
    'sofa': create_sofa,
    'l_sofa': create_l_sofa,
    'chair': create_chair,
    'table': create_table,
    'bed': create_bed,
    'lamp': create_lamp,
    'shelf': create_shelf,
    'wall_panel': create_wall_panel,
    'door': create_door,
    'window': create_window,
}
