from core.graphics_engine import Mesh
import numpy as np

def create_cube(size=1.0):
    s = size / 2.0
    vertices = [
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s], # Front
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]    # Back
    ]
    
    # Indices for triangles
    # Cube has 6 faces, 2 tris each = 12 tris
    faces = [
        [0, 1, 2], [0, 2, 3], # Front
        [5, 4, 7], [5, 7, 6], # Back
        [4, 0, 3], [4, 3, 7], # Left
        [1, 5, 6], [1, 6, 2], # Right
        [3, 2, 6], [3, 6, 7], # Top
        [4, 5, 1], [4, 1, 0]  # Bottom
    ]
    
    return Mesh(vertices, faces)

def create_pyramid(size=1.0, height=1.0):
    s = size / 2.0
    h = height / 2.0
    
    vertices = [
        [-s, h, -s], [s, h, -s], [s, h, s], [-s, h, s], # Base
        [0, -h, 0] # Tip
    ]
    
    # Pyramid: base (2 tris) + 4 sides
    faces = [
        [0, 1, 2], [0, 2, 3], # Base
        [0, 4, 1],
        [1, 4, 2],
        [2, 4, 3],
        [3, 4, 0]
    ]
    
    return Mesh(vertices, faces)

def create_plane(width=1.0, depth=1.0):
    w = width / 2.0
    d = depth / 2.0
    
    vertices = [
        [-w, 0, -d], [w, 0, -d], [w, 0, d], [-w, 0, d]
    ]
    
    faces = [
        [0, 1, 2], [0, 2, 3]
    ]
    
    return Mesh(vertices, faces)
