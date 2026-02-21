
import numpy as np

def simple_decimate(vertices, faces, target_count):
    """
    Simple mesh decimation by removing faces based on area.
    Fallback when pyfqmr is not available.
    """
    if len(faces) <= target_count:
        return vertices, faces
    
    # Calculate face areas
    areas = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = np.linalg.norm(np.cross(edge1, edge2)) * 0.5
        areas.append(area)
    
    # Sort faces by area (keep larger faces)
    indices = np.argsort(areas)[::-1]
    kept_faces = faces[indices[:target_count]]
    
    # Reindex vertices (remove unused)
    used_vertices = np.unique(kept_faces.flatten())
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
    
    new_vertices = vertices[used_vertices]
    new_faces = np.array([[vertex_map[v] for v in face] for face in kept_faces])
    
    return new_vertices, new_faces

def decimate_mesh(vertices, faces, target_count=5000):
    """
    Decimate mesh to target triangle count.
    """
    try:
        import pyfqmr
        # Use fast quadric mesh reduction
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(vertices, faces)
        mesh_simplifier.simplify_mesh(target_count=target_count, preserve_border=True, verbose=False)
        
        new_vertices, new_faces, _ = mesh_simplifier.getMesh()
        print(f"  ✂️  Decimated: {len(faces)} → {len(new_faces)} triangles")
        return new_vertices, new_faces
    except ImportError:
        # Fallback to simple decimation
        new_vertices, new_faces = simple_decimate(vertices, faces, target_count)
        print(f"  ✂️  Simple decimation: {len(faces)} → {len(new_faces)} triangles")
        return new_vertices, new_faces
    except Exception as e:
        print(f"  ⚠️  Decimation failed: {e}, using original mesh")
        return vertices, faces

def calculate_bounding_sphere(vertices):
    """Calculate bounding sphere for frustum culling"""
    center = np.mean(vertices, axis=0)
    radius = np.max(np.linalg.norm(vertices - center, axis=1))
    return center, radius
