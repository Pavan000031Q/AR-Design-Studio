"""
GLB/glTF Binary Loader
Extracts vertices, normals, indices, and material groups from GLB files.
Returns data ready for GPU upload via the Mesh/Object3D system.
"""

import numpy as np
import struct
from pygltflib import GLTF2

# glTF component type → numpy dtype
COMPONENT_TYPE_MAP = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

# glTF accessor type → number of components
ACCESSOR_TYPE_COUNT = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


class MaterialGroup:
    """One renderable sub-mesh with its own material/color."""
    def __init__(self, material_id, start_index, index_count, color):
        self.material_id = material_id
        self.start_index = start_index
        self.index_count = index_count
        self.color = color  # (R, G, B) 0-255


def _read_accessor(gltf, accessor_index, blob):
    """Read accessor data from the binary blob, return numpy array."""
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    
    dtype = COMPONENT_TYPE_MAP.get(accessor.componentType, np.float32)
    num_components = ACCESSOR_TYPE_COUNT.get(accessor.type, 1)
    
    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    byte_stride = buffer_view.byteStride
    count = accessor.count
    
    if byte_stride and byte_stride > num_components * np.dtype(dtype).itemsize:
        # Interleaved data — read element by element
        element_size = num_components * np.dtype(dtype).itemsize
        result = np.zeros((count, num_components), dtype=dtype)
        for i in range(count):
            offset = byte_offset + i * byte_stride
            result[i] = np.frombuffer(blob, dtype=dtype, count=num_components, offset=offset)
        return result
    else:
        # Tightly packed
        total_elements = count * num_components
        data = np.frombuffer(blob, dtype=dtype, count=total_elements, offset=byte_offset)
        if num_components > 1:
            data = data.reshape(count, num_components)
        return data


def _compute_normals(vertices, indices):
    """Compute smooth normals from vertices and triangle indices."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    faces = indices.reshape(-1, 3)
    
    for face in faces:
        v0, v1, v2 = vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        n = np.cross(edge1, edge2)
        norm = np.linalg.norm(n)
        if norm > 0:
            n /= norm
        normals[face[0]] += n
        normals[face[1]] += n
        normals[face[2]] += n
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals /= norms
    return normals


def _normalize_to_unit(vertices):
    """Scale and center vertices to fit within a unit bounding box [-0.5, 0.5]."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = (vmax - vmin).max()
    
    if extent < 1e-6:
        extent = 1.0
    
    normalized = (vertices - center) / extent
    return normalized


def load_glb(path, max_triangles=15000, normalize=True):
    """
    Load a GLB file and extract geometry + materials.
    
    Args:
        path: Path to .glb file
        max_triangles: Maximum triangle count (decimate if exceeded)
        normalize: If True, scale model to fit unit bounding box
    
    Returns:
        tuple: (vertices, normals, indices, material_groups)
            - vertices: np.array (N, 3) float32
            - normals: np.array (N, 3) float32
            - indices: np.array (M,) uint32
            - material_groups: list of MaterialGroup
        Returns (None, None, None, None) on failure.
    """
    try:
        gltf = GLTF2().load(path)
        blob = gltf.binary_blob()
        
        if blob is None:
            print(f"❌ GLB file has no binary data: {path}")
            return None, None, None, None
        
        all_vertices = []
        all_normals = []
        all_indices = []
        material_groups = []
        
        vertex_offset = 0  # Running offset for index remapping
        index_offset = 0   # Running offset for material group start
        
        for mesh_idx, mesh in enumerate(gltf.meshes):
            for prim_idx, primitive in enumerate(mesh.primitives):
                # ---- Extract positions ----
                if primitive.attributes.POSITION is None:
                    continue
                
                positions = _read_accessor(gltf, primitive.attributes.POSITION, blob)
                positions = positions.astype(np.float32)
                
                # ---- Extract normals ----
                if primitive.attributes.NORMAL is not None:
                    normals = _read_accessor(gltf, primitive.attributes.NORMAL, blob)
                    normals = normals.astype(np.float32)
                else:
                    normals = None  # Will compute later
                
                # ---- Extract indices ----
                if primitive.indices is not None:
                    indices = _read_accessor(gltf, primitive.indices, blob)
                    indices = indices.flatten().astype(np.uint32)
                else:
                    # No index buffer — generate sequential indices
                    indices = np.arange(len(positions), dtype=np.uint32)
                
                # Compute normals if not present
                if normals is None:
                    normals = _compute_normals(positions, indices)
                
                # ---- Extract material color ----
                color = (200, 200, 200)  # Default gray
                material_id = f"mat_{mesh_idx}_{prim_idx}"
                
                if primitive.material is not None and primitive.material < len(gltf.materials):
                    mat = gltf.materials[primitive.material]
                    if mat.pbrMetallicRoughness and mat.pbrMetallicRoughness.baseColorFactor:
                        r, g, b, a = mat.pbrMetallicRoughness.baseColorFactor
                        color = (int(r * 255), int(g * 255), int(b * 255))
                    material_id = mat.name or material_id
                
                # Remap indices to global vertex array
                remapped_indices = indices + vertex_offset
                
                # Create material group
                group = MaterialGroup(
                    material_id=material_id,
                    start_index=index_offset,
                    index_count=len(remapped_indices),
                    color=color,
                )
                material_groups.append(group)
                
                all_vertices.append(positions)
                all_normals.append(normals)
                all_indices.append(remapped_indices)
                
                vertex_offset += len(positions)
                index_offset += len(remapped_indices)
        
        if not all_vertices:
            print(f"⚠️ GLB file has no geometry: {path}")
            return None, None, None, None
        
        # Concatenate all primitives
        vertices = np.concatenate(all_vertices, axis=0).astype(np.float32)
        normals = np.concatenate(all_normals, axis=0).astype(np.float32)
        indices = np.concatenate(all_indices, axis=0).astype(np.uint32)
        
        num_tris = len(indices) // 3
        print(f"📦 GLB loaded: {len(vertices)} vertices, {num_tris} triangles, {len(material_groups)} material groups")
        
        # Decimate if needed
        if num_tris > max_triangles:
            from mesh_optimizer import decimate_mesh
            print(f"⚠️ Model has {num_tris} triangles, decimating to {max_triangles}...")
            faces = indices.reshape(-1, 3).astype(np.int32)
            vertices, faces = decimate_mesh(vertices, faces, target_count=max_triangles)
            indices = faces.flatten().astype(np.uint32)
            normals = _compute_normals(vertices, indices)
            
            # After decimation, collapse to single material group (indices changed)
            material_groups = [MaterialGroup(
                material_id="decimated",
                start_index=0,
                index_count=len(indices),
                color=material_groups[0].color if material_groups else (200, 200, 200),
            )]
        
        # Normalize scale
        if normalize:
            vertices = _normalize_to_unit(vertices)
        
        return vertices, normals, indices, material_groups
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error loading GLB {path}: {e}")
        return None, None, None, None
