
import numpy as np
from utils.mesh_optimizer import decimate_mesh

def load_obj(path, max_triangles=10000):
    """
    Parses an OBJ file with support for 'usemtl' (material groups).
    Returns:
        vertices: np.array of shape (N, 3), float32
        normals: np.array of shape (N, 3), float32
        tex_coords: np.array of shape (N, 2), float32
        indices: np.array of shape (M,), int32
        material_groups: list of (material_name, start_index, count)
    """

    vertices = []
    normals = []
    tex_coords = []
    
    # Store unique combinations of v/vt/vn to build index buffer
    unique_vertices = {}
    
    # Global buffers for the final mesh
    final_vertices = []
    final_normals = []
    final_tex_coords = []
    
    # Material grouping: 
    # { "MaterialName": [index1, index2, index3, ...] }
    material_indices = {}
    current_material = "default"
    
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                
                parts = line.split()
                
                if parts[0] == 'v':
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'vn':
                    normals.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'vt':
                    tex_coords.append([float(x) for x in parts[1:3]])
                elif parts[0] == 'usemtl':
                    # Switch current material
                    current_material = parts[1] if len(parts) > 1 else "default"
                elif parts[0] == 'f':
                    # Faces
                    face_vertices = parts[1:]
                    
                    # Ensure list for this material exists
                    if current_material not in material_indices:
                        material_indices[current_material] = []
                    
                    # Fan triangulation for polygons
                    for i in range(1, len(face_vertices) - 1):
                        tri_indices = [face_vertices[0], face_vertices[i], face_vertices[i+1]]
                        
                        for v_str in tri_indices:
                            # v_str format: v, v/vt, v//vn, v/vt/vn
                            if v_str in unique_vertices:
                                idx = unique_vertices[v_str]
                            else:
                                vals = v_str.split('/')
                                v_idx = int(vals[0]) - 1
                                
                                vt_idx = -1
                                vn_idx = -1
                                
                                if len(vals) > 1 and vals[1]:
                                    vt_idx = int(vals[1]) - 1
                                if len(vals) > 2 and vals[2]:
                                    vn_idx = int(vals[2]) - 1
                                
                                # Add vertex data
                                final_vertices.append(vertices[v_idx])
                                
                                if vn_idx != -1 and vn_idx < len(normals):
                                    final_normals.append(normals[vn_idx])
                                else:
                                    final_normals.append([0, 1, 0]) # Default normal
                                    
                                if vt_idx != -1 and vt_idx < len(tex_coords):
                                    final_tex_coords.append(tex_coords[vt_idx])
                                else:
                                    final_tex_coords.append([0, 0])
                                
                                idx = len(final_vertices) - 1
                                unique_vertices[v_str] = idx
                            
                            # Add index to CURRENT material group
                            material_indices[current_material].append(idx)
        
        if not final_vertices:
            print(f"⚠️ Warning: OBJ file {path} contained no valid vertices.")
            return None, None, None, None, None
        
        # --- Flatten Indices and Build Groups ---
        # We need a single index buffer, but ordered by material group
        # material_groups = [(mat_name, start_idx, count)]
        
        all_indices = []
        material_groups = []
        
        current_start = 0
        
        # Sort materials for deterministic order? (Optional, but good for debugging)
        for mat_name, indices in material_indices.items():
            count = len(indices)
            if count == 0: continue
            
            all_indices.extend(indices)
            material_groups.append((mat_name, current_start, count))
            current_start += count
            
        # Convert to numpy arrays
        verts = np.array(final_vertices, dtype='f4')
        norms = np.array(final_normals, dtype='f4')
        texs = np.array(final_tex_coords, dtype='f4')
        inds = np.array(all_indices, dtype='i4')
        
        # Optimization: Decimation (OPTIONAL - usually incompatible with strictly ordered groups unless aware)
        # We DISABLE decimation to support multi-materials safely.
        if len(inds) // 3 > max_triangles:
             print(f"⚠️ Model has {len(inds)//3} triangles. Decimation disabled to preserve material groups.")

        return verts, norms, texs, inds, material_groups
                
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error loading OBJ {path}: {e}")
        return None, None, None, None, None

def load_mtl(path):
    """
    Parses a .mtl file.
    Returns a dictionary:
    {
        "MaterialName": {
            "Kd": (r, g, b),    # Diffuse RGB (0-1)
            "map_Kd": "path/to/texture.jpg" # Texture path (relative or absolute)
        }
    }
    """
    materials = {}
    current_mtl = None
    
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                
                parts = line.split()
                
                if parts[0] == 'newmtl':
                    current_mtl = parts[1]
                    materials[current_mtl] = {}
                elif current_mtl:
                    if parts[0] == 'Kd':
                        # Diffuse color
                        materials[current_mtl]['Kd'] = (float(parts[1]), float(parts[2]), float(parts[3]))
                    elif parts[0] == 'map_Kd':
                        # Diffuse texture map
                        # Handle spaces in filenames (re-join parts)
                        tex_path = " ".join(parts[1:])
                        materials[current_mtl]['map_Kd'] = tex_path
                        
    except Exception as e:
        print(f"⚠️ Failed to load MTL {path}: {e}")
        
    return materials

def load_obj(path, max_triangles=10000):
    """
    Parses an OBJ file with support for 'usemtl' and associated .mtl files.
    Returns:
        vertices: np.array of shape (N, 3), float32
        normals: np.array of shape (N, 3), float32
        tex_coords: np.array of shape (N, 2), float32
        indices: np.array of shape (M,), int32
        material_groups: list of (material_name, start_index, count, color_rgb, texture_path)
    """

    vertices = []
    normals = []
    tex_coords = []
    
    # Store unique combinations of v/vt/vn to build index buffer
    unique_vertices = {}
    
    # Global buffers for the final mesh
    final_vertices = []
    final_normals = []
    final_tex_coords = []
    
    # Material grouping: 
    # { "MaterialName": [index1, index2, index3, ...] }
    material_indices = {}
    current_material = "default"
    
    # Load Materials (MTL) if available
    mtl_path = path.replace('.obj', '.mtl')
    materials_data = {}
    import os
    if os.path.exists(mtl_path):
        print(f"🎨 Found MTL file: {mtl_path}")
        materials_data = load_mtl(mtl_path)
    
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                
                parts = line.split()
                
                if parts[0] == 'v':
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'vn':
                    normals.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'vt':
                    tex_coords.append([float(x) for x in parts[1:3]])
                elif parts[0] == 'usemtl':
                    # Switch current material
                    current_material = parts[1] if len(parts) > 1 else "default"
                elif parts[0] == 'f':
                    # Faces
                    face_vertices = parts[1:]
                    
                    # Ensure list for this material exists
                    if current_material not in material_indices:
                        material_indices[current_material] = []
                    
                    # Fan triangulation for polygons
                    for i in range(1, len(face_vertices) - 1):
                        tri_indices = [face_vertices[0], face_vertices[i], face_vertices[i+1]]
                        
                        for v_str in tri_indices:
                            # v_str format: v, v/vt, v//vn, v/vt/vn
                            if v_str in unique_vertices:
                                idx = unique_vertices[v_str]
                            else:
                                vals = v_str.split('/')
                                v_idx = int(vals[0]) - 1
                                
                                vt_idx = -1
                                vn_idx = -1
                                
                                if len(vals) > 1 and vals[1]:
                                    vt_idx = int(vals[1]) - 1
                                if len(vals) > 2 and vals[2]:
                                    vn_idx = int(vals[2]) - 1
                                
                                # Add vertex data
                                final_vertices.append(vertices[v_idx])
                                
                                if vn_idx != -1 and vn_idx < len(normals):
                                    final_normals.append(normals[vn_idx])
                                else:
                                    final_normals.append([0, 1, 0]) # Default normal
                                    
                                if vt_idx != -1 and vt_idx < len(tex_coords):
                                    final_tex_coords.append(tex_coords[vt_idx])
                                else:
                                    final_tex_coords.append([0, 0])
                                
                                idx = len(final_vertices) - 1
                                unique_vertices[v_str] = idx
                            
                            # Add index to CURRENT material group
                            material_indices[current_material].append(idx)
        
        if not final_vertices:
            print(f"⚠️ Warning: OBJ file {path} contained no valid vertices.")
            return None, None, None, None, None
        
        # --- Flatten Indices and Build Groups ---
        # We need a single index buffer, but ordered by material group
        # material_groups = [(mat_name, start_idx, count, color, texture_path)]
        
        all_indices = []
        material_groups = []
        
        current_start = 0
        
        # Sort materials for deterministic order? (Optional, but good for debugging)
        for mat_name, indices in material_indices.items():
            count = len(indices)
            if count == 0: continue
            
            all_indices.extend(indices)
            
            # Lookup material properties
            color = (200, 200, 200) # Default grey
            texture_path = None
            
            if mat_name in materials_data:
                mat = materials_data[mat_name]
                if 'Kd' in mat:
                    # Convert 0-1 float rgb to 0-255 int rgb
                    r, g, b = mat['Kd']
                    color = (int(r*255), int(g*255), int(b*255))
                if 'map_Kd' in mat:
                    texture_path = mat['map_Kd']
            
            material_groups.append((mat_name, current_start, count, color, texture_path))
            current_start += count
            
        # Convert to numpy arrays
        verts = np.array(final_vertices, dtype='f4')
        norms = np.array(final_normals, dtype='f4')
        texs = np.array(final_tex_coords, dtype='f4')
        inds = np.array(all_indices, dtype='i4')
        
        # Optimization: Decimation (OPTIONAL - usually incompatible with strictly ordered groups unless aware)
        # We DISABLE decimation to support multi-materials safely.
        if len(inds) // 3 > max_triangles:
             print(f"⚠️ Model has {len(inds)//3} triangles. Decimation disabled to preserve material groups.")

        return verts, norms, texs, inds, material_groups
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error loading OBJ {path}: {e}")
        return None, None, None, None, None
