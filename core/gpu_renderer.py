
import moderngl
import numpy as np
from pyrr import Matrix44, Vector3
import cv2
import time

class GPURenderer:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        
        # Create standalone context
        try:
            self.ctx = moderngl.create_context(standalone=True)
        except Exception as e:
            print(f"❌ Failed to create ModernGL context: {e}")
            raise e
            
        # Create Framebuffer with TEXTURE (not Renderbuffer) so blur shader can sample it
        self.color_tex = self.ctx.texture((width, height), 4)
        self.depth_rbo = self.ctx.depth_renderbuffer((width, height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex],
            depth_attachment=self.depth_rbo
        )
        self.fbo.use()
        
        # Depth test and face culling
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        
        # Background texture for camera feed
        self.bg_texture = self.ctx.texture((width, height), 3)
        self.bg_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Background shader (simple textured quad)
        self.bg_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                
                void main() {
                    v_texcoord = in_texcoord;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D bg_tex;
                in vec2 v_texcoord;
                out vec4 f_color;
                
                void main() {
                    f_color = vec4(texture(bg_tex, v_texcoord).rgb, 1.0);
                }
            '''
        )
        
        # Fullscreen quad for background
        quad_vertices = np.array([
            # pos(x,y), texcoord(u,v)
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 0.0,
             1.0,  1.0,  1.0, 0.0,
        ], dtype='f4')
        
        self.bg_vbo = self.ctx.buffer(quad_vertices)
        self.bg_vao = self.ctx.vertex_array(
            self.bg_prog,
            [(self.bg_vbo, '2f 2f', 'in_vert', 'in_texcoord')]
        )
        
        # Grid Shader (for Virtual Mode)
        self.grid_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_vert;
                void main() {
                    gl_Position = Mvp * vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec4 Color;
                out vec4 f_color;
                void main() {
                    f_color = Color;
                }
            '''
        )
        
        # Grid VAO (Lines)
        # Create a grid on XZ plane at Y=0 (offset by target? No, usually Y=0 or Y=-100)
        # Let's make a grid centered near (0,0,500)?
        # Better: Infinite-ish grid at Y=-100 (Floor)
        grid_size = 2000
        step = 100
        grid_verts = []
        # Lines parallel to X
        for z in range(-grid_size, grid_size+1, step):
            grid_verts.extend([-grid_size, -100, z + 500,  grid_size, -100, z + 500])
        # Lines parallel to Z
        for x in range(-grid_size, grid_size+1, step):
            grid_verts.extend([x, -100, -grid_size + 500, x, -100, grid_size + 500])
            
        self.grid_vbo = self.ctx.buffer(np.array(grid_verts, dtype='f4').tobytes())
        self.grid_vao = self.ctx.vertex_array(
            self.grid_prog,
            [(self.grid_vbo, '3f', 'in_vert')]
        )
        self.grid_vertex_count = len(grid_verts) // 3
        
    # 3D Object Shader
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                
                uniform mat4 Mvp;
                uniform mat4 Model;
                
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord;
                
                out vec3 v_normal;
                out vec3 v_world_pos;
                out vec2 v_texcoord;
                
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_world_pos = (Model * vec4(in_position, 1.0)).xyz;
                    v_normal = mat3(Model) * in_normal;
                    v_texcoord = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                
                uniform vec3 Color;
                uniform vec3 LightDir;
                uniform vec3 CameraPos;
                uniform sampler2D diff_texture;
                uniform bool use_texture;
                
                in vec3 v_normal;
                in vec3 v_world_pos;
                in vec2 v_texcoord;
                
                out vec4 f_color;
                
                void main() {
                    vec3 norm = normalize(v_normal);
                    vec3 light = normalize(-LightDir);
                    
                    // Ambient
                    float ambient = 0.4;
                    
                    // Diffuse
                    float diff = max(dot(norm, light), 0.0);
                    
                    // Specular
                    vec3 viewDir = normalize(CameraPos - v_world_pos);
                    vec3 reflectDir = reflect(-light, norm);
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
                    
                    // Get base color from texture or uniform
                    vec3 baseColor = Color;
                    if (use_texture) {
                        vec4 texColor = texture(diff_texture, v_texcoord * vec2(1.0, -1.0)); // Flip V if needed
                        baseColor = texColor.rgb * Color; // Tint with color if needed, or just replace
                    }
                    
                    vec3 result = (ambient + diff + spec * 0.3) * baseColor;
                    f_color = vec4(result, 1.0);
                }
            '''
        )
        
        # Gaussian Blur Shader (2-Pass Optimized)
        self.blur_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    v_texcoord = in_texcoord;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D screen_tex;
                uniform vec2 texel_size;
                uniform vec2 direction; // (1,0) for horizontal, (0,1) for vertical
                in vec2 v_texcoord;
                out vec4 f_color;
                
                void main() {
                    vec2 uv = v_texcoord;
                    // Optimized 5-tap Gaussian (effectively 9-tap via linear sampling)
                    vec4 result = texture(screen_tex, uv) * 0.227027;
                    result += texture(screen_tex, uv + direction * texel_size * 1.38461538) * 0.31621622;
                    result += texture(screen_tex, uv - direction * texel_size * 1.38461538) * 0.31621622;
                    result += texture(screen_tex, uv + direction * texel_size * 3.23076923) * 0.07027027;
                    result += texture(screen_tex, uv - direction * texel_size * 3.23076923) * 0.07027027;
                    f_color = vec4(result.rgb, 1.0);
                }
            '''
        )
        
        # Ping-pong FBOs for multi-pass effects
        self.temp_tex = self.ctx.texture((width, height), 4)
        self.temp_fbo = self.ctx.framebuffer(color_attachments=[self.temp_tex])
        
        self.light_dir = Vector3([0.5, -1.0, -1.0])
        self.camera_pos = Vector3([0.0, 0.0, 0.0])
        
        # Texture Cache: path -> moderngl.Texture
        self.texture_cache = {}
        
        print("✅ GPU Renderer Initialized (with GPU Blur support)")

    def apply_blur(self, region_rect=None):
        """
        Apply GPU blur to the current FBO.
        region_rect: (x, y, w, h) in pixels. If None, blurs the whole screen.
        Only the specified region is blurred; the rest remains sharp.
        """
        if region_rect:
            x, y, w, h = region_rect
            # Clamp to valid framebuffer bounds
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(int(w), self.width - x)
            h = min(int(h), self.height - y)
            if w <= 0 or h <= 0:
                return
            # OpenGL Y is bottom-up, OpenCV is top-down
            gl_y = self.height - y - h
            scissor_rect = (x, gl_y, w, h)
        else:
            scissor_rect = None

        # Set scissor for both passes (restricts read AND write to region)
        if scissor_rect:
            self.ctx.scissor = scissor_rect
        else:
            self.ctx.scissor = None

        # Always use full viewport — the fullscreen quad needs full UV mapping
        # Scissor restricts which pixels actually get written
        self.ctx.viewport = (0, 0, self.width, self.height)

        # 1. Clear temp FBO in the scissored region to prevent stale data bleed
        self.temp_fbo.use()
        if scissor_rect:
            self.ctx.scissor = scissor_rect
        self.temp_fbo.clear(0.0, 0.0, 0.0, 1.0)

        # 2. Horizontal Pass: FBO Colors -> Temp FBO (only in region)
        self.color_tex.use(location=0)
        self.blur_prog['screen_tex'].value = 0
        self.blur_prog['direction'].value = (1.0, 0.0)
        self.blur_prog['texel_size'].value = (1.0 / self.width, 1.0 / self.height)
        self.bg_vao.render(moderngl.TRIANGLE_STRIP)

        # 3. Vertical Pass: Temp FBO -> Original FBO (only in region)
        self.fbo.use()
        if scissor_rect:
            self.ctx.scissor = scissor_rect
        self.temp_tex.use(location=0)
        self.blur_prog['direction'].value = (0.0, 1.0)
        self.bg_vao.render(moderngl.TRIANGLE_STRIP)
        
        # Reset state
        self.ctx.scissor = None
        self.ctx.viewport = (0, 0, self.width, self.height)

    def _load_texture(self, path):
        """Loads a texture from disk and caches it on GPU"""
        import os
        if not path or not os.path.exists(path):
            return None
            
        # Check cache
        if path in self.texture_cache:
            return self.texture_cache[path]
            
        try:
            # Load with OpenCV (BGR) -> RGB
            img = cv2.imread(path)
            if img is None:
                print(f"⚠️ Failed to load texture image: {path}")
                return None
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 0) # Flip Y for OpenGL
            
            # Create texture
            texture = self.ctx.texture(
                (img.shape[1], img.shape[0]), 
                3, 
                img.tobytes()
            )
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            
            self.texture_cache[path] = texture
            print(f"🖼️ Texture loaded: {path}")
            return texture
            
        except Exception as e:
            print(f"❌ Texture load error {path}: {e}")
            return None

    def render(self, frame, objects, camera_fov=60.0, blur_regions=None, view_matrix=None, draw_grid=False):
        # Upload camera frame to GPU texture
        # OpenCV BGR -> RGB (no vertical flip needed, texcoords handle it)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.bg_texture.write(frame_rgb.tobytes())
        
        # Clear framebuffer with depth
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        
        # Disable depth test for background
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Render background
        self.bg_prog['bg_tex'].value = 0
        self.bg_texture.use(location=0)
        self.bg_vao.render(moderngl.TRIANGLE_STRIP)
        
        # Enable depth test for 3D objects
        # DISABLE CULL_FACE for debugging (in case normals/winding are flipped)
        self.ctx.enable(moderngl.DEPTH_TEST) # | moderngl.CULL_FACE)
        
        # 3. Setup Camera Uniforms
        self.prog['CameraPos'].value = tuple(self.camera_pos)
        self.prog['LightDir'].value = tuple(self.light_dir)
        
        # Projection Matrix
        aspect = self.width / self.height
        proj = Matrix44.perspective_projection(camera_fov, aspect, 0.1, 5000.0)
        
        # View Matrix
        if view_matrix is None:
            # Fallback: push camera back so objects at origin are visible
            view = Matrix44.from_translation([0.0, 0.0, -500.0]).astype('f4')
        else:
            # view_matrix is row-major (numpy); transpose to column-major for pyrr
            view = Matrix44(np.array(view_matrix, dtype='f4').T)
            
        # Draw Floor Grid (Only if Virtual Mode / Custom View provided)
        # We assume if view_matrix is passed, we might want a grid? 
        # Or check a flag? For now, if view_matrix is passed, draw grid for context.
        if draw_grid:
             # Draw Grid (Subtle Grey)
             self.grid_prog['Mvp'].write((proj * view).astype('f4').tobytes())
             self.grid_prog['Color'].value = (0.4, 0.4, 0.45, 0.2) # Keeping it subtle against gradient
             self.ctx.enable(moderngl.BLEND)
             self.grid_vao.render(moderngl.LINES, vertices=self.grid_vertex_count)
             
             # Draw Axis Lines (Red X, Blue Z) for orientation
             # We can reuse the grid shader but just draw 2 specific lines? 
             # Or just create a small VBO for axes. 
             # Let's do a quick immediate-mode style draw for axes if possible, or just add them to VBO?
             # Easier: Just draw the grid. The gradient sky gives Up/Down orientation.
             # If user wants "Realism", a simple grid is fine. 
             
             self.ctx.disable(moderngl.BLEND)

        # Draw Objects
        for obj in objects:
            if not self.prepare_object(obj):
                continue
                
            model = Matrix44.from_translation(obj.position) * \
                    Matrix44.from_eulers(np.radians(obj.rotation)) * \
                    Matrix44.from_scale(obj.scale)
            
            mvp = proj * view * model
            
            self.prog['Mvp'].write(mvp.astype('f4').tobytes())
            self.prog['Model'].write(model.astype('f4').tobytes())
            
            # Render each material group with its own color
            for group in obj.mesh.material_groups:
                # Unpack - handle older 4-item groups safely if they exist
                if len(group) == 5:
                    mat_id, start_idx, idx_count, default_color, texture_path = group
                else:
                    mat_id, start_idx, idx_count, default_color = group
                    texture_path = None
                
                # Priority: obj.materials override > group default > obj.color
                color = np.array(default_color) / 255.0 # Default from OBJ
                
                if mat_id in obj.materials:
                     color = np.array(obj.materials[mat_id]) / 255.0
                elif default_color == (200, 200, 200): # If explicit grey
                     color = np.array(obj.color) / 255.0 # Use object base tint
                
                self.prog['Color'].value = tuple(color)
                
                # Texture binding
                self.prog['use_texture'].value = False
                if texture_path:
                    
                    # Fix relative paths (assume relative to assets/models if not absolute)
                    import os
                    if not os.path.exists(texture_path):
                         # Try looking in assets/models recursively? 
                         # For now, simplistic check
                         pass
                    
                    tex = self._load_texture(texture_path)
                    if tex:
                        tex.use(location=1)
                        self.prog['diff_texture'].value = 1
                        self.prog['use_texture'].value = True
                        # Reset color to white so we see full texture, or keep it to tint?
                        # Usually for pure texture we want white
                        self.prog['Color'].value = (1.0, 1.0, 1.0)
                
                # Draw sub-range of index buffer
                obj.mesh.vao.render(moderngl.TRIANGLES, vertices=idx_count, first=start_idx)
        
        # ✅ GPU POST-PROCESS: Apply blur regions BEFORE readback
        t0 = 0
        if objects:
             t0 = time.time()
        
        self.ctx.disable(moderngl.DEPTH_TEST)
        if blur_regions:
            for region in blur_regions:
                self.apply_blur(region_rect=region)
        
        # Read back final composite
        # ✅ OPTIMIZATION: Read 4 components (RGBA) instead of 3 (RGB). 
        # GPUs/Drivers are often MUCH faster at aligned 4-byte transfers.
        raw_data = self.fbo.read(components=4) 
        image = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 4))
        
        if objects and t0 > 0:
             dt = (time.time() - t0) * 1000
             # print(f"GPU Readback: {dt:.2f}ms")
        
        # Flip Y back for OpenCV
        image = cv2.flip(image, 0)
        
        # Convert RGBA -> BGR for OpenCV
        frame_out = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        return frame_out

    def prepare_object(self, obj):
        """Uploads mesh data to GPU VAO"""
        if obj.mesh.vao is not None:
            return True
            
        if not hasattr(obj.mesh, 'vbo_data'):
            return False

        vbo = self.ctx.buffer(obj.mesh.vbo_data)
        
        try:
            # New format: 3f pos, 3f norm, 2f tex
            obj.mesh.vao = self.ctx.vertex_array(
                self.prog,
                [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord')],
                index_buffer=self.ctx.buffer(obj.mesh.ibo_data) if hasattr(obj.mesh, 'ibo_data') else None
            )
            return True
        except Exception as e:
            print(f"Failed to create VAO for {obj.name}: {e}")
            return False
