import numpy as np
from pyrr import Matrix44, Vector3, Vector4

def test_projection():
    print("=== AR Projection Debug ===\n")

    # 1. Screen Dimensions
    width, height = 1920, 1080
    aspect = width / height
    fov = 60.0
    
    print(f"Screen: {width}x{height}, Aspect: {aspect:.2f}, FOV: {fov}")

    # 2. Camera Setup (Identity for now - camera at origin)
    # OpenGL Camera: Right (+X), Up (+Y), Back (+Z) -> Looks down -Z
    # We want to simulate the camera looking at an object.
    
    camera_pos = Vector3([0.0, 0.0, 0.0])
    
    # View Matrix
    # LookAt(eye, target, up)
    # Eye is at 0,0,0. Target is 0,0,-1 (Forward). Up is 0,1,0.
    view = Matrix44.look_at(
        (0, 0, 0),
        (0, 0, -1),
        (0, 1, 0)
    )
    print("\nView Matrix (LookAt 0,0,-1):")
    print(view)

    # 3. Projection Matrix
    # Z_Near = 0.1, Z_Far = 1000.0
    proj = Matrix44.perspective_projection(fov, aspect, 0.1, 1000.0)
    print("\nProjection Matrix:")
    print(proj)

    # 4. Test Object
    # Positioned 500 units in front of camera (Z = -500 in OpenGL)
    obj_pos = Vector3([0.0, 0.0, -500.0])
    print(f"\nObject World Pos: {obj_pos}")

    # Model Matrix
    model = Matrix44.from_translation(obj_pos)
    
    # 5. Pipeline Transform
    # MVP = Proj * View * Model
    mvp = proj * view * model
    
    print("\nMVP Matrix:")
    print(mvp)

    # 6. Clip Space Coordinate
    # v_clip = MVP * v_local
    # Local point at object origin (0,0,0,1)
    v_local = Vector4([0.0, 0.0, 0.0, 1.0])
    v_clip = mvp * v_local
    print(f"\nClip Space Pos: {v_clip}")

    # 7. NDC (Normalized Device Coordinates)
    # v_ndc = v_clip / v_clip.w
    if v_clip.w == 0:
        print("❌ Clip W is 0! Invalid projection.")
        return

    v_ndc = Vector3([v_clip.x / v_clip.w, v_clip.y / v_clip.w, v_clip.z / v_clip.w])
    print(f"NDC Pos: {v_ndc}")

    # Check Visibility
    if -1 <= v_ndc.x <= 1 and -1 <= v_ndc.y <= 1 and -1 <= v_ndc.z <= 1:
        print("✅ Object is INSIDE frustum.")
    else:
        print("❌ Object is OUTSIDE frustum.")

    # 8. Screen Coordinates
    # x = (ndc.x + 1) * 0.5 * width
    # y = (1 - ndc.y) * 0.5 * height  (Flip Y for Screen coords where 0 is top)
    
    screen_x = (v_ndc.x + 1) * 0.5 * width
    screen_y = (1 - v_ndc.y) * 0.5 * height # OpenGL Y is Up (+1), Screen Y is Down
    
    print(f"Screen Pos: ({screen_x:.1f}, {screen_y:.1f})")

    # --- Scenario 2: Camera Tracker Matrix ---
    # In the app, CameraTracker accumulates movement.
    # Let's verify what happens if we use the Identity matrix manually constructed logic vs LookAt.

if __name__ == "__main__":
    test_projection()
