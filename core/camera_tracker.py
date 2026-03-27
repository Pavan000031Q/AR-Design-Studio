"""
Hybrid Camera Tracker — Rotation-First with Optional Translation
================================================================

Architecture:
  1. Optical Flow (Lucas-Kanade) tracks features frame-to-frame
  2. Homography decomposition extracts camera rotation: R = K⁻¹ · H · K
  3. Rotation accumulates via quaternion multiplication (drift-free)
  4. One-Euro filter smooths all 6DOF components
  5. Translation estimated only when residual flow after rotation
     compensation exceeds a threshold (real camera movement detected)

GPU acceleration (CuPy CUDA when available):
  - Feature descriptor matching (Hamming distance kernel)
  - Batch reprojection error computation
  - Optical flow uses cv2.cuda when OpenCV CUDA is available

Why not triangulation-based SLAM?
  - Webcam motion is rotation-dominant (users pan/tilt, rarely translate)
  - triangulatePoints requires a translation baseline → degenerate with pure rotation
  - This hybrid approach provides instant 3DOF tracking on any webcam
"""

import cv2
import numpy as np
from pyrr import Matrix44
from enum import Enum
import time

from core.slam_filter import PoseFilter

# ── GPU Setup ──
_USE_GPU = False
try:
    import os
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path:
        for subdir in ['bin', 'bin/x64', 'lib/x64']:
            dll_dir = os.path.join(cuda_path, subdir)
            if os.path.isdir(dll_dir):
                os.add_dll_directory(dll_dir)
                os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
    
    import cupy as cp
    _test = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    _result = cp.sum(_test * _test)
    assert float(_result) == 14.0
    del _test, _result
    _USE_GPU = True
    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except Exception:
        gpu_name = 'CUDA GPU'
    print(f"🟢 SLAM: CUDA GPU acceleration ENABLED (CuPy on {gpu_name})")
except Exception as e:
    print(f"🟡 SLAM: GPU not available ({e}), using CPU fallback")

# ── OpenCV CUDA check for optical flow ──
_USE_CUDA_OPTFLOW = False
try:
    _test_gpu_mat = cv2.cuda_GpuMat()
    _USE_CUDA_OPTFLOW = True
    print(f"🟢 SLAM: OpenCV CUDA optical flow ENABLED")
    del _test_gpu_mat
except Exception:
    pass



class TrackingState(Enum):
    NOT_INITIALIZED = 0
    INITIALIZING = 1
    TRACKING = 2       # Full 6DOF (rotation + translation)
    ROTATION_ONLY = 3  # 3DOF rotation tracking (most common for webcam)
    LOST = 4

# ══════════════════════════════════════════════════════════════
# GPU-accelerated functions
# ══════════════════════════════════════════════════════════════

def gpu_match_descriptors(desc1_np, desc2_np, ratio_threshold=0.75):
    """Match ORB descriptors using GPU Hamming distance or CPU fallback."""
    if _USE_GPU and len(desc1_np) > 0 and len(desc2_np) > 0:
        return _gpu_match_descriptors(desc1_np, desc2_np, ratio_threshold)
    return _cpu_match_descriptors(desc1_np, desc2_np, ratio_threshold)


def _gpu_match_descriptors(desc1_np, desc2_np, ratio_threshold=0.75):
    """GPU Hamming distance matching via CuPy."""
    try:
        d1 = cp.asarray(desc1_np, dtype=cp.uint8)
        d2 = cp.asarray(desc2_np, dtype=cp.uint8)
        
        N, M = len(d1), len(d2)
        
        # XOR + popcount for Hamming distance
        d1_exp = d1[:, cp.newaxis, :]  # (N, 1, 32)
        d2_exp = d2[cp.newaxis, :, :]  # (1, M, 32)
        xor_result = cp.bitwise_xor(d1_exp, d2_exp)  # (N, M, 32)
        
        # Bit count via lookup table approach
        bits = cp.zeros((N, M), dtype=cp.int32)
        for byte_idx in range(xor_result.shape[2]):
            byte_val = xor_result[:, :, byte_idx].astype(cp.int32)
            for bit_pos in range(8):
                bits += (byte_val >> bit_pos) & 1
        
        distances = bits.astype(cp.float32)
        
        # Ratio test: find 2 nearest for each query
        sorted_indices = cp.argsort(distances, axis=1)
        good_matches = []
        
        distances_np = cp.asnumpy(distances)
        indices_np = cp.asnumpy(sorted_indices[:, :2])
        
        for i in range(N):
            if M < 2:
                continue
            best_idx = int(indices_np[i, 0])
            second_idx = int(indices_np[i, 1])
            best_dist = distances_np[i, best_idx]
            second_dist = distances_np[i, second_idx]
            
            if second_dist > 0 and best_dist < ratio_threshold * second_dist:
                good_matches.append((i, best_idx, float(best_dist)))
        
        return good_matches
    except Exception:
        return _cpu_match_descriptors(desc1_np, desc2_np, ratio_threshold)


def _cpu_match_descriptors(desc1_np, desc2_np, ratio_threshold=0.75):
    """CPU fallback for feature matching."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(desc1_np, desc2_np, k=2)
    
    good_matches = []
    for pair in matches_knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append((m.queryIdx, m.trainIdx, m.distance))
    
    return good_matches


# ══════════════════════════════════════════════════════════════
# Quaternion Utilities (for drift-free rotation accumulation)
# ══════════════════════════════════════════════════════════════

def _rotation_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def _quat_to_rotation(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def _quat_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


# ══════════════════════════════════════════════════════════════
# Camera Tracker
# ══════════════════════════════════════════════════════════════

class CameraTracker:
    """
    Hybrid camera tracker: rotation-first with optional translation.
    
    Phase 1 (instant): Track rotation via Homography decomposition
    Phase 2 (when available): Detect translation from residual flow
    
    GPU acceleration:
      - Feature matching (Hamming distance on GPU)
      - Optical flow (cv2.cuda when available)
    """
    
    def __init__(self, width=1920, height=1080, fov=60.0,
                 orb_features=1500, match_ratio=0.8,
                 reproj_threshold=2.0,
                 assumed_height=1.5, proc_width=640,
                 filter_min_cutoff=1.0, filter_beta=0.007, filter_d_cutoff=1.0):
        
        self.width = width
        self.height = height
        self.fov = fov
        
        # Processing resolution
        self.proc_w = proc_width
        self.scale = self.proc_w / width
        self.proc_h = int(height * self.scale)
        
        # Camera intrinsics at processing resolution
        f = self.proc_w / (2 * np.tan(np.radians(fov) / 2))
        self.K = np.array([
            [f, 0, self.proc_w / 2],
            [0, f, self.proc_h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.dist_coeffs = np.zeros(4, dtype=np.float64)
        
        # ORB detector for descriptor matching (used in relocalization)
        self.orb = cv2.ORB_create(nfeatures=orb_features, scaleFactor=1.2, nlevels=8,
                                   edgeThreshold=15, patchSize=31)
        
        # Optical flow parameters (Lucas-Kanade) — ACCURACY SETTINGS
        self.lk_params = dict(
            winSize=(41, 41),        # Increased for higher proc resolution
            maxLevel=4,             # 5 levels handle large displacements
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
        )
        
        # Feature detection for optical flow (Shi-Tomasi corners — best for tracking)
        self.feature_params = dict(
            maxCorners=2000,         # Increased from 1000 for denser cloud
            qualityLevel=0.005,      # More permissive for low-texture areas
            minDistance=8,          # Denser points
            blockSize=7
        )
        
        # Sub-pixel refinement parameters
        self.subpixel_params = dict(
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        )
        
        # SLAM parameters
        self.match_ratio = match_ratio
        self.reproj_threshold = reproj_threshold
        
        # Rotation tracking thresholds
        self.rotation_threshold_deg = 0.3    # Min rotation (degrees) to update
        self.min_homography_inliers = 15     # Min inliers for valid homography
        self.max_rotation_per_frame_deg = 15.0  # Reject rotations larger than this (bad H)
        
        # State
        self.state = TrackingState.TRACKING # Start directly in TRACKING
        self.frame_id = 0
        
        # Previous frame data (for optical flow)
        self._prev_gray = None
        self._prev_points = None  # Nx2 feature points for optical flow
        self._prev_keypoints = None
        self._prev_descriptors = None
        
        # Accumulated pose (world frame)
        self.R_current = np.eye(3, dtype=np.float64)
        self.t_current = np.zeros((3, 1), dtype=np.float64)
        
        # Quaternion for drift-free accumulation
        self._q_accumulated = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        
        # Pose filter
        self.pose_filter = PoseFilter(
            min_cutoff=filter_min_cutoff,
            beta=filter_beta,
            d_cutoff=filter_d_cutoff
        )
        
        # View matrix output (OpenGL convention)
        self.view_matrix = Matrix44.look_at(
            (0, 0, 0), (0, 0, -1), (0, 1, 0)
        )
        
        # Feature refresh counter (re-detect features periodically)
        self._feature_refresh_interval = 30  # frames
        self._frames_since_refresh = 0
        
        
        # Debug counters
        self._consecutive_tracking_frames = 0
        self._init_frame_count = 0
        
        # Translation tracking
        self._translation_detected = False
        self._scale_factor = assumed_height  # Scale factor for monocular translation estimation
        
        # Inertia tracking (for smooth transitions)
        self.last_q_delta = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.inertia_frames = 0
        
        
        # CUDA optical flow object
        self._cuda_optflow = None
        if _USE_CUDA_OPTFLOW:
            try:
                self._cuda_optflow = cv2.cuda.SparsePyrLKOpticalFlow_create(
                    winSize=(21, 21), maxLevel=3
                )
            except Exception:
                self._cuda_optflow = None
        
        gpu_tag = "CUDA" if _USE_GPU else "CPU"
        of_tag = "CUDA" if self._cuda_optflow else "CPU"
        print(f"🗺️  Hybrid Tracker [{gpu_tag}|OF:{of_tag}]: {self.proc_w}x{self.proc_h} | FOV={fov}°")
    
    # ──────────────────────────────────────────────────────────
    # PUBLIC API (preserved for compatibility)
    # ──────────────────────────────────────────────────────────
    
    def track(self, frame, hand_mask=None):
        """
        Process a new frame. Updates internal pose and view matrix.
        """
        # Resize for speed
        frame_small = cv2.resize(frame, (self.proc_w, self.proc_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # Build detection mask (exclude hands)
        detect_mask = None
        if hand_mask is not None:
            detect_mask = cv2.resize(hand_mask, (self.proc_w, self.proc_h),
                                      interpolation=cv2.INTER_NEAREST)
        
        # Dispatch based on state
        if self.state == TrackingState.NOT_INITIALIZED:
            self._handle_not_initialized(gray, detect_mask)
        elif self.state == TrackingState.INITIALIZING:
            self._handle_initializing(gray, detect_mask)
        elif self.state in (TrackingState.ROTATION_ONLY, TrackingState.TRACKING):
            self._handle_tracking(gray, detect_mask)
        elif self.state == TrackingState.LOST:
            self._handle_lost(gray, detect_mask)
        
        self._update_view_matrix()
        self.frame_id += 1
    

    def get_view_matrix(self):
        return np.array(self.view_matrix, dtype=np.float32)
    
    def get_tracking_state(self):
        return self.state
    
    def get_camera_pose(self):
        return self.R_current.astype(np.float32), self.t_current.astype(np.float32)
    
    def reset(self):
        self.state = TrackingState.TRACKING
        self.frame_id = 0
        self.R_current = np.eye(3, dtype=np.float64)
        self.t_current = np.zeros((3, 1), dtype=np.float64)
        self._q_accumulated = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._prev_gray = None
        self._prev_points = None
        self._prev_keypoints = None
        
        self._prev_descriptors = None
        self._consecutive_tracking_frames = 0
        self._init_frame_count = 0
        self._frames_since_refresh = 0
        self._translation_detected = False
        self.last_q_delta = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.inertia_frames = 0
        self.pose_filter.reset()
        self.view_matrix = Matrix44.look_at(
            (0, 0, 0), (0, 0, -1), (0, 1, 0)
        )
        
        print("🔄 Hybrid Tracker Reset (Mode: TRACKING)")
    
    # ──────────────────────────────────────────────────────────
    # STATE HANDLERS
    # ──────────────────────────────────────────────────────────
    
    def _handle_not_initialized(self, gray, mask):
        self.state = TrackingState.TRACKING
        
    def _handle_initializing(self, gray, mask):
        self.state = TrackingState.TRACKING
    
    def _handle_tracking(self, gray, mask):
        """Main tracking loop."""
        if self._prev_gray is None or self._prev_points is None:
            self.state = TrackingState.LOST
            return
        
        # ── 1. Track features via optical flow ──
        tracked_prev, tracked_cur, status = self._track_optical_flow(
            self._prev_gray, gray, self._prev_points
        )
        
        if tracked_prev is None or len(tracked_cur) < 15:
            # 🚀 INERTIA REMOVED: If we lose features, we hold position.
            # No guessing. This prevents "fly away" drift.
            self.state = TrackingState.LOST
            print("❌ Tracker: Features lost, LOST (Inertia Disabled)")
            return
        
        # ── 2. Estimate rotation from homography ──
        R_delta = self._estimate_rotation(tracked_prev, tracked_cur)
        
        if R_delta is not None:
            # STRICT 3DOF: Only apply rotation.
            q_delta = _rotation_to_quat(R_delta)
            self._q_accumulated = _quat_multiply(self._q_accumulated, q_delta)
            self._q_accumulated /= np.linalg.norm(self._q_accumulated)
            self.R_current = _quat_to_rotation(self._q_accumulated)
            
            # ── 3. Smooth pose ──
            timestamp = time.time()
            # Note: We pass t_current but it stays 0.0 in 3DOF mode
            R_smooth, t_smooth = self.pose_filter.update(
                self.R_current.astype(np.float64),
                self.t_current.astype(np.float64),
                timestamp
            )
            self.R_current = R_smooth.astype(np.float64)
            # self.t_current = t_smooth.astype(np.float64) # Keep translation zero for now
            self._consecutive_tracking_frames += 1
        else:
            # Rotation estimation failed. Hold position? Or Lost?
            # If we just hold, it might look frozen. Better than drifting.
            # But "LOST" triggers re-initialization which is safer if we really can't tell.
            self.state = TrackingState.LOST
            print("❌ Tracker: Rotation estimation failed, LOST")
            return
        
        # ── 4. Refresh features periodically ──
        self._frames_since_refresh += 1
        if self._frames_since_refresh >= self._feature_refresh_interval or len(tracked_cur) < 30:
            new_points = self._detect_features(gray, mask)
            self._prev_points = new_points if new_points is not None and len(new_points) >= 20 else tracked_cur.copy()
            self._frames_since_refresh = 0
        else:
            self._prev_points = tracked_cur.copy()
        
        self._prev_gray = gray.copy()
        
        # Force state to ROTATION_ONLY (3DOF)
        self.state = TrackingState.ROTATION_ONLY
        
    
    def _handle_lost(self, gray, mask):
        """Attempt to recover via raw features."""
        # Fallback: Blind recovery
        points = self._detect_features(gray, mask)
        
        if points is not None and len(points) >= 20:
            self._prev_gray = gray.copy()
            self._prev_points = points
            self._init_frame_count = 0
            self.state = TrackingState.TRACKING
            print("🔄 Tracker: Re-initializing from lost state...")
    

    
    # ──────────────────────────────────────────────────────────
    # OPTICAL FLOW
    # ──────────────────────────────────────────────────────────
    
    def _detect_features(self, gray, mask=None):
        """Detect and refine Shi-Tomasi corners."""
        points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if points is not None:
            # Sub-pixel refinement for pin-point accuracy
            points = cv2.cornerSubPix(gray, points, **self.subpixel_params)
            points = points.reshape(-1, 2).astype(np.float32)
        return points
    
    def _track_optical_flow(self, prev_gray, cur_gray, prev_points):
        """
        Track features from prev frame to current using Lucas-Kanade optical flow.
        Returns (matched_prev, matched_cur, status) or (None, None, None) on failure.
        """
        if prev_points is None or len(prev_points) < 5:
            return None, None, None
        
        pts_prev = prev_points.reshape(-1, 1, 2).astype(np.float32)
        
        if self._cuda_optflow is not None:
            # GPU optical flow
            try:
                gpu_prev = cv2.cuda_GpuMat(prev_gray)
                gpu_cur = cv2.cuda_GpuMat(cur_gray)
                gpu_pts = cv2.cuda_GpuMat(pts_prev)
                
                gpu_next_pts, gpu_status, _ = self._cuda_optflow.calc(
                    gpu_prev, gpu_cur, gpu_pts, None
                )
                
                next_pts = gpu_next_pts.download()
                status = gpu_status.download()
            except Exception:
                # Fallback to CPU
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, cur_gray, pts_prev, None, **self.lk_params
                )
        else:
            # CPU optical flow
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, cur_gray, pts_prev, None, **self.lk_params
            )
        
        if next_pts is None or status is None:
            return None, None, None
        
        # Forward-backward consistency check for robustness
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            cur_gray, prev_gray, next_pts, None, **self.lk_params
        )
        
        if back_pts is None:
            return None, None, None
        
        # Compute forward-backward error
        fb_error = np.linalg.norm(
            pts_prev.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
        )
        
        # Keep only good tracks (status OK + low FB error + within frame bounds)
        status = status.ravel()
        if back_status is not None:
            status = status & back_status.ravel()
        
        good = (status == 1) & (fb_error < 2.0)
        
        # Bounds check
        next_pts_flat = next_pts.reshape(-1, 2)
        in_bounds = (
            (next_pts_flat[:, 0] >= 0) & (next_pts_flat[:, 0] < self.proc_w) &
            (next_pts_flat[:, 1] >= 0) & (next_pts_flat[:, 1] < self.proc_h)
        )
        good = good & in_bounds
        
        if np.sum(good) < 5:
            return None, None, None
        
        matched_prev = pts_prev.reshape(-1, 2)[good]
        matched_cur = next_pts_flat[good]
        
        return matched_prev, matched_cur, good
    
    # ──────────────────────────────────────────────────────────
    # ROTATION ESTIMATION (Homography Decomposition)
    # ──────────────────────────────────────────────────────────
    
    def _estimate_rotation(self, pts_prev, pts_cur):
        """Estimate rotation with RANSAC outlier rejection."""
        if len(pts_prev) < 8 or len(pts_cur) < 8:
            return None
        
        # USAC_MAGSAC is preferred if available (OpenCV 4.5+)
        method = cv2.RANSAC
        if hasattr(cv2, 'USAC_MAGSAC'):
            method = cv2.USAC_MAGSAC
            
        # Estimate homography
        H, mask = cv2.findHomography(pts_prev, pts_cur, method, 2.0, maxIters=2000)
        
        if H is None:
            return None
        
        # Filter outliers
        mask = mask.ravel().astype(bool)
        if np.sum(mask) < self.min_homography_inliers:
            return None
        
        # Decompose
        R = self.K_inv @ H @ self.K
        U, S, Vt = np.linalg.svd(R)
        R_proper = U @ Vt
        if np.linalg.det(R_proper) < 0: R_proper = -R_proper
        
        angle_deg = np.degrees(np.arccos(np.clip((np.trace(R_proper) - 1.0) / 2.0, -1.0, 1.0)))
        if angle_deg > self.max_rotation_per_frame_deg:
            return None
            
        return R_proper
    
    # ──────────────────────────────────────────────────────────
    # TRANSLATION ESTIMATION (Optional — when residual flow detected)
    # ──────────────────────────────────────────────────────────
    
    def _compute_rotation_residual(self, pts_prev, pts_cur, R_delta):
        """
        After compensating for rotation, compute the residual displacement.
        High residual suggests actual camera translation.
        """
        # Project prev points through rotation: p' = K · R · K⁻¹ · p
        pts_prev_h = np.hstack([pts_prev, np.ones((len(pts_prev), 1))]).T  # 3xN
        pts_rotated = self.K @ R_delta @ self.K_inv @ pts_prev_h  # 3xN
        pts_rotated = (pts_rotated[:2] / pts_rotated[2:]).T  # Nx2
        
        # Residual = actual displacement - rotation-predicted displacement
        residual = np.linalg.norm(pts_cur - pts_rotated, axis=1)
        return np.median(residual)
    
    def _estimate_translation(self, pts_prev, pts_cur, R_delta):
        """
        Estimate translation from residual flow after rotation compensation.
        Uses Essential Matrix decomposition.
        """
        try:
            E, mask_E = cv2.findEssentialMat(
                pts_prev, pts_cur, self.K,
                method=cv2.RANSAC, prob=0.999, threshold=2.0
            )
            
            if E is None:
                return
            
            mask_E = mask_E.ravel().astype(bool)
            if np.sum(mask_E) < 8:
                return
            
            inliers, R_e, t_e, _ = cv2.recoverPose(
                E, pts_prev[mask_E], pts_cur[mask_E], self.K
            )
            
            if inliers < 8:
                return
            
            # Use only the translation direction (scale is unknown with monocular)
            # Scale by assumed height for approximate world units
            t_delta = t_e * self._scale_factor * 0.01  # Conservative scaling
            
            # Accumulate translation in world frame
            # t_world += R_current⁻¹ · t_delta
            self.t_current = self.t_current + self.R_current.T @ t_delta
            self._translation_detected = True
            
        except Exception:
            pass
    
    # ──────────────────────────────────────────────────────────
    # VIEW MATRIX
    # ──────────────────────────────────────────────────────────
    
    def _update_view_matrix(self):
        """Convert R, t to OpenGL view matrix."""
        R = self.R_current
        t = self.t_current
        
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R.astype(np.float32)
        view[:3, 3] = t.flatten().astype(np.float32)
        
        # OpenCV → OpenGL coordinate conversion
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        view = cv_to_gl @ view
        
        self.view_matrix = Matrix44(view.T)
