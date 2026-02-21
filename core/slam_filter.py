"""
One-Euro Filter for 6DOF Pose Smoothing
Smooths camera translation (3D) and rotation (quaternion) to eliminate jitter.

Reference: https://cristal.univ-lille.fr/~casiez/1euro/
"""

import numpy as np
import time


class OneEuroFilter1D:
    """One-Euro filter for a single scalar value."""
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))
    
    def update(self, x, t=None):
        if t is None:
            t = time.time()
        
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        
        dt = max(t - self.t_prev, 1e-6)
        
        # Derivative estimation
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filtering
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class PoseFilter:
    """
    One-Euro filter for 6DOF camera pose.
    Smooths translation (3 axes) and rotation (via quaternion).
    """
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        # 3 filters for translation (x, y, z)
        self.t_filters = [
            OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(3)
        ]
        # 4 filters for quaternion (w, x, y, z)
        self.q_filters = [
            OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(4)
        ]
        self.last_quat = None
    
    def _rotation_to_quat(self, R):
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
    
    def _quat_to_rotation(self, q):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
            [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
            [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)]
        ], dtype=np.float32)
        return R
    
    def update(self, R, t, timestamp=None):
        """
        Smooth a 6DOF pose.
        
        Args:
            R: 3x3 rotation matrix (numpy float32)
            t: 3x1 or (3,) translation vector (numpy float32)
            timestamp: time in seconds (float). If None, uses time.time().
        
        Returns:
            R_smooth: 3x3 rotation matrix
            t_smooth: 3x1 translation vector
        """
        if timestamp is None:
            timestamp = time.time()
        
        t_flat = t.flatten()
        
        # Smooth translation
        t_smooth = np.array([
            self.t_filters[i].update(float(t_flat[i]), timestamp)
            for i in range(3)
        ], dtype=np.float32).reshape(3, 1)
        
        # Smooth rotation via quaternion
        q = self._rotation_to_quat(R)
        
        # Ensure quaternion continuity (avoid sign flips)
        if self.last_quat is not None:
            if np.dot(q, self.last_quat) < 0:
                q = -q
        
        q_smooth = np.array([
            self.q_filters[i].update(float(q[i]), timestamp)
            for i in range(4)
        ], dtype=np.float64)
        
        # Renormalize quaternion
        norm = np.linalg.norm(q_smooth)
        if norm > 1e-8:
            q_smooth /= norm
        else:
            q_smooth = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.last_quat = q_smooth
        R_smooth = self._quat_to_rotation(q_smooth)
        
        return R_smooth, t_smooth
    
    def reset(self):
        """Reset all filter states."""
        for f in self.t_filters:
            f.reset()
        for f in self.q_filters:
            f.reset()
        self.last_quat = None
