import cv2
import numpy as np
import math
import time

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the OneEuroFilter."""
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev  # Avoid divide by zero

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class PointSmoother:
    """Wrapper for 2D point smoothing using OneEuroFilter"""
    def __init__(self, min_cutoff=0.004, beta=80.0):
        # Tuned for hand tracking cursor:
        # min_cutoff=0.004 -> Very stable when still (anti-jitter)
        # beta=80.0 -> Fast catch-up when moving quickly (low lag)
        self.f_x = OneEuroFilter(time.time(), 0, min_cutoff=min_cutoff, beta=beta)
        self.f_y = OneEuroFilter(time.time(), 0, min_cutoff=min_cutoff, beta=beta)
        self.first_update = True
        
    def update(self, x, y):
        t = time.time()
        if self.first_update:
            self.f_x = OneEuroFilter(t, x, min_cutoff=0.004, beta=80.0)
            self.f_y = OneEuroFilter(t, y, min_cutoff=0.004, beta=80.0)
            self.first_update = False
            return x, y
            
        sx = self.f_x.filter(t, x)
        sy = self.f_y.filter(t, y)
        return sx, sy

class KalmanSmoother:
    """
    Kalman Filter for zero-lag predictive smoothing.
    Model: Constant Velocity
    State: [x, y, dx, dy]
    Measurement: [x, y]
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)
        # Process Noise Covariance (Q) - How much we trust the model
        # Lower = smoother, Higher = faster reaction
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement Noise Covariance (R) - How much noise in sensor
        # Higher = more smoothing
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0 # Increased for stability
        
        self.first_update = True
        self.last_time = time.time()

    def update(self, x, y):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if self.first_update:
            # Initialize state at first measurement
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0  # High initial uncertainty
            self.first_update = False
            return x, y

        # Adapt transition matrix to time delta
        self.kf.transitionMatrix[0, 2] = dt * 30  # Scale velocity to ~30fps units
        self.kf.transitionMatrix[1, 3] = dt * 30

        # Predict phase
        prediction = self.kf.predict()
        
        # Correct phase
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        
        # Return predicted state (smoother) or corrected state (accurate)
        # Using corrected state avoids "overshooting" on sudden stops
        px, py = self.kf.statePost[0], self.kf.statePost[1]
        return float(px), float(py)
