"""
Temporal filtering and smoothing
Phase 2: Anti-flicker and noise reduction
"""


class TemporalFilter:
    """
    Temporal smoothing filter to reduce gesture flicker
    Requires signal to be stable for N frames before confirming
    """
    
    def __init__(self, threshold_frames=2):
        """
        Initialize temporal filter
        
        Args:
            threshold_frames: Number of consecutive frames needed to confirm state
        """
        self.threshold_frames = threshold_frames
        self.true_count = 0
        self.false_count = 0
        self.current_state = False
    
    def update(self, new_value):
        """
        Update filter with new boolean value
        
        Args:
            new_value: Boolean input signal
        
        Returns:
            bool: Filtered output (confirmed state)
        """
        if new_value:
            self.true_count += 1
            self.false_count = 0
            
            # Confirm TRUE if stable for threshold frames
            if self.true_count >= self.threshold_frames:
                self.current_state = True
        else:
            self.false_count += 1
            self.true_count = 0
            
            # Confirm FALSE if stable for threshold frames
            if self.false_count >= self.threshold_frames:
                self.current_state = False
        
        return self.current_state
    
    def get_state(self):
        """Get current filtered state"""
        return self.current_state
    
    def reset(self):
        """Reset filter"""
        self.true_count = 0
        self.false_count = 0
        self.current_state = False


class HysteresisFilter:
    """
    Hysteresis filter with different thresholds for on/off
    Prevents rapid toggling near threshold boundary
    """
    
    def __init__(self, low_threshold, high_threshold):
        """
        Initialize hysteresis filter
        
        Args:
            low_threshold: Threshold to turn OFF
            high_threshold: Threshold to turn ON
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.state = False
    
    def update(self, value):
        """
        Update filter with new value
        
        Args:
            value: Input value to compare against thresholds
        
        Returns:
            bool: Filtered state
        """
        if value < self.low_threshold:
            self.state = True  # Below low = pinch active
        elif value > self.high_threshold:
            self.state = False  # Above high = pinch inactive
        # Between thresholds = keep current state
        
        return self.state
    
    def get_state(self):
        """Get current state"""
        return self.state
