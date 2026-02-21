"""
Gesture State Machine
Phase 2: Convert raw pinch signals into interaction states
"""
from enum import Enum


class GestureState(Enum):
    """Gesture states for pinch interaction"""
    OPEN = "OPEN"              # No gesture
    PINCH_START = "START"      # Pinch just detected → Click event
    PINCH_HOLD = "HOLD"        # Pinch maintained → Drag
    PINCH_RELEASE = "RELEASE"  # Pinch released → Drop event


class GestureStateMachine:
    """
    State machine for gesture recognition
    Converts boolean pinch signal into meaningful interaction states
    """
    
    def __init__(self):
        """Initialize state machine"""
        self.current_state = GestureState.OPEN
        self.previous_state = GestureState.OPEN
    
    def update(self, pinch_active):
        """
        Update state machine with pinch signal
        
        Args:
            pinch_active: Boolean indicating pinch detection
        
        Returns:
            GestureState: Current gesture state
        """
        self.previous_state = self.current_state
        
        # State transition logic
        if self.current_state == GestureState.OPEN:
            if pinch_active:
                self.current_state = GestureState.PINCH_START
        
        elif self.current_state == GestureState.PINCH_START:
            if pinch_active:
                self.current_state = GestureState.PINCH_HOLD
            else:
                self.current_state = GestureState.OPEN
        
        elif self.current_state == GestureState.PINCH_HOLD:
            if not pinch_active:
                self.current_state = GestureState.PINCH_RELEASE
        
        elif self.current_state == GestureState.PINCH_RELEASE:
            self.current_state = GestureState.OPEN
        
        return self.current_state
    
    def get_state(self):
        """Get current state"""
        return self.current_state
    
    def get_state_name(self):
        """Get current state name as string"""
        return self.current_state.value
    
    def is_state_changed(self):
        """Check if state changed in last update"""
        return self.current_state != self.previous_state
    
    def is_click_event(self):
        """Check if this is a click event (START state)"""
        return self.current_state == GestureState.PINCH_START
    
    def is_dragging(self):
        """Check if currently dragging (HOLD state)"""
        return self.current_state == GestureState.PINCH_HOLD
    
    def is_release_event(self):
        """Check if this is a release event"""
        return self.current_state == GestureState.PINCH_RELEASE
    
    def reset(self):
        """Reset state machine"""
        self.current_state = GestureState.OPEN
        self.previous_state = GestureState.OPEN
