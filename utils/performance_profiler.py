
import time
import cv2
import numpy as np

class PerformanceProfiler:
    def __init__(self):
        self.frame_times = []
        self.section_times = {}
        self.current_section_start = None
        self.current_section_name = None
        self.max_samples = 60
        
    def start_frame(self):
        """Start timing a new frame"""
        self.frame_start = time.time()
        self.section_times = {}
        
    def start_section(self, name):
        """Start timing a section"""
        if self.current_section_name:
            self.end_section()
        self.current_section_name = name
        self.current_section_start = time.time()
        
    def end_section(self):
        """End timing current section"""
        if self.current_section_name:
            elapsed = (time.time() - self.current_section_start) * 1000  # ms
            self.section_times[self.current_section_name] = elapsed
            self.current_section_name = None
            
    def end_frame(self):
        """End frame and record time"""
        if self.current_section_name:
            self.end_section()
            
        frame_time = (time.time() - self.frame_start) * 1000  # ms
        self.frame_times.append(frame_time)
        # Keep only recent samples
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)
    
    def get_fps(self):
        """Get current FPS"""
        if not self.frame_times:
            return 0.0
        avg_frame_time = np.mean(self.frame_times)
        if avg_frame_time == 0:
            return 0.0
        return 1000.0 / avg_frame_time
    
    def get_frame_time(self):
        """Get average frame time in ms"""
        if not self.frame_times:
            return 0.0
        return np.mean(self.frame_times)
    
    def draw_stats(self, frame):
        """Draw FPS and timing stats on frame"""
        fps = self.get_fps()
        frame_time = self.get_frame_time()
        
        # Background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # FPS
        color = (0, 255, 0) if fps >= 30 else (0, 165, 255) if fps >= 20 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Frame time
        cv2.putText(frame, f"Frame: {frame_time:.1f}ms", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Section times
        y = 95
        for section, time_ms in self.section_times.items():
            text = f"{section}: {time_ms:.1f}ms"
            cv2.putText(frame, text, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
            if y > 140:
                break
                
        return frame
