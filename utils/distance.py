"""
Distance calculation utilities
Phase 2: Gesture detection math
"""
import math
import numpy as np


def euclidean_distance_2d(point1, point2):
    """
    Calculate 2D Euclidean distance between two points
    
    Args:
        point1: (x, y) or (x, y, z)
        point2: (x, y) or (x, y, z)
    
    Returns:
        float: Distance
    """
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def euclidean_distance_3d(point1, point2):
    """
    Calculate 3D Euclidean distance
    
    Args:
        point1: (x, y, z)
        point2: (x, y, z)
    
    Returns:
        float: Distance
    """
    x1, y1, z1 = point1[0], point1[1], point1[2]
    x2, y2, z2 = point2[0], point2[1], point2[2]
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance


def normalized_distance(point1, point2, use_z=False):
    """
    Calculate distance using normalized coordinates (0-1 range)
    Resolution independent
    
    Args:
        point1: Dictionary with 'x', 'y', 'z' keys (0-1 normalized)
        point2: Dictionary with 'x', 'y', 'z' keys
        use_z: Include Z-axis in calculation
    
    Returns:
        float: Normalized distance
    """
    if use_z:
        return euclidean_distance_3d(
            (point1['x'], point1['y'], point1['z']),
            (point2['x'], point2['y'], point2['z'])
        )
    else:
        return euclidean_distance_2d(
            (point1['x'], point1['y']),
            (point2['x'], point2['y'])
        )


def pixel_distance(point1, point2, use_z=False):
    """
    Calculate distance using pixel coordinates
    
    Args:
        point1: Dictionary with 'x', 'y', 'z' keys (pixel values)
        point2: Dictionary with 'x', 'y', 'z' keys
        use_z: Include Z-axis
    
    Returns:
        float: Pixel distance
    """
    if use_z:
        return euclidean_distance_3d(
            (point1['x'], point1['y'], point1['z']),
            (point2['x'], point2['y'], point2['z'])
        )
    else:
        return euclidean_distance_2d(
            (point1['x'], point1['y']),
            (point2['x'], point2['y'])
        )


def calculate_hand_scale(wrist, middle_mcp):
    """
    Calculate hand scale for dynamic thresholding
    Uses distance between wrist and middle finger knuckle
    
    Args:
        wrist: Wrist landmark (dict or tuple)
        middle_mcp: Middle finger MCP landmark
    
    Returns:
        float: Hand scale distance
    """
    if isinstance(wrist, dict):
        return normalized_distance(wrist, middle_mcp, use_z=False)
    else:
        return euclidean_distance_2d(wrist, middle_mcp)


def adaptive_threshold(hand_scale, multiplier=0.25, min_thresh=0.02, max_thresh=0.15):
    """
    Calculate adaptive pinch threshold based on hand size
    
    Args:
        hand_scale: Distance representing hand size
        multiplier: Scaling factor
        min_thresh: Minimum threshold
        max_thresh: Maximum threshold
    
    Returns:
        float: Adaptive threshold value
    """
    threshold = hand_scale * multiplier
    
    # Clamp between min and max
    threshold = max(min_thresh, min(max_thresh, threshold))
    
    return threshold
