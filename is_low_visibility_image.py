
import cv2
import numpy as np
def is_low_visibility_image(img, threshold=0.85):
    """
    Determines if an image has low visibility due to fog, heavy snow, etc.
    
    Args:
        img: Input image (BGR format)
        threshold: Whiteness threshold (0-1). Higher means more sensitive to white.
        
    Returns:
        bool: True if the image has low visibility, False otherwise
        float: Whiteness score of the image (0-1)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the average brightness
    avg_brightness = np.mean(gray) / 255.0
    
    # Calculate histogram to check for concentration in bright areas
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    
    # Calculate the weighted sum (more weight to brighter pixels)
    bright_weight = np.sum(hist[200:] * np.arange(200, 256) / 255.0)
    
    # Calculate a "whiteness score" combining brightness and concentration
    whiteness_score = (avg_brightness * 0.6) + (bright_weight * 0.4)
    
    # Check variance - low variance is another indicator of fog/snow
    variance = np.var(gray) / (255.0 * 255.0)
    low_variance = variance < 0.05
    
    # Image is low visibility if it's very white and has low variance
    is_low_vis = (whiteness_score > threshold) or (whiteness_score > 0.75 and low_variance)
    
    return is_low_vis, whiteness_score
