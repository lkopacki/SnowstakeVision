
from is_low_visibility_image import is_low_visibility_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def detect_red_black_stakes(image_path, output_dir=None, visibility_threshold=0.85):
    """
    Detect a single red and black stake in an image with trees in foreground and background.
    Calculates snow depth based on the visible portion of the stake.
    
    The stake has the following properties:
    - Total height: 150 cm
    - Alternating red and black bands of 10 cm each
    - Small red section at the very top, followed by a black band
    - The red color is approximately #ff4276, but may vary with lighting
    
    Args:
        image_path (str): Path to the input image
        output_dir (str, optional): Directory to save output images. If None, images are displayed only.
        visibility_threshold (float): Threshold for detecting low visibility conditions (0-1)
        
    Returns:
        tuple: (Original image, processed image with stakes highlighted, stakes list, snow depth measurements, 
               is_low_visibility flag, whiteness score)
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Check for low visibility conditions
    is_low_visibility, whiteness_score = is_low_visibility_image(img, visibility_threshold)
    
    # Convert from BGR to RGB for display purposes
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a copy for visualization
    result_img = img_rgb.copy()
    
    # If low visibility, annotate the image and return early
    if is_low_visibility:
        cv2.putText(result_img, f"Low Visibility Detected: {whiteness_score:.2f}", 
                   (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.putText(result_img, "Fog or Heavy Precipitation", 
                   (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.putText(result_img, "Unable to reliably detect stakes", 
                   (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        
        # If output directory is specified, save the image
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            base_name = Path(image_path).stem
            
            plt.figure(figsize=(12, 12))
            plt.imshow(result_img)
            plt.title("Low Visibility - Analysis Skipped")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / f"{base_name}_low_visibility.jpg")
            plt.close()
            
        return img_rgb, result_img, [], [], is_low_visibility, whiteness_score
    
    # Step 1: Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Step 2: Convert to different color spaces for better color detection
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    
    # The target red color #ff4276 in BGR format is approximately (118, 66, 255)
    # Let's create a mask specifically targeting this red color with a tolerance range
    
    # For HSV: Convert target RGB color to HSV
    target_rgb = np.uint8([[[0x42, 0x76, 0xff]]])  # BGR order for OpenCV
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2HSV)
    target_h = target_hsv[0][0][0]  # Extract the Hue value
    
    # Define a broader range around the target hue to account for lighting variation
    # Red is tricky in HSV since it wraps around 0/180 in OpenCV's representation
    hue_tolerance = 20  # Increased from 15 to capture more red variations
    sat_min = 80  # Reduced from 100 to capture more of the red areas that might be washed out
    val_min = 80  # Reduced from 100 for the same reason
    
    # Create red masks around the target color
    if target_h - hue_tolerance < 0:
        # If the range wraps around 0, we need two masks
        lower_red1 = np.array([0, sat_min, val_min])
        upper_red1 = np.array([target_h + hue_tolerance, 255, 255])
        lower_red2 = np.array([180 + (target_h - hue_tolerance), sat_min, val_min])
        upper_red2 = np.array([180, 255, 255])
    elif target_h + hue_tolerance > 180:
        # If the range wraps around 180, we need two masks
        lower_red1 = np.array([0, sat_min, val_min])
        upper_red1 = np.array([(target_h + hue_tolerance) - 180, 255, 255])
        lower_red2 = np.array([target_h - hue_tolerance, sat_min, val_min])
        upper_red2 = np.array([180, 255, 255])
    else:
        # Standard case - no wrapping
        lower_red1 = np.array([max(0, target_h - hue_tolerance), sat_min, val_min])
        upper_red1 = np.array([min(180, target_h + hue_tolerance), 255, 255])
        lower_red2 = np.array([0, 0, 0])  # Dummy values
        upper_red2 = np.array([0, 0, 0])  # Dummy values
    
    # Also try in RGB space using a direct color distance approach
    img_rgb_cv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    target_rgb_direct = np.array([0xff, 0x42, 0x76])  # RGB format
    rgb_distance = np.sqrt(np.sum((img_rgb_cv.astype(np.float32) - target_rgb_direct) ** 2, axis=2))
    rgb_mask = (rgb_distance < 100).astype(np.uint8) * 255  # Increased from 80 to capture more red
    
    # Combine multiple color detection approaches
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    if lower_red2[0] != 0 or upper_red2[0] != 0:  # If we have a valid second range
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask_hsv = cv2.bitwise_or(red_mask1, red_mask2)
    else:
        red_mask_hsv = red_mask1
    
    # Combine HSV and RGB masks
    red_mask = cv2.bitwise_or(red_mask_hsv, rgb_mask)
    
    # Apply morphological operations to enhance the mask
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Create a black mask to help identify black bands
    # For black parts, we'll detect dark areas with low value in HSV
    _, black_mask = cv2.threshold(hsv[:,:,2], 60, 255, cv2.THRESH_BINARY_INV)  # Increased from 50
    
    # Apply morphological operations to remove small noise
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Use Hough line detection to find potential straight lines for slanted stakes
    edges = cv2.Canny(red_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=30)
    
    # Store line angles to help identify stake orientation
    line_angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line angle (in degrees, 0 = horizontal, 90 = vertical)
            if x2 != x1:  # Avoid division by zero
                angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # We're interested in near-vertical lines (75-105 degrees)
                if 75 <= angle <= 105:
                    line_angles.append(angle - 90)  # Normalize so 0 = vertical
    
    # Determine predominant stake angle
    stake_angle = 0  # Default to vertical if no lines found
    if line_angles:
        stake_angle = np.median(line_angles)  # Use median to filter outliers
    
    # Find contours in the red mask - these will help us locate potential stakes
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find the most likely stake segments
    red_segments = []
    min_area = 80  # Reduced from 100 to capture smaller segments
    
    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # More lenient with height/width ratio to account for slanted stakes
            red_segments.append((x, y, w, h, area))
    
    # Sort segments by y-coordinate (top to bottom)
    red_segments.sort(key=lambda x: x[1])
    
    # Now we'll try to find entire stakes by grouping segments, accounting for the estimated angle
    stake_candidates = []
    
    # Group segments that could be part of the same stake, considering the predominant angle
    for i, (x1, y1, w1, h1, area1) in enumerate(red_segments):
        stake_segments = [(x1, y1, w1, h1, area1)]
        center_x1 = x1 + w1 // 2
        center_y1 = y1 + h1 // 2
        
        # Calculate the expected x-offset based on the predominant angle
        for j, (x2, y2, w2, h2, area2) in enumerate(red_segments):
            if i != j:
                center_x2 = x2 + w2 // 2
                center_y2 = y2 + h2 // 2
                
                # Calculate expected x-offset based on y-difference and estimated angle
                y_diff = center_y2 - center_y1
                expected_x_offset = y_diff * np.tan(np.radians(stake_angle))
                expected_x = center_x1 + expected_x_offset
                
                # Allow for some tolerance around the expected position
                x_tolerance = max(w1, w2) * 2.0  # Increased tolerance for slanted stakes
                
                if abs(center_x2 - expected_x) < x_tolerance:
                    # Check if the segment is not too distant vertically
                    vertical_gap = min(abs(y2 - (y1 + h1)), abs(y1 - (y2 + h2)))
                    max_gap = max(h1, h2) * 5  # Increased to allow larger gaps between segments
                    
                    if vertical_gap < max_gap:
                        stake_segments.append((x2, y2, w2, h2, area2))
        
        # If we found multiple segments for this stake
        if len(stake_segments) > 1:
            # Sort the segments by y-coordinate
            stake_segments.sort(key=lambda x: x[1])
            
            # Calculate the bounding box that encompasses all segments
            min_x = min(x for x, _, w, _, _ in stake_segments)
            min_y = min(y for _, y, _, _, _ in stake_segments)
            max_x = max(x + w for x, _, w, _, _ in stake_segments)
            max_y = max(y + h for _, y, _, h, _ in stake_segments)
            
            # Store the stake candidate with its bounding box and segment count
            stake_width = max_x - min_x
            stake_height = max_y - min_y
            stake_candidates.append((min_x, min_y, stake_width, stake_height, len(stake_segments), stake_segments))
    
    # If no multi-segment candidates, try single red segments as candidates
    if not stake_candidates and red_segments:
        for x, y, w, h, area in red_segments:
            ratio = h / w if w > 0 else 0
            if ratio > 1.2:  # Less strict ratio than before (was 1.5)
                stake_candidates.append((x, y, w, h, 1, [(x, y, w, h, area)]))
    
    # Try to use the black mask to find additional segments that might have been missed
    # This is especially helpful for finding the full stake length
    if stake_candidates:
        x, y, w, h, segment_count, segments = stake_candidates[0]
        
        # Expand the search area around the candidate
        search_x = max(0, x - w)
        search_width = min(black_mask.shape[1] - search_x, w * 3)
        
        # Look above the detected stake for more segments
        search_top = max(0, y - h * 2)  # Look up to 2x the height above
        
        # Extract region of interest
        top_roi_black = black_mask[search_top:y, search_x:search_x+search_width]
        top_roi_red = red_mask[search_top:y, search_x:search_x+search_width]
        
        # Find contours in this region
        black_contours, _ = cv2.findContours(top_roi_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        additional_red_contours, _ = cv2.findContours(top_roi_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the top center of the detected stake
        stake_top_x = x + w // 2
        stake_top_y = y
        
        # Process additional segments above the stake
        additional_segments = []
        
        # Process black contours above the stake
        for contour in black_contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                bx, by, bw, bh = cv2.boundingRect(contour)
                # Adjust coordinates to the full image
                bx += search_x
                by += search_top
                
                # Check if this could be part of the stake (aligned with the expected angle)
                center_bx = bx + bw // 2
                expected_x_at_by = stake_top_x + (by - stake_top_y) * np.tan(np.radians(stake_angle))
                
                if abs(center_bx - expected_x_at_by) < bw * 2:
                    # This could be part of the stake
                    additional_segments.append((bx, by, bw, bh, area))
        
        # Process additional red contours above the stake
        for contour in additional_red_contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                rx, ry, rw, rh = cv2.boundingRect(contour)
                # Adjust coordinates to the full image
                rx += search_x
                ry += search_top
                
                # Check if this could be part of the stake
                center_rx = rx + rw // 2
                expected_x_at_ry = stake_top_x + (ry - stake_top_y) * np.tan(np.radians(stake_angle))
                
                if abs(center_rx - expected_x_at_ry) < rw * 2:
                    # This could be part of the stake
                    additional_segments.append((rx, ry, rw, rh, area))
        
        # Add additional segments to the stake
        if additional_segments:
            segments.extend(additional_segments)
            
            # Recalculate the bounding box
            min_x = min(s[0] for s in segments)
            min_y = min(s[1] for s in segments)
            max_x = max(s[0] + s[2] for s in segments)
            max_y = max(s[1] + s[3] for s in segments)
            
            # Update the stake candidate
            stake_candidates[0] = (min_x, min_y, max_x - min_x, max_y - min_y, len(segments), segments)
    
    # Now, rank the candidates based on multiple factors:
    # 1. Number of segments (more is better)
    # 2. Total height (taller is better)
    # 3. Presence of alternating red/black pattern
    
    # First, filter out candidates that are too short or too fat
    min_height_ratio = 2.0  # Reduced from 3 to account for slanted stakes
    filtered_candidates = []
    
    for x, y, w, h, segment_count, segments in stake_candidates:
        ratio = h / w if w > 0 else 0
        if ratio >= min_height_ratio:
            # Calculate a score based on height and segment count
            height_score = h
            segment_score = segment_count * 50  # Give significant weight to segment count
            
            # Check for alternating pattern using the black mask
            pattern_score = 0
            if h > 20:  # Only check pattern for tall enough candidates
                # Expand the bounding box slightly to account for alignment issues
                expanded_x = max(0, x - 5)
                expanded_w = min(black_mask.shape[1] - expanded_x, w + 10)
                
                # Account for the slant - sample along the stake angle
                sample_points = 15  # Increased from 10 for more samples
                black_samples = []
                
                for i in range(sample_points):
                    # Calculate sampling position along the angled line
                    sample_y = y + (i * h // sample_points)
                    if sample_y < black_mask.shape[0]:
                        # Calculate x-offset based on angle
                        x_offset = int((sample_y - y) * np.tan(np.radians(stake_angle)))
                        sample_x = x + w // 2 + x_offset
                        
                        # Make sure we're within image bounds
                        if 0 <= sample_x < black_mask.shape[1]:
                            # Sample in a small window around the point
                            window_size = max(w // 2, 5)
                            x_start = max(0, sample_x - window_size // 2)
                            x_end = min(black_mask.shape[1], sample_x + window_size // 2)
                            
                            # Calculate the percentage of black in this window
                            slice_black = np.mean(black_mask[sample_y, x_start:x_end]) > 127
                            black_samples.append(slice_black)
                
                # Count transitions between black and non-black (indicating alternating pattern)
                transitions = sum(1 for i in range(1, len(black_samples)) if black_samples[i] != black_samples[i-1])
                pattern_score = transitions * 100  # Give high weight to alternating pattern
            
            total_score = height_score + segment_score + pattern_score
            filtered_candidates.append((x, y, w, h, segment_count, segments, total_score))
    
    # Sort candidates by total score (higher is better)
    filtered_candidates.sort(key=lambda x: x[6], reverse=True)
    
    # Initialize lists
    stakes = []
    snow_depths = []
    bands = []  # Make bands available globally
    
    # Process the best candidate if found
    if filtered_candidates:
        x, y, w, h, segment_count, segments, score = filtered_candidates[0]
        stakes.append((x, y, w, h))
        
        # Add the stake angle to the visualization
        cv2.putText(result_img, f"Angle: {stake_angle:.1f}°", (x, y - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # For visualization, add a diagnostic image of the red mask
        mask_rgb = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2RGB)
        mask_height, mask_width = red_mask.shape[:2]
        result_height, result_width = result_img.shape[:2]
        
        # Position the mask in the top-right corner (if it fits)
        mask_display_width = min(250, mask_width)
        mask_display_height = int((mask_display_width / mask_width) * mask_height)
        mask_resized = cv2.resize(mask_rgb, (mask_display_width, mask_display_height))
        
        if mask_display_width < result_width and mask_display_height < result_height:
            result_img[10:10+mask_display_height, result_width-mask_display_width-10:result_width-10] = mask_resized
            
        # Draw text to indicate this is the red mask
        cv2.putText(result_img, "Red Mask", (result_width-mask_display_width-10, 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calculate stake endpoints considering the angle
        top_x = x + w // 2
        top_y = y
        bottom_x = top_x + int(h * np.tan(np.radians(stake_angle)))
        bottom_y = y + h
        
        # Draw the angled stake centerline
        cv2.line(result_img, (top_x, top_y), (bottom_x, bottom_y), (255, 0, 255), 2)
        
        # Extract the stake region
        stake_roi = enhanced_img[y:y+h, x:x+w]
        stake_hsv = cv2.cvtColor(stake_roi, cv2.COLOR_BGR2HSV)
        
        # Create masks for red within the stake region
        red_stake_mask1 = cv2.inRange(stake_hsv, lower_red1, upper_red1)
        if lower_red2[0] != 0 or upper_red2[0] != 0:
            red_stake_mask2 = cv2.inRange(stake_hsv, lower_red2, upper_red2)
            red_stake_mask = cv2.bitwise_or(red_stake_mask1, red_stake_mask2)
        else:
            red_stake_mask = red_stake_mask1
        
        # For black, use a simple threshold on value channel
        _, black_stake_mask = cv2.threshold(stake_hsv[:,:,2], 60, 255, cv2.THRESH_BINARY_INV)
        
        # Analyze vertical profile of colors to identify bands, accounting for the slant
        # Sample along the angled stake line to detect bands
        num_samples = h  # One sample per pixel of height
        sample_points_y = np.linspace(0, h-1, num_samples).astype(int)
        sample_points_x = np.array([int((y_pos * np.tan(np.radians(stake_angle))) + w//2) for y_pos in sample_points_y])
        
        # Keep only points within the ROI
        valid_indices = (sample_points_x >= 0) & (sample_points_x < w) & (sample_points_y >= 0) & (sample_points_y < h)
        sample_points_x = sample_points_x[valid_indices]
        sample_points_y = sample_points_y[valid_indices]
        
        # Sample red and black values along the stake
        red_values = []
        black_values = []
        
        for y_pos, x_pos in zip(sample_points_y, sample_points_x):
            # Get values at this point
            if 0 <= y_pos < h and 0 <= x_pos < w:
                red_values.append(red_stake_mask[y_pos, x_pos])
                black_values.append(black_stake_mask[y_pos, x_pos])
        
        # Convert to numpy arrays
        red_values = np.array(red_values)
        black_values = np.array(black_values)
        
        # Smooth the sampled values to reduce noise
        window_size = max(5, int(h / 30))
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd kernel size
            
        # Create 1D kernel for smoothing
        smooth_kernel = np.ones(window_size) / window_size
        
        # Apply convolution to smooth the signals if we have enough samples
        if len(red_values) > window_size:
            red_values_smooth = np.convolve(red_values, smooth_kernel, mode='valid')
            black_values_smooth = np.convolve(black_values, smooth_kernel, mode='valid')
            
            # Adjust the sample points to match the smoothed values
            offset = window_size // 2
            sample_points_y = sample_points_y[offset:offset+len(red_values_smooth)]
        else:
            # If we don't have enough samples for smoothing, use the original values
            red_values_smooth = red_values
            black_values_smooth = black_values
        
        # Detect transitions between red and black
        # Use a threshold to classify points as red or black
        red_threshold = 127
        black_threshold = 127
        
        red_regions = red_values_smooth > red_threshold
        black_regions = black_values_smooth > black_threshold
        
        # Identify bands by detecting transitions
        bands = []
        current_band = None
        band_start_idx = 0
        
        # Define band priority: if both red and black are detected, which one wins
        for i in range(len(red_regions)):
            is_red = red_regions[i]
            is_black = black_regions[i]
            
            if is_red and not is_black:
                new_band = 'red'
            elif is_black and not is_red:
                new_band = 'black'
            elif is_red and is_black:
                # If both detected, prefer the stronger signal
                new_band = 'red' if red_values_smooth[i] > black_values_smooth[i] else 'black'
            else:
                # If neither is strong, maintain the current band
                new_band = current_band
            
            # Handle band transitions
            if current_band is None:
                current_band = new_band
                band_start_idx = i
            elif new_band != current_band and new_band is not None:
                # Minimum band size check (at least 5 pixels)
                if i - band_start_idx >= 5:
                    # Convert indices to original image coordinates
                    start_y = y + sample_points_y[band_start_idx]
                    end_y = y + sample_points_y[i]
                    bands.append((current_band, start_y, end_y))
                
                current_band = new_band
                band_start_idx = i
        
        # Add the final band
        if current_band is not None and len(sample_points_y) > band_start_idx:
            i = len(sample_points_y) - 1
            # Minimum band size check
            if i - band_start_idx >= 5:
                start_y = y + sample_points_y[band_start_idx]
                end_y = y + sample_points_y[i]
                bands.append((current_band, start_y, end_y))
        
        # Merge very small bands that might be noise
        if bands:
            merged_bands = []
            current_band = bands[0]
            min_band_height = max(10, h / 20)  # Minimum height for a band
            
            for i in range(1, len(bands)):
                curr_color, curr_start, curr_end = current_band
                next_color, next_start, next_end = bands[i]
                
                curr_height = curr_end - curr_start
                
                # If current band is very small, merge it with the next band
                if curr_height < min_band_height and next_color == curr_color:
                    current_band = (curr_color, curr_start, next_end)
                else:
                    if curr_height >= min_band_height:
                        merged_bands.append(current_band)
                    current_band = bands[i]
            
            # Add the last band if it's large enough
            curr_color, curr_start, curr_end = current_band
            if curr_end - curr_start >= min_band_height:
                merged_bands.append(current_band)
            
            bands = merged_bands
            
        # Calculate approximate scale (pixels per cm)
        # The entire stake is 150cm tall
        # For slanted stakes, use the path length along the angle
        stake_path_length = np.sqrt(h**2 + (h * np.tan(np.radians(stake_angle)))**2)
        px_per_cm = stake_path_length / 150
        
        # Determine snow depth based on the pattern of visible bands
        if bands:
            # Check the top band color
            top_band_color = bands[0][0]
            
            # Calculate visible height
            visible_height_cm = 0
            
            # Draw bands on the image for visualization
            for i, (color, start_y, end_y) in enumerate(bands):
                band_height_px = end_y - start_y
                
                # Calculate x positions based on stake angle
                start_x = top_x + int((start_y - top_y) * np.tan(np.radians(stake_angle)))
                end_x = top_x + int((end_y - top_y) * np.tan(np.radians(stake_angle)))
                
                # Draw rectangle around the band (approximating with a straight line)
                half_width = w // 2
                if color == 'red':
                    band_color = (255, 0, 0)  # Red
                else:
                    band_color = (0, 0, 0)  # Black
                
                # Draw angled rectangle
                pts = np.array([
                    [start_x - half_width, start_y],
                    [start_x + half_width, start_y],
                    [end_x + half_width, end_y],
                    [end_x - half_width, end_y]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(result_img, [pts], True, band_color, 2)
                
                # Label the band
                mid_y = (start_y + end_y) // 2
                mid_x = top_x + int((mid_y - top_y) * np.tan(np.radians(stake_angle)))
                cv2.putText(result_img, f"{i+1}", (mid_x + half_width + 5, mid_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, band_color, 1)
                
                # Calculate band height considering the slant
                band_path_length = np.sqrt(band_height_px**2 + (band_height_px * np.tan(np.radians(stake_angle)))**2)
                band_height_cm = band_path_length / px_per_cm
                
                # First band might be partially visible
                if i == 0:
                    if top_band_color == 'red':
                        # Top red section is about 5cm
                        visible_height_cm += min(band_height_cm, 5)
                    else:
                        # If top is black, we're at least 5cm down (after the top red section)
                        visible_height_cm += min(band_height_cm, 10)
                else:
                    # Regular bands are 10cm each
                    visible_height_cm += min(band_height_cm, 10)
            
            # Snow depth is the stake height minus the visible portion
            snow_depth_cm = 150 - visible_height_cm
            snow_depths.append(snow_depth_cm)
            
            # Draw stake and snow depth on the image
            # Draw angled rectangle for the stake
            pts = np.array([
                [top_x - w//2, top_y],
                [top_x + w//2, top_y],
                [bottom_x + w//2, bottom_y],
                [bottom_x - w//2, bottom_y]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
            
            cv2.putText(result_img, f"Stake ({segment_count} segments)", (x, y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(result_img, f"Snow: {snow_depth_cm:.1f} cm", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw individual segments for diagnostic purposes
            for sx, sy, sw, sh, _ in segments:
                cv2.rectangle(result_img, (sx, sy), (sx+sw, sy+sh), (255, 255, 0), 2)
            
            # Create band visualization
            # For slanted stakes, we'll create a straightened version
            band_viz_width = 50
            band_viz = np.zeros((int(stake_path_length), band_viz_width, 3), dtype=np.uint8)
            
            # Draw bands on the straightened visualization
            for color, start_y, end_y in bands:
                # Calculate position in the straightened visualization
                # Convert from image coordinates to stake path position
                start_dist = np.sqrt((start_y - top_y)**2 + ((start_y - top_y) * np.tan(np.radians(stake_angle)))**2)
                end_dist = np.sqrt((end_y - top_y)**2 + ((end_y - top_y) * np.tan(np.radians(stake_angle)))**2)
                
                start_viz = int(start_dist)
                end_viz = int(end_dist)
                
                band_color = (0, 0, 255) if color == 'red' else (0, 0, 0)  # BGR format
                cv2.rectangle(band_viz, (0, start_viz), (band_viz_width, end_viz), band_color, -1)
            
            # Add band visualization next to the stake
            viz_x = x + w + 10
            viz_end_x = viz_x + band_viz_width
            
            # Make sure visualization fits in the image and resize if needed
            if band_viz.shape[0] > h:
                band_viz = cv2.resize(band_viz, (band_viz_width, h))
            
            # Make sure visualization fits in the image horizontally
            if viz_end_x < result_img.shape[1] and y + band_viz.shape[0] <= result_img.shape[0]:
                result_img[y:y+band_viz.shape[0], viz_x:viz_end_x] = band_viz
                
                # Label the band visualization
                cv2.putText(result_img, "Bands", (viz_x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # No bands detected, just draw the angled bounding box for the stake
            pts = np.array([
                [top_x - w//2, top_y],
                [top_x + w//2, top_y],
                [bottom_x + w//2, bottom_y],
                [bottom_x - w//2, bottom_y]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
            
            cv2.putText(result_img, f"Stake ({segment_count} segments, no bands detected)", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # No stake found, add message to image
        cv2.putText(result_img, "No stake detected", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.putText(result_img, "Try adjusting detection parameters", (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        # Display the red mask for diagnostic purposes
        mask_rgb = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2RGB)
        mask_height, mask_width = red_mask.shape[:2]
        result_height, result_width = result_img.shape[:2]
        
        # Position the mask in the top-right corner (if it fits)
        mask_display_width = min(250, mask_width)
        mask_display_height = int((mask_display_width / mask_width) * mask_height)
        mask_resized = cv2.resize(mask_rgb, (mask_display_width, mask_display_height))
        
        if mask_display_width < result_width and mask_display_height < result_height:
            result_img[10:10+mask_display_height, result_width-mask_display_width-10:result_width-10] = mask_resized
            
        cv2.putText(result_img, "Red Mask", (result_width-mask_display_width-10, 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save or display results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        base_name = Path(image_path).stem
        
        # Save main result
        plt.figure(figsize=(12, 12))
        plt.imshow(result_img)
        plt.title("Detected Stakes and Snow Depth" if not is_low_visibility else "Low Visibility - Analysis Limited")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"{base_name}_detected_stakes.jpg")
        
        if not is_low_visibility and stakes:
            # Save intermediate processing steps
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            axes[0].imshow(img_rgb)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Enhanced Contrast")
            axes[1].axis('off')
            
            axes[2].imshow(red_mask, cmap='gray')
            axes[2].set_title("Red Mask")
            axes[2].axis('off')
            
            axes[3].imshow(black_mask, cmap='gray')
            axes[3].set_title("Black Mask")
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{base_name}_processing_steps.jpg")
        
        plt.close('all')
        
        print(f"Results saved to {output_dir}")
    else:
        # Display the original and result images
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_img)
        plt.title("Detected Stakes and Snow Depth" if not is_low_visibility else "Low Visibility - Analysis Limited")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return img_rgb, result_img, stakes, snow_depths, is_low_visibility, whiteness_score
