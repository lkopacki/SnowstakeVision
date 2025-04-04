import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import traceback
from datetime import datetime
import re
from detectStakes import detect_red_black_stakes

def create_gui():
    """
    Create a simple tkinter GUI for testing the stake detection algorithm.
    """
    def select_image():
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not path:
            return
        
        # Get the visibility threshold from the slider
        vis_threshold = float(visibility_threshold_var.get())
        process_image(path, vis_threshold)
    
    def select_directory():
        dir_path = filedialog.askdirectory()
        if not dir_path:
            return
        
        # Get the visibility threshold from the slider
        vis_threshold = float(visibility_threshold_var.get())
        
        output_dir = Path(dir_path) / "results"
        results = process_multiple_images(dir_path, str(output_dir), vis_threshold, export_excel=True)
        
        # Update the results text
        results_text.delete(1.0, tk.END)
        
        # Group results by visibility
        low_vis_images = []
        processed_images = []
        
        for img_name, stakes, snow_depths, is_low_visibility, whiteness_score in results:
            if is_low_visibility:
                low_vis_images.append((img_name, whiteness_score))
            else:
                processed_images.append((img_name, len(stakes), snow_depths))
        
        # Display summary
        results_text.insert(tk.END, f"Processed {len(results)} images\n")
        results_text.insert(tk.END, f"Low visibility images: {len(low_vis_images)}\n")
        results_text.insert(tk.END, f"Successfully analyzed images: {len(processed_images)}\n\n")
        
        # Excel export info
        excel_path = output_dir / "snow_depth_measurements.xlsx"
        if excel_path.exists():
            results_text.insert(tk.END, f"Excel report saved to: {excel_path}\n\n")
        
        # Display low visibility images
        if low_vis_images:
            results_text.insert(tk.END, "Low Visibility Images (Skipped Analysis):\n")
            for img_name, whiteness_score in low_vis_images:
                results_text.insert(tk.END, f"  {img_name} - Whiteness score: {whiteness_score:.2f}\n")
            results_text.insert(tk.END, "\n")
        
        # Display processed images
        if processed_images:
            results_text.insert(tk.END, "Analyzed Images:\n")
            for img_name, stake_count, snow_depths in processed_images:
                avg_depth = sum(snow_depths) / len(snow_depths) if snow_depths else "N/A"
                if isinstance(avg_depth, float):
                    avg_depth = f"{avg_depth:.1f} cm"
                results_text.insert(tk.END, f"  {img_name} - Stakes: {stake_count}, Avg Snow Depth: {avg_depth}\n")
    
    def process_image(path, visibility_threshold=0.85):
        try:
            orig, result, stakes, snow_depths, is_low_visibility, whiteness_score = detect_red_black_stakes(
                path, visibility_threshold=visibility_threshold)
            
            # Store these globally for potential export
            global orig_img_global, stakes_global, snow_depths_global, is_low_visibility_global, whiteness_score_global, bands_global
            orig_img_global = orig
            stakes_global = stakes
            snow_depths_global = snow_depths
            is_low_visibility_global = is_low_visibility
            whiteness_score_global = whiteness_score
            bands_global = bands if 'bands' in locals() else []
            
            # Convert images for display
            orig_pil = Image.fromarray(orig)
            result_pil = Image.fromarray(result)
            
            # Resize if needed
            max_width = 500
            ratio = max_width / orig_pil.width
            new_size = (max_width, int(orig_pil.height * ratio))
            
            orig_pil = orig_pil.resize(new_size)
            result_pil = result_pil.resize(new_size)
            
            # Update images
            orig_img = ImageTk.PhotoImage(orig_pil)
            result_img = ImageTk.PhotoImage(result_pil)
            
            orig_label.config(image=orig_img)
            orig_label.image = orig_img
            
            result_label.config(image=result_img)
            result_label.image = result_img
            
            # Update info
            if is_low_visibility:
                info_label.config(text=f"Low Visibility Image (Whiteness: {whiteness_score:.2f})")
            else:
                if stakes:
                    info_label.config(text=f"Stake detected" + (f", Snow Depth: {snow_depths[0]:.1f} cm" if snow_depths else ""))
                else:
                    info_label.config(text="No stake detected - try adjusting parameters")
            
            # Update the results text
            results_text.delete(1.0, tk.END)
            
            if is_low_visibility:
                results_text.insert(tk.END, "Image has low visibility due to fog or heavy precipitation.\n")
                results_text.insert(tk.END, f"Whiteness score: {whiteness_score:.2f} (Threshold: {visibility_threshold})\n")
                results_text.insert(tk.END, "Analysis skipped to avoid unreliable measurements.\n")
            elif not stakes:
                results_text.insert(tk.END, "No stake detected in the image.\n")
                results_text.insert(tk.END, "Suggestions:\n")
                results_text.insert(tk.END, "- Try adjusting the visibility threshold\n")
                results_text.insert(tk.END, "- Check that the image contains a red and black stake\n")
                results_text.insert(tk.END, "- Ensure the image has good lighting and contrast\n")
            else:
                x, y, w, h = stakes[0]
                snow_depth_text = f", Snow depth: {snow_depths[0]:.1f} cm" if snow_depths else ""
                results_text.insert(tk.END, f"Stake detected: x={x}, y={y}, width={w}, height={h}{snow_depth_text}\n")
                
                # Add band information if available
                if bands_global:
                    results_text.insert(tk.END, "\nBands detected (from top to bottom):\n")
                    for i, (color, start_y, end_y) in enumerate(bands_global):
                        results_text.insert(tk.END, f"{i+1}: {color.upper()} band, height: {end_y-start_y} pixels\n")
                
                # Add calculation explanation
                if snow_depths:
                    results_text.insert(tk.END, f"\nSnow Depth Calculation:\n")
                    results_text.insert(tk.END, f"- Stake total height: 150 cm\n")
                    visible_height = 150 - snow_depths[0]
                    results_text.insert(tk.END, f"- Visible portion: {visible_height:.1f} cm\n")
                    results_text.insert(tk.END, f"- Snow depth: 150 - {visible_height:.1f} = {snow_depths[0]:.1f} cm\n")
        except Exception as e:
            info_label.config(text=f"Error: {str(e)}")
            # Show detailed error for debugging
            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, traceback.format_exc())

    # Create the main window
    root = tk.Tk()
    root.title("Snow Depth Measurement Tool")
    root.geometry("1200x800")  # Larger window for better visualization
    
    # Create frames
    top_frame = ttk.Frame(root, padding=10)
    top_frame.pack(fill=tk.X)
    
    # Add visibility threshold slider
    visibility_frame = ttk.LabelFrame(top_frame, text="Low Visibility Detection", padding=10)
    visibility_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    visibility_threshold_var = tk.StringVar(value="0.85")
    ttk.Label(visibility_frame, text="Whiteness Threshold:").pack(side=tk.LEFT)
    threshold_slider = ttk.Scale(
        visibility_frame, 
        from_=0.5, 
        to=0.95, 
        orient=tk.HORIZONTAL, 
        length=200,
        variable=visibility_threshold_var
    )
    threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    threshold_slider.set(0.85)
    
    # Display the current value
    threshold_label = ttk.Label(visibility_frame, textvariable=visibility_threshold_var)
    threshold_label.pack(side=tk.LEFT, padx=5)
    
    # Update the label when the slider changes
    def update_threshold_label(*args):
        threshold_label.config(text=f"{float(visibility_threshold_var.get()):.2f}")
    
    threshold_slider.config(command=lambda v: visibility_threshold_var.set(f"{float(v):.2f}"))
    
    # Add buttons to top frame
    button_frame = ttk.Frame(top_frame, padding=10)
    button_frame.pack(side=tk.LEFT, fill=tk.X)
    
    ttk.Button(button_frame, text="Select Single Image", command=select_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Process Directory", command=select_directory).pack(side=tk.LEFT, padx=5)
    
    # Add Excel export button for single image
    def export_single_image_results():
        # Check if we have processed an image
        if 'orig_img_global' not in globals() or 'stakes_global' not in globals() or 'snow_depths_global' not in globals():
            info_label.config(text="No image processed yet. Please select an image first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile="snow_depth_measurement.xlsx"
        )
        if not file_path:
            return
            
        # Create results data in the same format as process_multiple_images
        img_name = "single_image.jpg"  # Default name for single image
        
        results = [(img_name, stakes_global, snow_depths_global, is_low_visibility_global, whiteness_score_global)]
        
        if export_to_excel(results, file_path):
            info_label.config(text=f"Results exported to {file_path}")
        else:
            info_label.config(text="Failed to export results")
            
    ttk.Button(button_frame, text="Export Current to Excel", command=export_single_image_results).pack(side=tk.LEFT, padx=5)
    
    image_frame = ttk.Frame(root, padding=10)
    image_frame.pack(fill=tk.BOTH, expand=True)
    
    info_frame = ttk.Frame(root, padding=10)
    info_frame.pack(fill=tk.X)
    
    # Add image labels to image frame
    orig_label = ttk.Label(image_frame)
    orig_label.pack(side=tk.LEFT, padx=5)
    
    result_label = ttk.Label(image_frame)
    result_label.pack(side=tk.LEFT, padx=5)
    
    # Add info label
    info_label = ttk.Label(info_frame, text="Select an image to begin")
    info_label.pack(anchor=tk.W)
    
    # Create a section to explain the snow depth calculation
    explanation_frame = ttk.LabelFrame(info_frame, text="Measurement Information", padding=10)
    explanation_frame.pack(fill=tk.X, pady=5)
    
    explanation_text = tk.Text(explanation_frame, height=5, width=60, wrap=tk.WORD)
    explanation_text.insert(tk.END, 
        "Snow Depth Calculation:\n"
        "• The stake is 150cm tall with alternating 10cm red and black bands\n"
        "• Top of stake has a small red section, followed by a black band\n"
        "• Snow depth = 150cm - visible length of the stake\n"
        "• Band detection may need adjustment for different lighting conditions"
    )
    explanation_text.config(state=tk.DISABLED)
    explanation_text.pack(fill=tk.X)
    
    # Add results text area
    results_frame = ttk.LabelFrame(info_frame, text="Detection Results", padding=10)
    results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    results_text = tk.Text(results_frame, height=8, width=60)
    results_text.pack(fill=tk.BOTH, expand=True)
    
    root.mainloop()
