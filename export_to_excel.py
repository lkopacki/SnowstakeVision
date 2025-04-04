def export_to_excel(results, output_path):
    """
    Export snow depth measurements to an Excel file.
    
    Args:
        results: List of tuples containing (image_name, stakes, snow_depths, is_low_visibility, whiteness_score)
        output_path: Path to save the Excel file
    """
    try:
        # Create data for the Excel file
        data = []
        
        for img_name, stakes, snow_depths, is_low_visibility, whiteness_score in results:
            # Try to extract date from filename (assuming format like YYYY-MM-DD or similar)
            date_str = None
            date_match = re.search(r'(\d{4}[-_/]?\d{1,2}[-_/]?\d{1,2})', img_name)
            if date_match:
                date_str = date_match.group(1)
            
            if is_low_visibility:
                data.append({
                    'Image Name': img_name,
                    'Date': date_str,
                    'Status': 'Low Visibility',
                    'Whiteness Score': f"{whiteness_score:.2f}",
                    'Stakes Detected': 0,
                    'Snow Depth (cm)': None,
                    'Notes': 'Excluded due to fog/precipitation'
                })
            else:
                # If no stakes were detected
                if not snow_depths:
                    data.append({
                        'Image Name': img_name,
                        'Date': date_str,
                        'Status': 'Processed',
                        'Whiteness Score': f"{whiteness_score:.2f}",
                        'Stakes Detected': 0,
                        'Snow Depth (cm)': None,
                        'Notes': 'No stakes detected'
                    })
                else:
                    # For each stake in the image
                    for i, depth in enumerate(snow_depths):
                        data.append({
                            'Image Name': img_name,
                            'Date': date_str,
                            'Status': 'Processed',
                            'Whiteness Score': f"{whiteness_score:.2f}",
                            'Stakes Detected': len(snow_depths),
                            'Stake Number': i + 1,
                            'Snow Depth (cm)': f"{depth:.1f}",
                            'Notes': ''
                        })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add summary statistics
        total_images = len(results)
        low_vis_images = sum(1 for _, _, _, is_low_vis, _ in results if is_low_vis)
        processed_images = total_images - low_vis_images
        
        valid_depths = [depth for _, _, depths, is_low_vis, _ in results 
                       for depth in depths if not is_low_vis]
        
        avg_depth = sum(valid_depths) / len(valid_depths) if valid_depths else None
        min_depth = min(valid_depths) if valid_depths else None
        max_depth = max(valid_depths) if valid_depths else None
        
        # Create a summary DataFrame
        summary_data = [
            {'Metric': 'Total Images', 'Value': total_images},
            {'Metric': 'Low Visibility Images', 'Value': low_vis_images},
            {'Metric': 'Processed Images', 'Value': processed_images},
            {'Metric': 'Average Snow Depth (cm)', 'Value': f"{avg_depth:.1f}" if avg_depth is not None else 'N/A'},
            {'Metric': 'Minimum Snow Depth (cm)', 'Value': f"{min_depth:.1f}" if min_depth is not None else 'N/A'},
            {'Metric': 'Maximum Snow Depth (cm)', 'Value': f"{max_depth:.1f}" if max_depth is not None else 'N/A'},
            {'Metric': 'Processing Date', 'Value': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ]
        summary_df = pd.DataFrame(summary_data)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path) as writer:
            df.to_excel(writer, sheet_name='Snow Depth Measurements', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Auto-adjust columns' width
            for sheet in writer.sheets:
                worksheet = writer.sheets[sheet]
                for i, col in enumerate(df.columns if sheet == 'Snow Depth Measurements' else summary_df.columns):
                    column_len = max(df[col].astype(str).map(len).max() if sheet == 'Snow Depth Measurements' else 
                                    summary_df[col].astype(str).map(len).max(), 
                                    len(col)) + 2
                    worksheet.set_column(i, i, column_len)
                    
        print(f"Excel report saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        traceback.print_exc()
        return False
