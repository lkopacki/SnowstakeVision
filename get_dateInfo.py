#get metadata
"""This file pulls date and time information stored in the metadata of each photo"""
import os
import win32com.client

def get_windows_date_taken(image_path):
    """Get the 'Date taken' property as shown in Windows Explorer."""
    shell = win32com.client.Dispatch("Shell.Application")
    folder = shell.NameSpace(os.path.dirname(os.path.abspath(image_path)))
    file = folder.ParseName(os.path.basename(image_path))
    
    # In Windows, property indices can vary by system
    # We'll scan through properties to find "Date taken"
    date_taken = None
    for i in range(0, 300):
        property_name = folder.GetDetailsOf(None, i)
        if property_name == "Date taken":
            date_taken = folder.GetDetailsOf(file, i)
            break
            
    return date_taken if date_taken else "No date information found"

# Example usage
image_path = r"C:\Users\LukasKopacki\Downloads\SnowstakeData_GS\SnowstakeData_GS\22_1\22_1\WSCT1017.JPG"
date_taken = get_windows_date_taken(image_path)
print(f"Date taken: {date_taken}")