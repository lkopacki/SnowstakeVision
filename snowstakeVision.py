import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime
import re
import traceback

# Import sub scripts
import export_to_excel
from create_gui import create_gui


if __name__ == "__main__":
    # Launch the GUI for interactive testing
    print("Starting Snow Depth Measurement Tool GUI...")
    try:
        create_gui()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        traceback.print_exc()