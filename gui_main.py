import tkinter as tk
from tkinter import filedialog, simpledialog
from main import run_experiment
import sys
import os

if getattr(sys, 'frozen', False):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(sys._MEIPASS, 'platforms')

def run_gui():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Ask user to select a video file
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    if not video_path:
        print("No video selected. Exiting.")
        return

    # Ask user if cropping is needed (accepting y/yes/n/no)
    crop_input = simpledialog.askstring(
        "Cropping Images",
        "Do you need to crop the video to just the box floor? (y/n):"
    )
    if crop_input is None:
        print("No input provided for cropping. Exiting.")
        return
    crop_input = crop_input.lower().strip()
    if crop_input in ["y", "yes"]:
        crop = True
    elif crop_input in ["n", "no"]:
        crop = False
    else:
        print("Invalid input for cropping. Exiting.")
        return

    # Ask for additional parameters
    start_time = simpledialog.askinteger("Start Time", "Enter start time in seconds:", minvalue=0)
    duration = simpledialog.askinteger("Duration", "Enter duration in seconds:", minvalue=1)
    center_percent = simpledialog.askinteger(
        "Center Area",
        "Enter center area percentage (e.g., 50 for inner 50%):",
        minvalue=10, maxvalue=100
    )

    if start_time is None or duration is None or center_percent is None:
        print("Invalid input. Exiting.")
        return

    # Run tracking with user inputs
    run_experiment(video_path, crop=crop, start_time=start_time, duration=duration, center_percent=center_percent)

if __name__ == "__main__":
    run_gui()
