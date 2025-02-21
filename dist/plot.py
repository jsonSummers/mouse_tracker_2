#!/usr/bin/env python3
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def load_frame_image(videoname):
    """
    Attempt to load an image from outputs/cropped_frames/<videoname>.
    If <videoname> is a directory, pick the first image (png or jpg).
    Otherwise, assume it is a direct file.
    """
    base_path = os.path.join("outputs", "cropped_frames", videoname)
    if os.path.isdir(base_path):
        # Try to find a PNG or JPG in the directory.
        image_files = glob.glob(os.path.join(base_path, "*.png")) + glob.glob(os.path.join(base_path, "*.jpg"))
        if not image_files:
            raise FileNotFoundError(f"No image file found in directory: {base_path}")
        image_path = image_files[0]
    else:
        # Assume base_path is a file path.
        if not os.path.isfile(base_path):
            raise FileNotFoundError(f"No image file found at: {base_path}")
        image_path = base_path

    return Image.open(image_path)

def main(videoname, inner_area_percent=80, smoothing_window=5):
    # Load tracking CSV (assumes columns: 'frame', 'x', 'y', and a boolean column)
    csv_path = os.path.join("outputs", "results", f"{videoname}_tracking.csv")
    df = pd.read_csv(csv_path)

    # Load an image to get the original dimensions
    image = load_frame_image(videoname)
    width, height = image.size

    # Normalize coordinates to a [0,1] range
    df['x_norm'] = df['x'] / width
    df['y_norm'] = df['y'] / height

    # Smooth the coordinates using a rolling average (with a centered window)
    df['x_smooth'] = df['x_norm'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df['y_smooth'] = df['y_norm'].rolling(window=smoothing_window, center=True, min_periods=1).mean()

    # Set seaborn style for a cleaner look
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 8))

    # Plot the smooth mouse trajectory
    plt.plot(df['x_smooth'], df['y_smooth'], color='blue', linewidth=2, label='Mouse Trajectory')

    # Draw the outer square: the normalized area [0, 1] x [0, 1]
    outer_square = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2, label='Outer Square')
    plt.gca().add_patch(outer_square)

    # Calculate inner square side length so its area is inner_area_percent of the outer square area.
    # Outer square area is 1; hence side = sqrt(inner_area_percent/100)
    side_inner = np.sqrt(inner_area_percent / 100)
    lower_left = ((1 - side_inner) / 2, (1 - side_inner) / 2)
    inner_square = plt.Rectangle(lower_left, side_inner, side_inner, fill=False, edgecolor='red', linewidth=2, label=f'Inner Square ({inner_area_percent}%)')
    plt.gca().add_patch(inner_square)

    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.title(f"Mouse Trajectory for {videoname}")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main(
        videoname='xage183',
        inner_area_percent=50,
        smoothing_window=10
    )
