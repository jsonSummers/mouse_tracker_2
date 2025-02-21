import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from dataloader import DataLoader  # Ensure this is implemented correctly


def compute_center_box(roi_width, roi_height, center_percent):
    """Compute the center box coordinates given a percentage."""
    box_w = roi_width * center_percent / 100
    box_h = roi_height * center_percent / 100
    top_left_x = (roi_width - box_w) / 2
    top_left_y = (roi_height - box_h) / 2
    bottom_right_x = top_left_x + box_w
    bottom_right_y = top_left_y + box_h
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def analyze_tracking_data(csv_file, video_path, cropped_folder, center_percent=50):
    """Analyzes mouse tracking results and generates statistics/plots."""

    # Load tracking data
    df = pd.read_csv(csv_file)

    # Load video data using DataLoader
    loader = DataLoader(video_path, cropped_folder)
    fps = loader.fps
    roi_width, roi_height = loader.get_frame_dimensions()

    # Compute center box coordinates
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = compute_center_box(roi_width, roi_height, center_percent)

    # Recompute 'in_center' in case we need verification
    df["in_center"] = df.apply(lambda row:
                               top_left_x <= row["x"] <= bottom_right_x and top_left_y <= row["y"] <= bottom_right_y,
                               axis=1)

    # Compute statistics
    total_frames = len(df)
    frames_in_center = df["in_center"].sum()
    frames_out = total_frames - frames_in_center
    time_in_center = frames_in_center / fps
    time_out = frames_out / fps

    print(f"Total frames analyzed: {total_frames}")
    print(f"Frames in center: {frames_in_center} ({frames_in_center / total_frames * 100:.2f}%)")
    print(f"Frames out of center: {frames_out} ({frames_out / total_frames * 100:.2f}%)")
    print(f"Time in center: {time_in_center:.2f} seconds")
    print(f"Time outside center: {time_out:.2f} seconds")

    # Compute distance from ROI center
    roi_center_x = roi_width / 2
    roi_center_y = roi_height / 2
    df["distance_from_center"] = np.sqrt((df["x"] - roi_center_x) ** 2 + (df["y"] - roi_center_y) ** 2)

    # Plot distance from center over time
    plt.figure(figsize=(10, 6))
    plt.plot(df["frame"], df["distance_from_center"], label="Distance from Center", color='b')
    plt.xlabel("Frame Number")
    plt.ylabel("Distance from Center (pixels)")
    plt.title("Mouse Distance from Center Over Time")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    analyze_tracking_data(
        csv_file="tracking_results.csv",
        video_path="061224/cage 183/xage183.mp4",
        cropped_folder="cropped_frames",
        center_percent=50  # Modify as needed
    )

