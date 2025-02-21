# main.py
import cv2
import csv
import os

from framing import frame_crop
from dataloader import DataLoader
from tracking import MouseTracker

def draw_center_box(frame, percent=50):
    """Draws a centered rectangle covering n% of the frame's area."""
    height, width = frame.shape[:2]
    box_w = int(width * percent / 100)
    box_h = int(height * percent / 100)
    top_left = ((width - box_w) // 2, (height - box_h) // 2)
    bottom_right = (top_left[0] + box_w, top_left[1] + box_h)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
    return top_left, bottom_right

def is_inside_box(coordinate, top_left, bottom_right):
    """Checks if a coordinate (x,y) is inside the defined center box."""
    if coordinate:
        x, y = coordinate
        return top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]
    return False


def run_experiment(video_path, crop=True, start_time=30, duration=300, center_percent=50):
    # Extract the video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Set up output directories
    results_folder = os.path.join("outputs", "results")
    os.makedirs(results_folder, exist_ok=True)
    output_csv = os.path.join(results_folder, f"{video_name}_tracking.csv")

    if crop:
        cropped_frames_folder = os.path.join("outputs", "cropped_frames", video_name)
        os.makedirs(cropped_frames_folder, exist_ok=True)
        # Crop frames and save to the designated folder
        frame_crop(video_path, cropped_frames_folder)
        # Use the cropped frames folder for DataLoader
        loader = DataLoader(video_path, cropped_frames_folder)
    else:
        # If no cropping, assume DataLoader will handle the video directly
        loader = DataLoader(video_path)

    print(f"Video FPS: {loader.fps}")
    tracker = MouseTracker(min_area=500)
    start_frame = int(start_time * loader.fps)
    num_frames = int(duration * loader.fps)

    # Create and open the CSV file for writing tracking results
    with open(output_csv, mode="w", newline="") as csvfile:
        fieldnames = ["frame", "x", "y", "in_center"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for frame_idx in range(start_frame, start_frame + num_frames):
            frame = loader.get_frame_by_index(frame_idx)
            box_top_left, box_bottom_right = draw_center_box(frame, percent=center_percent)
            tracked_frame, keypoints = tracker.track_frame(frame)
            coordinate = keypoints[0] if keypoints else tracker.last_coordinate
            in_center = is_inside_box(coordinate, box_top_left, box_bottom_right)

            writer.writerow({
                "frame": frame_idx,
                "x": coordinate[0] if coordinate else "",
                "y": coordinate[1] if coordinate else "",
                "in_center": in_center
            })

            # Display the tracked frame
            cv2.imshow("Tracked Frame", tracked_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print(f"Tracking results saved to {output_csv}")