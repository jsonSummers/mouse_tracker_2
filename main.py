# main.py
import cv2
from framing import frame_crop
from dataloader import DataLoader
from tracking import track_mouse_in_frame
import cv2
import csv
import os

# def main(video_path, crop=True, cropped_folder="cropped_frames"):
#     if crop:
#         frame_crop(video_path, cropped_folder)
#
#     loader = DataLoader(video_path, cropped_folder)
#
#     frame = loader.get_frame_by_index(1010)  # or use loader.load_frames(...)
#     tracked_frame, keypoints = track_mouse_in_frame(frame)
#     print(f"Detected {len(keypoints)} mouse candidates.")
#
#     # Show the tracked frame
#     cv2.imshow("Tracked Frame", tracked_frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2
from framing import frame_crop
from dataloader import DataLoader
from tracking import MouseTracker  # or use track_mouse_in_frame()

def draw_center_box(frame, percent=50):
    """
    Draws a centered rectangle covering n% of the frame's area.
    Returns the top-left and bottom-right coordinates of the box.
    """
    height, width = frame.shape[:2]
    box_w = int(width * percent / 100)
    box_h = int(height * percent / 100)
    top_left = ((width - box_w) // 2, (height - box_h) // 2)
    bottom_right = (top_left[0] + box_w, top_left[1] + box_h)
    # Draw the center box (yellow)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
    return top_left, bottom_right

def is_inside_box(coordinate, top_left, bottom_right):
    """
    Returns True if coordinate (x,y) is inside the box defined by top_left and bottom_right.
    """
    x, y = coordinate
    if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
        return True
    return False

def main(video_path, crop=True, cropped_folder="cropped_frames", start_time=30, duration=300, center_percent=50, output_csv="tracking_results.csv"):
    if crop:
        frame_crop(video_path, cropped_folder)

    loader = DataLoader(video_path, cropped_folder)
    print(f"Video FPS: {loader.fps}")

    tracker = MouseTracker(min_area=500)  # adjust as needed
    start_frame = int(start_time * loader.fps)
    num_frames = int(duration * loader.fps)

    print("frames", start_frame, num_frames)
    # start_frame = 1010
    # num_frames = 200

    # Create/open a CSV file to record results.
    with open(output_csv, mode="w", newline="") as csvfile:
        fieldnames = ["frame", "x", "y", "in_center"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for frame_idx in range(start_frame, start_frame + num_frames):
            frame = loader.get_frame_by_index(frame_idx)
            # Draw the center box (this returns the box coordinates)
            box_top_left, box_bottom_right = draw_center_box(frame, percent=center_percent)

            tracked_frame, keypoints = tracker.track_frame(frame)
            if keypoints:
                coordinate = keypoints[0]
            else:
                coordinate = tracker.last_coordinate  # fallback

            # Mark the coordinate on the frame (if not already marked by the tracker)
            if coordinate is not None:
                cv2.circle(tracked_frame, coordinate, 5, (0, 0, 255), -1)
                # Determine if the coordinate is inside the center box
                in_center = is_inside_box(coordinate, box_top_left, box_bottom_right)
            else:
                in_center = False

            # Write result to CSV
            writer.writerow({
                "frame": frame_idx,
                "x": coordinate[0] if coordinate is not None else "",
                "y": coordinate[1] if coordinate is not None else "",
                "in_center": in_center
            })

            # Optionally, display the tracked frame
            cv2.imshow("Tracked Frame", tracked_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print(f"Tracking results saved to {output_csv}")

# def main(video_path, crop=True, cropped_folder="cropped_frames"):
#     if crop:
#         frame_crop(video_path, cropped_folder)
#
#     loader = DataLoader(video_path, cropped_folder)
#     print(f"Video FPS: {loader.fps}")
#
#     tracker = MouseTracker(min_area=400, use_blob_detector=True)
#     start_frame = 1010
#     num_frames = 500
#     mouse_coordinates = []
#
#     for frame_idx in range(start_frame, start_frame + num_frames):
#         frame = loader.get_frame_by_index(frame_idx)
#         tracked_frame, keypoints = tracker.track_frame(frame)
#
#         if keypoints:
#             mouse_coordinates.append(keypoints[0])
#         else:
#             mouse_coordinates.append(tracker.last_coordinate)
#
#         cv2.imshow("Tracked Frame", tracked_frame)
#         if cv2.waitKey(30) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#     print("Mouse coordinates:", mouse_coordinates)



if __name__ == "__main__":
    main(
        video_path = '061224/cage 183/xage183.mp4',
        crop=False,
        cropped_folder = "cropped_frames",
        start_time=30,
        duration=20,
        center_percent=50,  # inner 50% of the area
        output_csv="tracking_results.csv"
    )