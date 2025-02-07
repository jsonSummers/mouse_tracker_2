# main.py
import cv2
from framing import frame_crop
from dataloader import DataLoader
from tracking import track_mouse_in_frame


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


def main(video_path, crop=True, cropped_folder="cropped_frames"):
    if crop:
        frame_crop(video_path, cropped_folder)

    loader = DataLoader(video_path, cropped_folder)
    print(f"Video FPS: {loader.fps}")

    tracker = MouseTracker(min_area=700)
    start_frame = 1010
    num_frames = 2000

    mouse_coordinates = []

    for frame_idx in range(start_frame, start_frame + num_frames):
        frame = loader.get_frame_by_index(frame_idx)
        tracked_frame, keypoints = tracker.track_frame(frame)

        if keypoints:
            mouse_coordinates.append(keypoints[0])
        else:
            mouse_coordinates.append(tracker.last_coordinate)

        cv2.imshow("Tracked Frame", tracked_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    print("Mouse coordinates over frames:")
    for coord in mouse_coordinates:
        print(coord)



if __name__ == "__main__":
    main(
        video_path = '061224/cage 183/xage183.mp4',
        crop=False,
        cropped_folder = "cropped_frames",
        )