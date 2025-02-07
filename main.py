# main.py
import cv2
from framing import frame_crop
from dataloader import DataLoader
from tracking import track_mouse_in_frame

def main(video_path, crop=True, cropped_folder="cropped_frames"):
    if crop:
        frame_crop(video_path, cropped_folder)

    loader = DataLoader(video_path, cropped_folder)
    print(f"Video FPS: {loader.fps}")

    frame = loader.get_frame_by_index(1010)  # or use loader.load_frames(...)
    tracked_frame, keypoints = track_mouse_in_frame(frame)
    print(f"Detected {len(keypoints)} mouse candidates.")

    # Show the tracked frame
    cv2.imshow("Tracked Frame", tracked_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main(
        video_path = '061224/cage 183/xage183.mp4',
        crop=True,
        cropped_folder = "cropped_frames",
        )
