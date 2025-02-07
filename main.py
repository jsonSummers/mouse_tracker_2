# main.py
import cv2
from framing import frame_crop
from dataloader import DataLoader

def main(video_path, crop=True, cropped_folder="cropped_frames"):
    if crop:
        frame_crop(video_path, cropped_folder)

    loader = DataLoader(video_path, cropped_folder)
    print(f"Video FPS: {loader.fps}")



if __name__ == "__main__":
    main(
        video_path = '061224/cage 183/xage183.mp4',
        crop=False,
        cropped_folder = "cropped_frames",
        )
