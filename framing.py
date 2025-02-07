import cv2
import numpy as np
import os


def enhance_roi(frame, alpha=1.2, beta=-25, gamma=1.2):
# def enhance_roi(frame, alpha=1.0, beta=0, gamma=1.2):
    """
    Enhance the cropped ROI by reducing brightness, applying gamma correction,
    boosting local contrast via CLAHE, and applying Gaussian blur to reduce noise.

    Args:
        frame (numpy.ndarray): Input cropped color image.
        alpha (float): Contrast control (1.0 means no change).
        beta (int): Brightness control (negative values darken the image).
        gamma (float): Gamma correction value (values > 1 darken the image).

    Returns:
        final (numpy.ndarray): The enhanced image.
    """
    # 1. Adjust brightness and contrast.
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # 2. Apply gamma correction.
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(adjusted, table)

    # 3. Boost local contrast using CLAHE on the luminance channel.
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 4. Apply Gaussian blur to reduce noise.
    final = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return final


def select_roi(frame):
    """
    Display a window for the user to select the ROI.
    The user can click and drag to draw the ROI; press ENTER or SPACE to confirm,
    and press c to cancel.
    Returns a tuple (x, y, w, h).
    """
    roi = cv2.selectROI("Select ROI - Press ENTER/SPACE when done", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI - Press ENTER/SPACE when done")
    return roi


def crop_video(video_path, output_folder, roi, apply_enhancement=True):
    """
    Process the video frame by frame, crop each frame to the ROI, optionally enhance it,
    and then save the result.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder where cropped frames will be saved.
        roi (tuple): Region of interest (x, y, w, h).
        apply_enhancement (bool): Whether to enhance the ROI before saving.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h = roi
        cropped_frame = frame[y:y + h, x:x + w]

        # Optionally enhance the cropped frame.
        if apply_enhancement:
            cropped_frame = enhance_roi(cropped_frame)

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, cropped_frame)
        frame_count += 1

    cap.release()
    print(f"Saved {frame_count} cropped frames to '{output_folder}'")


def frame_crop(video_path, output_folder):
    """
    Opens a video, lets the user select a ROI on a sample frame,
    then processes the video to crop (and enhance) each frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Video contains no frames.")
        cap.release()
        return

    # Use the middle frame for ROI selection.
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error reading the sample frame from the video.")
        return

    roi = select_roi(frame)
    print("Selected ROI:", roi)

    # Process the video: crop, enhance, and save each frame.
    crop_video(video_path, output_folder, roi, apply_enhancement=True)


if __name__ == "__main__":
    video_path = '061224/cage 183/xage183.mp4'
    output_folder = "cropped_frames"
    frame_crop(video_path, output_folder)
