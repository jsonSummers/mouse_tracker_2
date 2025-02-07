import cv2
import os


def select_roi(frame):
    """
    Display a window where the user can select the ROI.
    The user can click and drag to draw the ROI; press ENTER or SPACE to confirm,
    and c to cancel.
    """
    # The function returns a tuple (x, y, w, h)
    roi = cv2.selectROI("Select ROI - Press ENTER/SPACE when done", frame,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI - Press ENTER/SPACE when done")
    return roi


def crop_video(video_path, output_folder, roi):
    """
    Process the video frame by frame. Crop each frame to the ROI and save the result.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder where cropped frames will be saved.
        roi (tuple): Region of interest (x, y, w, h).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h = roi
        cropped_frame = frame[y:y + h, x:x + w]
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, cropped_frame)
        frame_count += 1

    cap.release()
    print(f"Saved {frame_count} cropped frames to '{output_folder}'")


def frame_crop(video_path, output_folder):

    # Open the video and grab a sample frame.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    # Retrieve the total number of frames.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Video contains no frames.")
        cap.release()
        return

    # Set the position to the middle frame.
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error reading the first frame of the video.")
        return

    # Let the user select the ROI on the sample frame.
    roi = select_roi(frame)
    print("Selected ROI:", roi)

    # Process the video: crop and save each frame.
    crop_video(video_path, output_folder, roi)
