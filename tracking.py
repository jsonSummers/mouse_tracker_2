import cv2
import numpy as np


def track_mouse_in_frame(frame, prev_center=None, max_distance=100):
    """
    Detect dark blobs in the given frame using OpenCV's SimpleBlobDetector.
    Returns a processed frame with the best candidate (assumed to be the mouse) highlighted.

    If a previous center is provided, the blob closest to that center is chosen,
    provided it is within max_distance; otherwise, the largest blob is selected.

    Args:
        frame (numpy.ndarray): The input frame (BGR image).
        prev_center (tuple): Optional (x, y) coordinate from the previous frame.
        max_distance (float): Maximum allowed distance from prev_center to consider a blob.

    Returns:
        frame_with_detection (numpy.ndarray): The frame with the chosen candidate highlighted.
        best_candidate (list): A list containing the best candidate keypoint, or empty if none found.
    """
    # Convert frame to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise.
    gray_blurred = cv2.medianBlur(gray, 11)

    # Set up blob detector parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 256
    params.filterByColor = True
    params.blobColor = 0  # look for dark regions
    params.filterByArea = True
    params.minArea = 500  # increase minArea to filter out small noise
    params.maxArea = 10000
    # You can optionally enable filters like circularity if your mouse has a predictable shape.
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_blurred)

    best_candidate = None

    if keypoints:
        if prev_center is not None:
            # Compute the Euclidean distance between each keypoint and the previous center.
            distances = [np.linalg.norm(np.array(kp.pt) - np.array(prev_center))
                         for kp in keypoints]
            # Find the candidate that is closest.
            min_idx = np.argmin(distances)
            if distances[min_idx] <= max_distance:
                best_candidate = keypoints[min_idx]
            else:
                # Fall back to choosing the largest blob.
                best_candidate = max(keypoints, key=lambda kp: kp.size)
        else:
            # If no previous location is provided, choose the largest blob.
            best_candidate = max(keypoints, key=lambda kp: kp.size)
        keypoints = [best_candidate]  # Always return as a list.
    else:
        keypoints = []

    # Draw the selected candidate on the frame.
    frame_with_detection = cv2.drawKeypoints(
        frame, keypoints, np.array([]),
        (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return frame_with_detection, keypoints


# Example usage:
if __name__ == '__main__':
    image_path = 'cropped_frames/frame_02700.png'
    frame = cv2.imread(image_path)

    # For testing, assume a previous center; in practice, update this each frame.
    previous_center = (330, 130)  # Example coordinate
    output_frame, keypoints = track_mouse_in_frame(frame, prev_center=previous_center)

    if keypoints:
        print("Detected candidate at:", keypoints[0].pt)
    else:
        print("No candidate detected.")

    cv2.imshow("Tracked Frame", output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()