# tracking.py
import cv2
import numpy as np


def track_mouse_in_frame(frame):
    """
    Detects dark blobs in the given frame using OpenCV's SimpleBlobDetector.
    Assumes that the mouse appears as a dark (low intensity) object.

    Args:
        frame (numpy.ndarray): The input frame (BGR image).

    Returns:
        frame_with_detections (numpy.ndarray): The frame with detected blobs highlighted.
        keypoints (list): A list of detected keypoints corresponding to blobs.
    """
    # Convert the frame to grayscale for blob detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optional: apply a blur to reduce noise (tune the kernel size as needed)
    gray_blurred = cv2.medianBlur(gray, 5)

    # Set up the detector with parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds (we search over the full grayscale range).
    params.minThreshold = 0
    params.maxThreshold = 256

    # Filter by color to detect dark blobs.
    params.filterByColor = True
    params.blobColor = 0  # 0 means dark

    # Filter by area: adjust these based on your expected mouse size in the frame.
    params.filterByArea = True
    params.minArea = 100  # Minimum area in pixels
    params.maxArea = 2000  # Maximum area in pixels

    # Optionally, you can enable filtering by circularity, convexity, or inertia.
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    # Create a detector with the parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(gray_blurred)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the circle size corresponds to the blob size.
    frame_with_detections = cv2.drawKeypoints(
        frame, keypoints, np.array([]),
        (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return frame_with_detections, keypoints


if __name__ == '__main__':
    # For testing this module independently:
    # Run the script with an image file path as an argument:
    #   python tracking.py path_to_image
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tracking.py path_to_image")
        sys.exit(1)

    image_path = sys.argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: could not load image from", image_path)
        sys.exit(1)

    output_frame, keypoints = track_mouse_in_frame(frame)
    print(f"Detected {len(keypoints)} potential mouse blobs.")

    # Display the result with highlighted keypoints.
    cv2.imshow("Mouse Tracking", output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
