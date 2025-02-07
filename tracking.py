# tracking.py

import cv2
import numpy as np


class MouseTracker:
    def __init__(self, min_area=100):
        """
        Initialize the tracker with a background subtractor.
        :param min_area: Minimum contour area to be considered valid motion.
        """
        # Create a background subtractor. Tweak history and varThreshold as needed.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.min_area = min_area
        self.last_coordinate = None

    def track_frame(self, frame):
        """
        Process a single frame to detect motion. Only the largest valid contour is tracked.

        :param frame: The current video frame.
        :return: Tuple of (annotated frame, list containing a single keypoint tuple)
        """
        # Apply the background subtractor to obtain the foreground mask.
        fg_mask = self.bg_subtractor.apply(frame)

        # Threshold the mask to remove shadows (which appear as gray regions).
        _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

        # Use morphological operations to remove noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Find contours in the mask.
        contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables for tracking the largest contour.
        largest_contour = None
        largest_area = 0

        # Loop through contours and select the largest one that exceeds min_area.
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue  # Skip small areas that are likely noise.
            if area > largest_area:
                largest_area = area
                largest_contour = cnt

        # If a valid contour is found, compute its centroid.
        if largest_contour is not None:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.last_coordinate = (cX, cY)

                # Draw a red circle at the centroid.
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

                # Draw a green rectangle around the contour.
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                keypoints = [(cX, cY)]
            else:
                keypoints = []
        else:
            # No valid motion detected; optionally, mark the last known coordinate.
            if self.last_coordinate is not None:
                cv2.circle(frame, self.last_coordinate, 5, (255, 0, 0), -1)
                keypoints = [self.last_coordinate]
            else:
                keypoints = []

        return frame, keypoints


def track_mouse_in_frame(frame, tracker=None):
    """
    A convenience function to process a single frame.
    If no tracker is passed, a new MouseTracker is instantiated.

    :param frame: The frame to process.
    :param tracker: (Optional) an existing MouseTracker instance.
    :return: Tuple of (annotated frame, list with a single keypoint)
    """
    if tracker is None:
        tracker = MouseTracker()
    return tracker.track_frame(frame)
