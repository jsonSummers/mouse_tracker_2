# dataloader.py
import cv2
import os


class DataLoader:
    """
    A memory efficient dataloader that:
      - Extracts the FPS from the original video file without loading all frames.
      - Uses the FPS to convert a given start time (in seconds) into a frame index.
      - Loads frames (from a folder of cropped frames) corresponding to a duration (in minutes)
        starting from that time, much like a deep learning data loader.
    """

    def __init__(self, video_path, cropped_frames_folder):
        """
        Args:
            video_path (str): Path to the original video file (used to extract FPS).
            cropped_frames_folder (str): Folder where the cropped frame images are stored.
        """
        self.video_path = video_path
        self.cropped_frames_folder = cropped_frames_folder

        # Extract FPS from the video file without loading the whole video.
        self.fps = self._get_video_fps(video_path)
        if self.fps is None:
            raise ValueError(f"Failed to extract FPS from video: {video_path}")

        # Prepare a sorted list of frame file paths from the cropped frames folder.
        self.frame_paths = self._get_sorted_frame_paths(cropped_frames_folder)
        self.total_frames = len(self.frame_paths)

    def _get_video_fps(self, video_path):
        """Open the video file briefly to extract the FPS."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def _get_sorted_frame_paths(self, folder):
        """Return a lexicographically sorted list of image file paths from the folder."""
        valid_extensions = ('.png', '.jpg', '.jpeg')
        frame_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(valid_extensions)
        ]
        # Assumes the filenames are zero-padded so that lexicographical order is numerical.
        frame_files.sort()
        full_paths = [os.path.join(folder, f) for f in frame_files]
        return full_paths

    def get_frame_by_index(self, index):
        """
        Load a single frame image by its index in the sorted list.

        Args:
            index (int): Index of the frame to load.

        Returns:
            frame (numpy.ndarray): The loaded image frame.
        """
        if index < 0 or index >= self.total_frames:
            raise IndexError("Frame index out of range.")
        frame_path = self.frame_paths[index]
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not read frame from {frame_path}")
        return frame

    def load_frames(self, start_seconds, duration_minutes):
        """
        Yield frames corresponding to a specified time interval.

        Args:
            start_seconds (float): The starting time in seconds.
            duration_minutes (float): How many minutes of frames to load.

        Yields:
            frame (numpy.ndarray): Next frame in the sequence.
        """
        # Compute the starting frame index using the FPS.
        start_index = int(start_seconds * self.fps)
        # Compute how many frames to load for the given duration.
        num_frames = int(duration_minutes * 60 * self.fps)
        end_index = start_index + num_frames

        if start_index >= self.total_frames:
            raise ValueError("The start_seconds is beyond the available frames.")

        # Adjust end_index if it goes beyond the available frames.
        if end_index > self.total_frames:
            end_index = self.total_frames

        for idx in range(start_index, end_index):
            yield self.get_frame_by_index(idx)

    def iter_batches(self, start_seconds, duration_minutes, batch_size=32):
        """
        Yield batches of frames for analysis (optional deep learning style batching).

        Args:
            start_seconds (float): The starting time in seconds.
            duration_minutes (float): Duration (in minutes) of frames to load.
            batch_size (int): Number of frames per batch.

        Yields:
            batch (list of numpy.ndarray): A list containing a batch of frames.
        """
        batch = []
        for frame in self.load_frames(start_seconds, duration_minutes):
            batch.append(frame)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch