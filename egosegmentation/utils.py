import os
import cv2
import csv
import copy
import tempfile
import numpy as np
from pupil_labs.realtime_api.simple import discover_one_device


def IoU(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First box as (x, y, w, h).
        box2: Second box as (x, y, w, h).

    Returns:
        float: IoU score in the range [0, 1].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x_min = max(x1, x2)
    y_min = max(y1, y2)
    x_max = min(x1+w1, x2+w2)
    y_max = min(y1+h1, y2+h2)
    
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def get_timestamp(frame_idx, fps=30):
    """Convert frame index to a mm:ss.mmm timestamp string.

    Args:
        frame_idx: Zero-based frame index.
        fps: Frames per second of the source video.

    Returns:
        str: Timestamp formatted as mm:ss.mmm.
    """
    time = frame_idx / fps
    minutes = int(time // 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def dict_builder(objects_key, frame_idx):
    """Initialize or copy frame-level object metadata for the current frame.

    Args:
        objects_key: Dictionary storing tracked objects per frame.
        frame_idx: Current frame index.

    Returns:
        None.
    """
    if frame_idx == 0:
        objects_key[frame_idx] = {}
    else:
        objects_key[frame_idx] = copy.deepcopy(objects_key[frame_idx - 1])

def check_if_already_segmented(frame_idx, x, y, video_segments):
    """Check whether a pixel lies inside any existing mask at a frame.

    Args:
        frame_idx: Current frame index to inspect.
        x: Pixel x-coordinate.
        y: Pixel y-coordinate.
        video_segments: Nested dictionary of frame and object masks.

    Returns:
        tuple: (is_segmented, object_id).
    """
    if frame_idx > 0:
        if frame_idx in video_segments:
            for obj_id in video_segments[frame_idx].keys():
                mask = np.squeeze(video_segments[frame_idx].get(obj_id))
                if mask[y, x]:
                    return True, obj_id
    return False, 0

def build_object_box(obj_detection):
    """Convert detector output to box format and center point.

    Args:
        obj_detection: Sequence in format [x1, y1, x2, y2, conf, cls].

    Returns:
        tuple: ((x, y, w, h), (cx, cy)).
    """
    x1, y1, x2, y2, _, _ = obj_detection
    w, h = x2 - x1, y2 - y1
    center = (int(x1 + w/2), int(y1 + h/2))
    return (x1, y1, w, h), center

def build_gaze_box(gaze_x, gaze_y, box_size=50):
    """Build a fixed-size gaze-centered box constrained to non-negative coords.

    Args:
        gaze_x: Gaze x-coordinate in pixels.
        gaze_y: Gaze y-coordinate in pixels.
        box_size: Width and height of the square gaze box.

    Returns:
        tuple: Gaze box as (x, y, w, h).
    """
    x = max(0, gaze_x - box_size // 2)
    y = max(0, gaze_y - box_size // 2)
    return (x, y, box_size, box_size)

def extract_frames(scene_arr):
    """Save scene frames as JPEG files in a temporary folder.

    Args:
        scene_arr: Iterable of image frames.

    Returns:
        tuple: (tempdir_path, tempdir) where tempdir_path is the folder path and
        tempdir is the TemporaryDirectory handle.
    """

    tempdir = tempfile.TemporaryDirectory()
    tempdir_path = tempdir.name
    for i, frame in enumerate(scene_arr):
        image_path = os.path.join(tempdir_path, f"{i:04d}.jpg")
        cv2.imwrite(image_path, frame)
    return tempdir_path, tempdir

def load_video_frames(video_path):
    """Load all frames from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        list: Frames loaded from the video in display order.
    """
    
    cap = cv2.VideoCapture(video_path)
    scene_arr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        scene_arr.append(frame)
    cap.release()
    return scene_arr

def load_gaze_data(gaze_csv_path):
    """Load gaze coordinates from CSV into a NumPy array.

    Args:
        gaze_csv_path: Path to CSV file with gaze x and y values per row.

    Returns:
        numpy.ndarray: Array with shape (n, 2) of gaze coordinates.
    """
    gaze_data = []
    with open(gaze_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            gaze_data.append([float(row[0]), float(row[1])])
    return np.array(gaze_data)

def process_npz(npz_path):
    """Load scene and gaze arrays from an NPZ file.

    Args:
        npz_path: Path to NPZ file containing scene and gaze keys.

    Returns:
        tuple: (scene_arr, gaze_data).
    """
    data_dict = np.load(npz_path)
    scene_arr = data_dict['scene']
    gaze_data = data_dict['gaze']
    return scene_arr, gaze_data

def path_checker(data_npz_path, video_path, csv_path):
    """Validate mutually exclusive and required input path combinations.

    Args:
        data_npz_path: Optional NPZ path.
        video_path: Optional video path.
        csv_path: Optional gaze CSV path.

    Returns:
        None.

    Raises:
        ValueError: If an invalid path combination is provided.
    """
    if data_npz_path is not None:
        if csv_path is not None or video_path is not None:
            raise ValueError("When using data_npz_path, video_path and csv_path should be None.")
    elif video_path is not None:
        if csv_path is None or data_npz_path is not None:
            raise ValueError("video_path and csv_path cannot be None when data_npz_path is not provided.")
    else:
        raise ValueError("Either data_npz_path or video_path must be provided.")
    
def process_data(data_npz_path, video_path, csv_path):
    """Load scene and gaze data from either NPZ or video+CSV inputs.

    Args:
        data_npz_path: Optional NPZ path containing scene and gaze arrays.
        video_path: Video path used when NPZ is not provided.
        csv_path: Gaze CSV path used with video input.

    Returns:
        tuple: (scene_arr, gaze_data).
    """
    path_checker(data_npz_path, video_path, csv_path)

    if data_npz_path is not None:
        scene_arr, gaze_data = process_npz(data_npz_path)
    else:
        scene_arr = load_video_frames(video_path, "temp_frames")
        gaze_data = load_gaze_data(csv_path)
    return scene_arr, gaze_data

def connect_to_pupil():
    """Establish connection to the Pupil Labs eye tracker and return gaze state."""
    # Look for devices. Returns as soon as it has found the first device.
    print("Looking for the next best device...")
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        print("No device found. Are the glasses on and connected to the same network? Exiting.")
        raise SystemExit(-1)
    return device

def get_data_live(device):
    frame, gaze = device.receive_matched_scene_video_frame_and_gaze()
    if frame is not None and gaze is not None:
        # Extract gaze coordinates
        x = int(gaze.x)
        y = int(gaze.y)
        return frame.bgr_pixels, int(x), int(y)
    else:
        print("Failed to receive matched scene video frame and gaze data. \
                         Are the glasses on and connected to the same network?")
        raise SystemExit(-1)

