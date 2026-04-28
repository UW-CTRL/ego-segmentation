import numpy as np
import cv2
from .utils import (
    IoU, extract_frames, get_timestamp, dict_builder, 
    check_if_already_segmented, build_object_box, build_gaze_box, process_data,
    connect_to_pupil, get_data_live
)
from .models import setup_device, load_models, load_live_model

def run_segmentation_pipeline(sam2_checkpoint, model_cfg, detection_threshold=0.3, elapse_threshold=10, data_npz_path=None, video_path=None, csv_path=None):    
    """Run the full egocentric segmentation pipeline over input frames.

    The pipeline loads models, prepares frame and gaze data, identifies gaze-aligned
    detections, seeds SAM2 prompts, propagates masks through the video, and returns
    per-frame object metadata and segmentation masks.

    Args:
        sam2_checkpoint: Path to the SAM2 checkpoint.
        model_cfg: SAM2 model configuration path or identifier.
        detection_threshold: Minimum IoU between object and gaze box to trigger
            segmentation.
        elapse_threshold: Maximum number of frames an object can remain unseen
            before being removed.
        data_npz_path: Optional path to NPZ file with scene and gaze arrays.
        video_path: Optional path to source video when NPZ is not provided.
        csv_path: Optional path to gaze CSV paired with video input.

    Returns:
        tuple: (objects_key, video_segments), where objects_key stores per-frame
        object metadata and video_segments stores propagated mask arrays.
    """
    # Setup and load
    device = setup_device()
    predictor, sam2, yolo_model = load_models(sam2_checkpoint, model_cfg, device)

    scene_arr, gaze_data = process_data(data_npz_path, video_path, csv_path)

    video_folder, tempdir = extract_frames(scene_arr)

    try:
        # Algorithm Initialization
        inference_state = predictor.init_state(video_folder)
        prompts = {}  # Holds all the clicks for visualization
        video_segments = {}  # Stores per-frame segmentation results
        objects_key = {}
        ann_obj_id = 0

        object_name_mapping = {0: "human",
                            15: "cat",
                            16: "dog",
                            17: "horse",
                            18: "sheep",
                            19: "cow",
                            20: "elephant"}  
        
        for frame_idx, frame in enumerate(scene_arr):
            # Run YOLO inference (Fixed variable name)
            results = yolo_model(frame)
            detected_objects = results.xyxy[0].cpu().numpy()

            gaze_x = int(gaze_data[frame_idx, 0])
            gaze_y = int(gaze_data[frame_idx, 1])

            if frame_idx not in objects_key:
                dict_builder(objects_key, frame_idx)

            if len(detected_objects) == 0:
                continue
            
            # Flag to track if we need to run propagation for this frame
            new_prompts_added = False
            
            for obj in detected_objects:
                object_type = int(obj[5])  # Assuming class ID is at index 5
                object_box, object_center = build_object_box(obj)
                gaze_box = build_gaze_box(gaze_x, gaze_y)

                # Check if gaze is within the bounding box
                if IoU(object_box, gaze_box) > detection_threshold:
                    x = object_center[0]
                    y = object_center[1]
                    print(f"gaze x:{x}, gaze y:{y}, frame:{frame_idx}")
                    
                    points = np.array([[x, y]])
                    labels = np.ones((len(points),), dtype=np.int32)
                    time = get_timestamp(frame_idx)

                    already_segmented, object_id = check_if_already_segmented(frame_idx, x, y, video_segments)
                    
                    if already_segmented:
                        # Protect against Frame 0 KeyError
                        if frame_idx > 0 and object_id in objects_key[frame_idx - 1]:
                            first_seen_time = objects_key[frame_idx - 1][object_id]["First Seen (Time)"]
                            first_seen_frame = objects_key[frame_idx - 1][object_id]["First Seen (Frame)"]
                        else:
                            first_seen_time = time
                            first_seen_frame = frame_idx
                            
                        objects_key[frame_idx][object_id] = {
                            "First Seen (Time)": first_seen_time,
                            "First Seen (Frame)": first_seen_frame,
                            "Last Seen (Time)": time,
                            "Last Seen (Frame)": frame_idx,
                            "object_type": object_name_mapping.get(object_type, "unknown"),
                            "BBox": object_box,
                        }
                        continue
                    else:
                        # Save current ID before incrementing to avoid mismatch
                        current_obj_id = ann_obj_id
                        
                        # Add new object to the key
                        objects_key[frame_idx][current_obj_id] = {
                            "First Seen (Time)": time,
                            "First Seen (Frame)": frame_idx,
                            "Last Seen (Time)": time,
                            "Last Seen (Frame)": frame_idx,
                            "object_type": object_name_mapping.get(object_type, "unknown"),
                            "BBox": object_box,
                        }
                        ann_obj_id += 1

                        # Store points and labels
                        prompts[current_obj_id] = points, labels
                        
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=current_obj_id,
                            points=points,
                            labels=labels,
                        )
                        new_prompts_added = True

            # Run propagation ONCE per frame if new points were added
            if new_prompts_added:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    if out_frame_idx not in video_segments:
                        video_segments[out_frame_idx] = {}
                        
                    for j, out_obj_id in enumerate(out_obj_ids):
                        video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[j] > 0.0).cpu().numpy()

        # Add the masks to the objects_key
        for frame_idx in range(len(scene_arr)):
            if frame_idx not in video_segments:
                print(f"Frame {frame_idx} not found in video_segments.")
                continue
                
            # Wrap in list() to avoid dictionary size changed during iteration error
            for id in list(objects_key[frame_idx].keys()):
                mask_data = video_segments[frame_idx].get(id)
                if mask_data is not None:
                    mask = np.squeeze(mask_data)
                    objects_key[frame_idx][id]["Mask"] = mask.astype(float)

                ## Time elapse removal logic (can be adjusted based on expected object disappearance time)
                saw_elapsed_time = frame_idx - objects_key[frame_idx][id]["Last Seen (Frame)"]
                if saw_elapsed_time > elapse_threshold:
                    print(f"Object {id} in frame {frame_idx} was last seen at frame {objects_key[frame_idx][id]['Last Seen (Frame)']} and will be removed from the key.")
                    del objects_key[frame_idx][id]
    finally:
        tempdir.cleanup()  # Clean up temporary directory
    print("Pipeline Complete")
    return objects_key, video_segments

def run_live_pipeline():
    """Real-time pipeline using Ultralytics tracking and mouse-simulated gaze."""
    
    # Use your custom setup to get CUDA/MPS/CPU
    device = setup_device()

    # Initialize YOLOv8 segmentation model (it will auto-download the tiny weights)
    print("Loading Ultralytics model...")
    model = load_live_model(device)

    # Connect to Pupil Labs eye tracker and start receiving gaze data
    pupil_device = connect_to_pupil()
    frame, gx, gy = get_data_live(pupil_device)

    print("Starting live feed. Press 'q' to quit.")

    while True:
        frame, gx, gy = get_data_live(pupil_device)
        if frame is None:
            break

        # Run tracking and segmentation on the current frame
        results = model.track(frame, persist=True)
        
        # Plot the native Ultralytics annotations
        annotated_frame = results[0].plot()

        # Overlay the simulated gaze cursor as a red dot
        cv2.circle(annotated_frame, (gx, gy), 8, (0, 0, 255), -1)

        cv2.imshow('Live Egocentric Tracking', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()