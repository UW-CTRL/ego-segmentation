import os
import torch
from ultralytics import YOLO

def setup_device():
    """Select and configure the best available Torch device.

    Args:
        None.

    Returns:
        torch.device: CUDA, MPS, or CPU device in that priority order.
    """
    # If using Apple MPS, fall back to CPU for unsupported ops
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    return device

def load_live_model(device):
    """Load the Ultralytics segmentation model for real-time tracking."""
    print(f"Loading YOLOv8 segmentation model on {device}...")
    
    # Load the nano segmentation model for the best real-time framerate
    model = YOLO('yolov8n-seg.pt')
    
    # Move the model to the optimal hardware device
    model.to(device)
    
    return model

def load_models(sam2_checkpoint, model_cfg, device):
    """Load SAM2 predictor/models and a YOLOv5 detector onto the given device.

    Args:
        sam2_checkpoint: Path to the SAM2 checkpoint.
        model_cfg: SAM2 model configuration path or identifier.
        device: Torch device to place loaded models on.

    Returns:
        tuple: (predictor, sam2, yolo_model).
    """
    from sam2.build_sam import build_sam2_video_predictor, build_sam2

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    
    # Load YOLOv5s
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.to(device)

    # Update the classes to include person (0) and animals (15 to 20)
    yolo_model.classes = [0, 15, 16, 17, 18, 19, 20]
    
    # Keep the strict confidence threshold
    yolo_model.conf = 0.90
    
    return predictor, sam2, yolo_model