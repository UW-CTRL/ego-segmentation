import argparse
from .core import run_segmentation_pipeline, run_live_pipeline

def main():
    parser = argparse.ArgumentParser(description="Egocentric Segmentation CLI")
    
    # Real-time trigger
    parser.add_argument("--live", action="store_true", help="Run live webcam tracking using Ultralytics YOLO")
    parser.add_argument("--safety", action="store_true", help="Enable safety features")
    # Offline parameters
    parser.add_argument("--sam2_checkpoint", type=str, help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, help="SAM2 config")
    parser.add_argument("--npz", type=str, help="Path to NPZ file")
    
    args = parser.parse_args()

    if args.live:
        run_live_pipeline()
        if args.safety:
            run_live_pipeline(safety_mode=True)
    else:
        # Pass the arguments to your existing offline pipeline
        if not args.sam2_checkpoint or not args.model_cfg:
            print("Error: Offline mode requires --sam2_checkpoint and --model_cfg")
            return
            
        run_segmentation_pipeline(
            sam2_checkpoint=args.sam2_checkpoint,
            model_cfg=args.model_cfg,
            data_npz_path=args.npz
        )

if __name__ == "__main__":
    main()