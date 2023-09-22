import os

import cv2

from predictor import CRAFT
from utils.image_util import draw_bboxes


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="CRAFT Text Detection")
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument(
        "--image_path",
        required=True,
        type=str,
        help="folder path to input images",
    )
    parser.add_argument(
        "--output_dir",
        default="data/out",
        type=str,
        help="folder path to output images",
    )
    parser.add_argument(
        "--text_threshold",
        default=0.7,
        type=float,
        help="text confidence threshold",
    )
    parser.add_argument(
        "--link_threshold",
        default=0.4,
        type=float,
        help="link confidence threshold",
    )
    parser.add_argument(
        "--low_text",
        default=0.4,
        type=float,
        help="text low-bound score",
    )
    parser.add_argument(
        "--long_size",
        default=1280,
        type=int,
        help="max image size for inference",
    )
    parser.add_argument(
        "--poly",
        default=True,
        action="store_true",
        help="enable polygon type",
    )
    parser.add_argument(
        "--cuda",
        default=False,
        type=bool,
        help="Use cuda for inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Initialize model
    model = CRAFT(**args)
    # Predict
    result = model.predict(**args)
    # Draw bboxes
    image = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
    image = draw_bboxes(image=image, bboxes=result["boxes"])
    text_heatmap = result["heatmaps"]["text_score_heatmap"]
    link_heatmap = result["heatmaps"]["link_score_heatmap"]
    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, "image.jpg"), image)
    cv2.imwrite(os.path.join(args.output_dir, "text_heatmap.jpg"), text_heatmap)
    cv2.imwrite(os.path.join(args.output_dir, "link_heatmap.jpg"), link_heatmap)
    print("Done")
