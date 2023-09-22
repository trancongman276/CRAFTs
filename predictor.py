import cv2
import numpy as np
from openvino.runtime import Core

from utils.box_util import adjustResultCoordinates, getDetBoxes
from utils.image_util import (
    cvt2HeatmapImg,
    normalizeMeanVariance,
    read_image,
    resize_aspect_ratio,
    draw_bboxes,
)


class CRAFT:
    def __init__(self, model_path, cuda=False, *args, **kwargs) -> None:
        ie = Core()
        ie.set_property({"CACHE_DIR": "../cache"})
        model = ie.read_model(model_path)
        self.model = ie.compile_model(
            model=model, device_name="CPU" if not cuda else "GPU"
        )

    def preprocess(self, image, long_size=1280):
        # read/convert image
        image = read_image(image)

        # resize
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
            image, long_size, interpolation=cv2.INTER_LINEAR
        )

        # preprocessing
        x = normalizeMeanVariance(img_resized)
        x = np.transpose(x, (2, 0, 1))  # [h, w, c] to [c, h, w]
        x = np.expand_dims(x, axis=0)  # [c, h, w] to [b, c, h, w]
        return x, target_ratio

    def postprocess(
        self,
        origin_image,
        score_text,
        score_link,
        ratio,
        poly,
        text_threshold,
        link_threshold,
        low_text,
        **kwargs,
    ):
        boxes, polys = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly
        )

        # coordinate adjustment
        ratio_h = ratio_w = 1 / ratio
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        # get image size
        img_height = origin_image.shape[0]
        img_width = origin_image.shape[1]

        # calculate box coords as ratios to image size
        boxes_as_ratio = []
        for box in boxes:
            boxes_as_ratio.append(box / [img_width, img_height])
        boxes_as_ratio = np.array(boxes_as_ratio)

        # calculate poly coords as ratios to image size
        polys_as_ratio = []
        for poly in polys:
            polys_as_ratio.append(poly / [img_width, img_height])
        polys_as_ratio = np.array(polys_as_ratio)

        text_score_heatmap = cvt2HeatmapImg(score_text)
        link_score_heatmap = cvt2HeatmapImg(score_link)
        return {
            "boxes": boxes,
            "boxes_as_ratios": boxes_as_ratio,
            "polys": polys,
            "polys_as_ratios": polys_as_ratio,
            "heatmaps": {
                "text_score_heatmap": text_score_heatmap,
                "link_score_heatmap": link_score_heatmap,
            },
        }

    def predict(
        self,
        image,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        long_size: int = 1280,
        poly: bool = True,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            image: path to the image to be processed or numpy array or PIL image
            output_dir: path to the results to be exported
            craft_net: craft net model
            refine_net: refine net model
            text_threshold: text confidence threshold
            link_threshold: link confidence threshold
            low_text: text low-bound score
            cuda: Use cuda for inference
            canvas_size: image size for inference
            long_size: desired longest image size for inference
            poly: enable polygon type
        Output:
            {"masks": lists of predicted masks 2d as bool array,
            "boxes": list of coords of points of predicted boxes,
            "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
            "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
            "heatmaps": visualizations of the detected characters/links,
            "times": elapsed times of the sub modules, in seconds}
        """
        config = {
            "text_threshold": text_threshold,
            "link_threshold": link_threshold,
            "low_text": low_text,
            "long_size": long_size,
            "poly": poly,
            "origin_image": image,
        }
        # Preprocess
        x, ratio = self.preprocess(image, long_size=long_size)

        # forward pass
        y, y_refined = self.model(x).values()

        # make score and link map
        score_text = y[0, :, :, 0]
        score_link = y_refined[0, :, :, 0]

        config.update(
            {
                "score_text": score_text,
                "score_link": score_link,
                "ratio": ratio,
            }
        )
        # Post-processing
        result = self.postprocess(**config)
        result.update({"image": image})
        return result


if __name__ == "__main__":
    image = read_image("images/input.png")
    craft = CRAFT("weights/crafts.xml")
    result = craft.predict(image)
    image = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
    image = draw_bboxes(image, result["boxes"])
    cv2.imshow("result", image), cv2.waitKey(0)
