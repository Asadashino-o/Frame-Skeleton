import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch
import cv2


def keypoint_initialize():
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "./weights/model_final_a6e10b.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor


def keypoint(frame, predictor, is_first_frame=False, first_frame_bbox=None, alpha=0.0):
    height, width = frame.shape[:2]
    frame_center = np.array([(width / 2) - alpha * width, height / 2])

    with torch.no_grad():
        outputs = predictor(frame)

    instances = outputs["instances"]
    bboxes = instances.pred_boxes.tensor.cpu().numpy()
    keypoints = instances.pred_keypoints.cpu().numpy()
    scores = instances.scores.cpu().numpy()

    if is_first_frame:
        min_distance = float('inf')
        closest_bbox = None
        closest_keypoints = None

        for i, bbox in enumerate(bboxes):
            bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            distance = np.linalg.norm(bbox_center - frame_center)

            if scores[i] > 0.9 and distance < min_distance:
                min_distance = distance
                closest_bbox = bbox
                closest_keypoints = keypoints[i]

        if closest_bbox is None:
            raise ValueError("No instance found with confidence > 90%.")

        first_frame_bbox = closest_bbox
        first_frame_keypoints = closest_keypoints

        return first_frame_bbox, first_frame_keypoints

    min_distance = float('inf')
    closest_bbox = None
    closest_keypoints = None

    first_bbox_center = np.array([(first_frame_bbox[0] + first_frame_bbox[2]) / 2,
                                  (first_frame_bbox[1] + first_frame_bbox[3]) / 2])

    for i, bbox in enumerate(bboxes):
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        distance = np.linalg.norm(bbox_center - first_bbox_center)

        if distance < min_distance:
            min_distance = distance
            closest_bbox = bbox
            closest_keypoints = keypoints[i]

    return closest_bbox, closest_keypoints


if __name__ == '__main__':
    image_path = "./123456.jpg"
    image = cv2.imread(image_path)
    predictor = keypoint_initialize()

    first_frame_bbox, first_frame_keypoints = keypoint(image, predictor, is_first_frame=True, alpha=0.2)

    print(first_frame_keypoints)