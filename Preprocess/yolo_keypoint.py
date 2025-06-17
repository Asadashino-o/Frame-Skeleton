import numpy as np
from ultralytics import YOLO
import cv2


def keypoint_initialize():
    model = YOLO("./weights/yolov11l-pose.pt")
    return model


def keypoint(frame, model, is_first_frame=False, first_frame_bbox=None, alpha=0.0):
    height, width = frame.shape[:2]
    frame_center = np.array([(width / 2) - alpha * width, height / 2])

    results = model(frame)
    result = results[0]

    bboxes = result.boxes.xyxy.cpu().numpy()

    scores = result.boxes.conf.cpu().numpy()

    keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None

    if is_first_frame:
        min_distance = float('inf')
        closest_bbox = None
        closest_keypoints = None

        for i, bbox in enumerate(bboxes):
            bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            distance = np.linalg.norm(bbox_center - frame_center)

            if scores[i] > 0.8 and distance < min_distance:
                min_distance = distance
                closest_bbox = bbox
                closest_keypoints = keypoints[i]

        if closest_bbox is None:
            raise ValueError("No instance found with confidence > 80%.")

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
    image_path = "00.jpg"
    image = cv2.imread(image_path)
    predictor = keypoint_initialize()

    first_frame_bbox, first_frame_keypoints = yolo_keypoint(image, predictor, is_first_frame=True, alpha=0.2)

    print(first_frame_keypoints)