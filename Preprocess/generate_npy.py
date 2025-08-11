import os
import cv2
import numpy as np


def get_keypoint_module(mode):
    if mode == 'detectron2':
        from Preprocess.detectron2_keypoint import keypoint, keypoint_initialize
    elif mode == 'yolo':
        from Preprocess.yolo_keypoint import keypoint, keypoint_initialize
    else:
        raise ValueError(f"Unsupported mode: {mode}, choose 'detectron2' or 'yolo'")
    return keypoint, keypoint_initialize


def process_video(video_path, output_folder='', mode='detectron2'):
    keypoint_func, keypoint_initialize = get_keypoint_module(mode)
    predictor = keypoint_initialize()

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints_array = np.zeros((frame_count, 17, 3), dtype=np.float32)

    left_Shoulder_list = []
    left_Wrist_list = []

    frame_idx = 0
    first_frame_bbox = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == 0:
            box, keypoints = keypoint_func(frame, predictor, True)
            first_frame_bbox = box
        else:
            _, keypoints = keypoint_func(frame, predictor, first_frame_bbox=first_frame_bbox)

        if mode == 'detectron2':
            left_Shoulder_list.append(keypoints[5, 1] if keypoints is not None else 0)
            left_Wrist_list.append(keypoints[9, 1] if keypoints is not None else 0)

        if keypoints is not None:
            keypoints = np.array(keypoints, dtype=np.float32)
        else:
            keypoints = np.zeros((17, 3), dtype=np.float32)

        keypoints = np.nan_to_num(keypoints, nan=0.0)

        if keypoints.shape != (17, 3):
            print(f"Keypoints for frame {frame_idx} have incorrect shape: {keypoints.shape}. Setting to zeros.")
            keypoints = np.zeros((17, 3), dtype=np.float32)

        keypoints_array[frame_idx] = keypoints
        frame_idx += 1

    cap.release()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    npy_file_path = os.path.join(output_folder, f'{video_name}.npy')
    np.save(npy_file_path, keypoints_array)

    if mode == 'detectron2':
        return keypoints_array, left_Shoulder_list, left_Wrist_list
    else:
        return keypoints_array


def process_videos_in_folder(input_folder, output_folder, mode='detectron2'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(input_folder):
        if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.MP4')):
            print(f"Skipping non-video file: {video_file}")
            continue

        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        npy_file_path = os.path.join(output_folder, f'{video_name}.npy')

        if os.path.exists(npy_file_path):
            print(f"Skipping {video_file}: .npy file already exists.")
            continue

        print(f"Processing video: {video_file}")
        process_video(video_path, output_folder=output_folder, mode=mode)


if __name__ == '__main__':
    mode = 'yolo'  # or 'detectron2'
    input_folder = '/data/ssd1/xietingyu/frameflow_fo/videos'
    output_folder = '/data/ssd1/xietingyu/frameflow_fo/yolo_npy'

    process_videos_in_folder(input_folder, output_folder, mode=mode)
