import os

import cv2
import numpy as np

# COCO 17 个关键点的连线关系
# 关节点的名称参考 COCO 数据集
COCO_PAIRS = [
    (16, 14), (15, 13), (14, 12), (13, 11), (12, 11), (12, 6), (11, 5),
    (6, 5), (6, 8), (5, 7), (8, 10), (7, 9), (0, 5), (0, 6), (0, 1), (0, 2),
    (2, 1), (2, 4), (1, 3)
]



# 在帧上绘制关键点及连线的函数
def draw_keypoints(frame, keypoints, pairs, threshold=0.1):
    # 绘制每个关键点
    for i, (x, y , _) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # 绘制每两个关键点之间的连线
    for pair in pairs:
        pt1, pt2 = pair
        if keypoints[pt1][0] > 0 and keypoints[pt1][1] > 0 and keypoints[pt2][0] > 0 and keypoints[pt2][1] > 0:
            cv2.line(frame, (int(keypoints[pt1][0]), int(keypoints[pt1][1])),
                     (int(keypoints[pt2][0]), int(keypoints[pt2][1])), (255, 0, 0), 2)


# 生成带有关键点和连线的验证视频
def generate_keypoint_video(video_path, keypoints_npy_path, output_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧的宽高和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 打开保存视频的对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 加载关键点数据 (帧数, 17, 3)
    keypoints_data = np.load(keypoints_npy_path)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 获取对应帧的关键点
        keypoints = keypoints_data[frame_idx]

        # 在帧上绘制关键点和连线
        draw_keypoints(frame, keypoints, COCO_PAIRS)

        # 写入新的帧到输出视频
        out.write(frame)

        frame_idx += 1
        if frame_idx >= len(keypoints_data):
            break

    # 释放视频对象
    cap.release()
    out.release()

def generate_keypoint_videofolder(input_folder, keypoints_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # 遍历输入文件夹中的所有文件
    for video_file in os.listdir(input_folder):
        # 构建完整的视频文件路径
        video_path = os.path.join(input_folder, video_file)
        # 获取输出文件夹中对应的 .npy 文件路径
        video_name = os.path.splitext(video_file)[0]
        npy_file_path = os.path.join(keypoints_folder, f'{video_name}.npy')
        output_file_path =  os.path.join(output_folder, f'output_{video_name}.mp4')
        generate_keypoint_video(video_path,npy_file_path,output_file_path)
        print(f"generate {video_name}.mp4")


# 使用示例
generate_keypoint_videofolder("D:/learning/dataset/golfdb160/face-on/fo_video",
                              "D:/learning/dataset/golfdb160/face-on/fo_keypoint",
                              "D:/learning/dataset/golfdb160/face-on/fo_output")
