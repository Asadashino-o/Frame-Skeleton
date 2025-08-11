import argparse
import glob
import os
import time

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from dataloader_fo import ToTensor, Normalize
from model import EventDetector
from generate_npy import process_video
from key_point import keypoint_initialize


class SampleVideo(Dataset):
    def __init__(self, path, keypoints, input_size=160, transform=None):
        self.path = path
        self.keypoints = keypoints
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        keypoints = self.keypoints
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        keypoints[:, :, 0] /= frame_size[1]  # Normalize x to [0, 1]
        keypoints[:, :, 1] /= frame_size[0]  # Normalize y to [0, 1]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return f
        images, keys = [], []
        pos = 0
        while True:
            ret, img = cap.read()

            if not ret:  # 检查是否成功读取帧
                # print(f"Warning: {a['id']}'s Frame {pos} could not be read. Stopping the loop.")
                break  # 读取失败，退出循环

            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
            keys.append(keypoints[pos])

            pos += 1  # 手动增加帧数计数器
        cap.release()
        labels = np.zeros(len(images))  # only for compatibility with transforms
        sample = {'images': np.asarray(images), 'keys': np.asarray(keys), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def test(video_path):
    global probs, save_dict
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default=video_path)
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
    args = parser.parse_args()
    seq_length = args.seq_length
    predictor = keypoint_initialize()
    device = 'cuda:2'

    print('Preparing video: {}'.format(args.path))
    keypoints, _, _ = process_video(args.path, predictor, os.path.dirname(args.path))

    ds = SampleVideo(args.path, keypoints, transform=transforms.Compose([ToTensor(),
                                                                         Normalize([0.485, 0.456, 0.406],
                                                                                   [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    try:
        save_dict = torch.load('./APP_fo_models//FS_FO_10800.pth.tar')
    except:
        print(
            "Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

    print('Using device:', device)
    model = EventDetector(pretrain=False,
                          width_mult=1.,
                          lstm_layers=2,
                          lstm_hidden=256,
                          num_classes=12,
                          fused_dim=256,
                          device=device
                          )
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")
    print('Testing...')

    for sample in dl:
        images, keys = sample['images'].to(device), sample['keys'].to(device)
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
                key_batch = keys[:, batch * seq_length:, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                key_batch = keys[:, batch * seq_length:(batch + 1) * seq_length, :, :]
            logits = model(image_batch.to(device), key_batch.to(device))
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Confidence: {}'.format([np.round(c, 3) for c in confidence]))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理 GPU 缓存")

    return events, confidence


def process_videos(file_path, main_output_dir):
    """处理单个视频文件"""
    start = time.time()
    file_name = os.path.basename(file_path)

    # 调用检测函数（假设test函数已定义）
    events, confidence = test(file_path)

    # 创建视频对应的输出子目录
    video_output_dir = os.path.join(main_output_dir, f'output_{os.path.splitext(file_name)[0]}')
    os.makedirs(video_output_dir, exist_ok=True)

    # 处理视频帧
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件 {file_path}")

    try:
        for i, e in enumerate(events):
            cap.set(cv2.CAP_PROP_POS_FRAMES, e)
            ret, img = cap.read()
            if not ret:
                print(f"警告：无法读取帧 {e} @ {file_name}")
                continue

            # 添加置信度文本并保存
            cv2.putText(img, f'{confidence[i]:.3f}', (20, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
            cv2.imwrite(os.path.join(video_output_dir, f'frame_{e}.jpg'), img)

        process_time = time.time() - start
        print(f"[{file_name}] 处理完成，耗时 {process_time:.2f} 秒")
        print(f"生成 {len(events)} 张图片到 {video_output_dir}\n{'-' * 50}")
        return True
    finally:
        cap.release()


if __name__ == '__main__':
    INPUT_DIR = "./data/ex_fo/"  # 视频存放目录
    MAIN_OUTPUT_DIR = "./data/ex_fo_output"  # 主输出目录
    VIDEO_EXTS = ['*.mp4', '*.avi', '*.mov', '*.mkv']  # 支持的视频格式

    # 创建主输出目录
    os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

    # 获取视频文件列表
    video_files = []
    for ext in VIDEO_EXTS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not video_files:
        print(f"在 {INPUT_DIR} 中未找到视频文件")
        exit()

    print(f"发现 {len(video_files)} 个待处理视频")
    print(f"所有结果将保存到: {os.path.abspath(MAIN_OUTPUT_DIR)}")
    total_start = time.time()

    # 批量处理视频
    success_count = 0
    for idx, file_path in enumerate(video_files, 1):
        try:
            print(f"\n处理进度 ({idx}/{len(video_files)}): {os.path.basename(file_path)}")
            if process_videos(file_path, MAIN_OUTPUT_DIR):
                success_count += 1
        except Exception as e:
            print(f"处理失败: {os.path.basename(file_path)}\n错误信息: {str(e)}")

    # 输出总结报告
    total_time = time.time() - total_start
    print(f"\n{'=' * 50}")
    print(f"处理完成 {success_count}/{len(video_files)} 个视频")
    print(f"总耗时: {total_time:.2f} 秒 (平均 {total_time / len(video_files):.2f} 秒/个)")
    print(f"所有结果已保存到: {os.path.abspath(MAIN_OUTPUT_DIR)}")
