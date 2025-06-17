import numpy as np
import torch.nn as nn
import cv2


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def judge_param(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total_params / 1e6:.2f}M')


def correct_preds(probs, labels, tol=-1):
    """
    通用评估函数：支持 DTL 和 FO 模式自动判断事件数量。
    默认最后一个类别为背景，其余为事件类别。

    Args:
        probs: np.array, shape (seq_len, num_classes)
        labels: np.array, shape (seq_len,)
        tol: int, optional, 动态容忍窗口（默认自动推断）

    Returns:
        events: ground truth 事件帧位置 (num_events,)
        preds: 预测帧位置 (num_events,)
        deltas: 偏差帧数 (num_events,)
        tol: 使用的容忍窗口
        correct: 逐事件准确与否 (num_events,)
    """

    num_classes = probs.shape[1]
    num_events = num_classes - 1

    events = np.where(labels < num_events)[0]

    if len(events) != num_events:
        raise ValueError(f"Expected {num_events} events, but got {len(events)}.")

    if tol == -1:
        if num_events > 8:
            tol = int(max(np.round((events[7] - events[0]) / 30), 1))
        else:
            tol = int(max(np.round((events[5] - events[0]) / 30), 1))

    preds = np.zeros(num_events)
    for i in range(num_events):
        preds[i] = np.argsort(probs[:, i])[-1]

    deltas = np.abs(events - preds)
    correct = (deltas <= tol).astype(np.uint8)

    return events, preds, deltas, tol, correct


def freeze_layers(num_freeze, net):
    i = 1
    for child in net.children():
        if i == 1:
            j = 1
            for child_child in child.children():
                if j <= num_freeze:
                    for param in child_child.parameters():
                        param.requires_grad = False
                j += 1
        i += 1


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def flip_video(input_path, output_path):
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_path}")
        return

    # 获取视频的基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出格式为 mp4

    # 初始化视频写入对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 逐帧读取并翻转视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 左右翻转帧
        flipped_frame = cv2.flip(frame, 1)  # 1 表示左右翻转

        # 写入翻转后的帧
        out.write(flipped_frame)

    # 释放资源
    cap.release()
    out.release()
    print(f"视频已成功翻转并保存到: {output_path}")
