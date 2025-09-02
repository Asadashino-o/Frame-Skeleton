import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import EventDetector
from Preprocess.generate_npy import process_video


class ToTensor(object):
    def __call__(self, sample):
        images, keys, labels = sample['images'], sample['keys'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'keys': torch.from_numpy(keys).float(),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, keys, labels = sample['images'], sample['keys'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'keys': keys, 'labels': labels}


class SampleVideo(Dataset):
    def __init__(self, path, keypoints, input_size=160, transform=None):
        self.path = path
        self.keypoints = keypoints
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        self.keypoints[:, :, 0] /= frame_size[1]
        self.keypoints[:, :, 1] /= frame_size[0]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images, keys = [], []
        pos = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
            keys.append(self.keypoints[pos])
            pos += 1
        cap.release()
        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'keys': np.asarray(keys), 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample


def test(video_path, mode='detectron2', task='dtl'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default=video_path)
    parser.add_argument('-s', '--seq-length', type=int, default=64)
    args = parser.parse_args()
    seq_length = args.seq_length

    print(f'Preparing video: {args.path} using {mode}...')
    result = process_video(args.path, output_folder='', mode=mode)
    keypoints = result[0] if isinstance(result, tuple) else result

    ds = SampleVideo(args.path, keypoints, transform=transforms.Compose([
        ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    try:
        save_dict = torch.load('/path/to/your/weight/file')
    except:
        print("Model weights not found.")
        return

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 设置类别数
    num_classes = 9 if task == 'dtl' else 12

    model = EventDetector(pretrain=False, width_mult=1., lstm_layers=2,
                          lstm_hidden=256, num_classes=num_classes, fused_dim=256, device=device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()

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
            logits = model(image_batch, key_batch)
            softmax_out = F.softmax(logits.data, dim=1).cpu().numpy()
            probs = softmax_out if batch == 0 else np.append(probs, softmax_out, 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames:', events)
    print('Confidence:', [np.round(probs[e, i], 3) for i, e in enumerate(events)])
    return events, args.path


if __name__ == '__main__':
    file_path = './data/fo.mov'
    mode = 'detectron2'  # or 'yolo'
    events, _ = test(file_path, mode=mode, task='fo')

    output_dir = f'./data/output_images_{os.path.basename(file_path)}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(file_path)
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        image_filename = os.path.join(output_dir, f'{i}th_keyframe_{e}.jpg')
        cv2.imwrite(image_filename, img)
        print(f"Saved: {image_filename}")
