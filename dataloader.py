import os.path as osp
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, npy_dir, seq_length, input_size=160,
                 transform=None, train=True, multi_format=False, mode='dtl'):
        """
        mode: 'dtl'（默认）→ 8类（含1个背景类）,关键10帧,第1和最后一个动作默认不检测
              'fo' → 12类（含1个背景类）,关键13帧,第1和最后一个动作默认不检测
        """
        self.df = read_txt(data_file)
        self.vid_dir = vid_dir
        self.npy_dir = npy_dir
        self.input_size = input_size
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.multi_format = multi_format

        if mode == 'fo':
            self.default_label = 11  # FO共0~11：11类 + 背景
        elif mode == 'dtl':
            self.default_label = 8   # DTL共0~8：8类 + 背景
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'dtl' or 'fo'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df[idx]
        events = np.array(a['events'])

        keypoints = np.load(osp.join(self.npy_dir, '{}.npy'.format(a['id'])))
        video_path = self._get_video_path(a['id'])
        cap = cv2.VideoCapture(video_path)

        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        keypoints[:, :, 0] /= frame_size[1]
        keypoints[:, :, 1] /= frame_size[0]

        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images, keys, labels = [], [], []

        if self.train:
            max_start = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1 - self.seq_length)
            start_frame = np.random.randint(0, max_start)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            last_img = None
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    processed = self._process_img(img, new_size, top, bottom, left, right)
                    images.append(processed)
                    keys.append(keypoints[pos])
                    labels.append(self._get_label(events, pos))
                    last_img = processed
                    pos += 1
                else:
                    if last_img is not None:
                        images.append(last_img)
                        keys.append(keypoints[pos - 1])
                        labels.append(self.default_label)
            cap.release()
        else:
            pos = 0
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                processed = self._process_img(img, new_size, top, bottom, left, right)
                images.append(processed)
                keys.append(keypoints[pos])
                labels.append(self._get_label(events, pos))
                pos += 1
            cap.release()

        sample = {'images': np.asarray(images), 'keys': np.asarray(keys), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_video_path(self, vid_id):
        if self.multi_format:
            extensions = ['mp4', 'mov', 'MP4', 'MOV']
            for ext in extensions:
                path = osp.join(self.vid_dir, f'{vid_id}.{ext}')
                if osp.exists(path):
                    return path
            raise FileNotFoundError(f"No video found for ID {vid_id} with supported extensions.")
        else:
            return osp.join(self.vid_dir, f'{vid_id}.mp4')

    def _process_img(self, img, new_size, top, bottom, left, right):
        resized = cv2.resize(img, (new_size[1], new_size[0]))
        b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
        return cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)

    def _get_label(self, events, pos):
        return np.where(events[1:-1] == pos)[0][0] if pos in events[1:-1] else self.default_label


class ToTensor(object):
    def __call__(self, sample):
        images = sample['images'].transpose((0, 3, 1, 2))
        return {
            'images': torch.from_numpy(images).float().div(255.),
            'keys': torch.from_numpy(sample['keys']).float(),
            'labels': torch.from_numpy(sample['labels']).long()
        }


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, keys, labels = sample['images'], sample['keys'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'keys': keys, 'labels': labels}


def read_txt(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 1:
                data.append({'id': parts[0], 'events': list(map(int, parts[1:]))})
    return data


if __name__ == '__main__':
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    dataset_dtl = GolfDB(
        data_file='/data/ssd1/xietingyu/frameflow_dtl/label.txt',
        vid_dir='/data/ssd1/xietingyu/frameflow_dtl/videos',
        npy_dir='/data/ssd1/xietingyu/frameflow_dtl/npy',
        seq_length=64,
        transform=transforms.Compose([ToTensor(), norm]),
        train=False,
        multi_format=True,
        mode='dtl'
    )
    dataset_fo = GolfDB(
        data_file='/data/ssd1/xietingyu/frameflow_fo/label.txt',
        vid_dir='/data/ssd1/xietingyu/frameflow_fo/video',
        npy_dir='/data/ssd1/xietingyu/frameflow_fo/npy',
        seq_length=64,
        transform=transforms.Compose([ToTensor(), norm]),
        train=False,
        multi_format=True,
        mode='fo'
    )
