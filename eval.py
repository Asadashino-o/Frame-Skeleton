from torch.utils.data import DataLoader
from dataloader import GolfDB, Normalize, ToTensor
from model import EventDetector
import os
import torch
import torch.nn.functional as F
import numpy as np
from util import correct_preds
from torchvision import transforms


def eval(model, seq_length, data_loader, disp, device='cuda:0', eval_event_num=8):
    model.eval()
    correct = []
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            images, keys, labels = sample['images'].to(device), sample['keys'].to(device), sample['labels'].to(device)
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
            _, _, _, _, c = correct_preds(probs, labels.squeeze().cpu())
            if disp:
                print(i, c)
            correct.append(c)
    PCE = np.mean(correct)
    single_PCE = np.mean(correct, 0)
    print(f'PCE for {eval_event_num} events: {single_PCE}')
    print('Average PCE: {}'.format(PCE))
    return PCE, single_PCE

def validation(mode, fusion_type, seq_length, input_size, device='cuda:0', n_cpu=1, i=10):
    weight_path = f"weights/{mode}_models/{fusion_type}/image{input_size}/seq{seq_length}"
    config = {
        'dtl': {
            'data_file': '/data/ssd1/xietingyu/frameflow_dtl/val.txt',
            'vid_dir': '/data/ssd1/xietingyu/frameflow_dtl/videos/',
            'npy_dir': '/data/ssd1/xietingyu/frameflow_dtl/npy/',
            'num_classes': 9,
            'eval_events': 8
        },
        'fo': {
            'data_file': '/data/ssd1/xietingyu/frameflow_fo/val.txt',
            'vid_dir': '/data/ssd1/xietingyu/frameflow_fo/videos/',
            'npy_dir': '/data/ssd1/xietingyu/frameflow_fo/npy/',
            'num_classes': 12,
            'eval_events': 11
        },
        'golfdb': {
            'data_file': '/data/ssd1/xietingyu/Frame-Skeleton/data/golfdb/val.txt',
            'vid_dir': '/data/ssd1/xietingyu/Frame-Skeleton/data/golfdb/all_mp4_file/',
            'npy_dir': '/data/ssd1/xietingyu/Frame-Skeleton/data/golfdb/key_point/',
            'num_classes': 9,
            'eval_events': 8
        }
    }

    cfg = config[mode]

    model = EventDetector(pretrain=False,
                          width_mult=1.,
                          lstm_layers=2,
                          lstm_hidden=256,
                          num_classes=cfg['num_classes'],
                          fused_dim=256,
                          fusion_type=fusion_type,
                          dropout=False)

    val_dataset = GolfDB(data_file=cfg['data_file'],
                         vid_dir=cfg['vid_dir'],
                         npy_dir=cfg['npy_dir'],
                         seq_length=seq_length,
                         input_size=input_size,
                         transform=transforms.Compose([ToTensor(),
                                                       Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                         train=False,
                         multi_format=True,
                         mode=mode)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=n_cpu, drop_last=False)
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        # print(f"{name}: {num_params} 个参数")

    print(f"模型总参数量: {total_params}")
    while i >= 0:
        torch.cuda.empty_cache()
        save_dict = torch.load(os.path.join(weight_path,f"FS_{mode}_{i}.pth.tar"))
        print(f"评估 FS_{mode}_{i}.pth模型....")
        model.load_state_dict(save_dict['model_state_dict'])
        model.to(device)
        eval(model, seq_length, val_loader, disp=False, device=device, eval_event_num=cfg['eval_events'])
        i -= 1

if __name__ == '__main__':
    mode = 'dtl' # mode=golfdb/dtl/fo
    fusion_type = 'concat' # fusion_type=add/concat/gate/mul/mlp
    seq_length = 128 # seq_length=32/64/128/256
    input_size = 224 # 160/224
    device = 'cuda:3'
    validation(mode=mode,fusion_type=fusion_type,seq_length=seq_length,input_size=input_size,device=device)
    

    
