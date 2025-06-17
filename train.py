import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import EventDetector
from util import AverageMeter, freeze_layers
from dataloader import GolfDB, Normalize, ToTensor


def count_frozen_parameters(model):
    total = 0
    frozen = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if not param.requires_grad:
            frozen += param.numel()
    print(f"冻结参数比例: {frozen}/{total} ({100. * frozen / total:.2f}%)")


def train_model(mode='dtl'):
    assert mode in ['dtl', 'fo'], "mode must be either 'dtl' or 'fo'"

    config = {
        'dtl': {
            'iterations': 10500,
            'it_save': 1050,
            'data_file': '/data/ssd1/xietingyu/frameflow_dtl/label.txt',
            'vid_dir': '/data/ssd1/xietingyu/frameflow_dtl/videos/',
            'npy_dir': '/data/ssd1/xietingyu/frameflow_dtl/npy/',
            'num_classes': 9,
            'save_path': 'weights/dtl_models/FS_DTL_{}.pth.tar',
            'weights': [1 / 8] * 8 + [1 / 35],
            'multi_format': True
        },
        'fo': {
            'iterations': 10800,
            'it_save': 1080,
            'data_file': '/data/ssd1/xietingyu/frameflow_fo/label.txt',
            'vid_dir': '/data/ssd1/xietingyu/frameflow_fo/videos/',
            'npy_dir': '/data/ssd1/xietingyu/frameflow_fo/npy',
            'num_classes': 12,
            'save_path': 'weights/fo_models/FS_FO_{}.pth.tar',
            'weights': [1 / 11] * 11 + [1 / 30],
            'multi_format': True
        }
    }

    cfg = config[mode]
    iterations = cfg['iterations']
    it_save = cfg['it_save']
    n_cpu = 4
    seq_length = 64
    bs = 6
    k = 10
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = EventDetector(pretrain=True, width_mult=1., lstm_layers=2,
                          lstm_hidden=256, num_classes=cfg['num_classes'],
                          fused_dim=256, device=device)
    freeze_layers(k, model)
    count_frozen_parameters(model)
    model.train()
    model.to(device)

    # 数据加载
    dataset = GolfDB(
        data_file=cfg['data_file'],
        vid_dir=cfg['vid_dir'],
        npy_dir=cfg['npy_dir'],
        seq_length=seq_length,
        input_size=160,
        transform=transforms.Compose([ToTensor(),
                                      Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])]),
        train=True,
        multi_format=cfg['multi_format'],
        mode=mode
    )
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=n_cpu, drop_last=True)

    weights = torch.FloatTensor(cfg['weights']).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    train_losses = AverageMeter()

    # 开始训练
    i = 0
    epoch = 0
    while i <= iterations:
        for sample in data_loader:
            images, keys, labels = sample['images'].to(device), sample['keys'].to(device), sample['labels'].to(device)
            logits = model(images, keys)
            labels = labels.view(bs * seq_length)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            train_losses.update(loss.item(), images.size(0))
            optimizer.step()

            print(f'Epoch: {epoch}, Iteration: {i}, Training Loss: {train_losses.val:.4f} ({train_losses.avg:.4f})')

            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, cfg['save_path'].format(i))
            scheduler.step()
            i += 1
        epoch += 1


if __name__ == '__main__':
    train_model(mode='dtl')  # or mode='fo'
    print("finish")
