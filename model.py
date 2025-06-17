import torch
import torch.nn as nn
from torch.autograd import Variable
from Model.image_encode import MobileNetV2
from Model.semantic_encoder import STGCN


class FeatureFusionModule(nn.Module):
    def __init__(self, cnn_dim=1280, stgcn_dim=256, fused_dim=256):
        super(FeatureFusionModule, self).__init__()
        self.cnn_proj = nn.Linear(cnn_dim, fused_dim)
        self.stgcn_proj = nn.Linear(stgcn_dim, fused_dim)

        self.attention = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 4),
            nn.ReLU(),
            nn.Linear(fused_dim // 4, fused_dim),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, stgcn_feat):
        cnn_feat = self.cnn_proj(cnn_feat)  # (batch_size, timesteps, fused_dim)
        stgcn_feat = self.stgcn_proj(stgcn_feat)

        fused_feat = cnn_feat + stgcn_feat

        attention_weights = self.attention(fused_feat)
        fused_feat = fused_feat * attention_weights

        return fused_feat


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, num_classes, fused_dim,
                 t_kernel_size=9,
                 bidirectional=True,
                 dropout=True,
                 device='cuda:0'):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.num_classes = num_classes
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.fused_dim = fused_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        net = MobileNetV2(width_mult=width_mult)
        if pretrain:
            state_dict_mobilenet = torch.load('./weights/mobilenet_v2.pth.tar')
            net.load_state_dict(state_dict_mobilenet, False)

        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.stgcn = STGCN(in_channels=3, graph_args={'layout': 'coco', 'strategy': 'spatial'},
                           edge_importance_weighting=True, t_kernel_size=t_kernel_size)
        self.feature_fusion = FeatureFusionModule(cnn_dim=1280, stgcn_dim=256, fused_dim=self.fused_dim)
        self.rnn = nn.LSTM(int(self.fused_dim * width_mult if width_mult > 1.0 else self.fused_dim),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, num_classes)
        else:
            self.lin = nn.Linear(self.lstm_hidden, num_classes)
        if self.training and self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return (
                Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to(device),
                         requires_grad=True),
                Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to(device),
                         requires_grad=True))
        else:
            return (
                Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(device), requires_grad=True),
                Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(device), requires_grad=True))

    def forward(self, x, y):
        batch_size, timesteps, C, H, W = x.size()
        batch_size, timesteps, num_keypoints, channel = y.size()
        self.hidden = self.init_hidden(batch_size, self.device)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        c_out = c_out.view(batch_size, timesteps, -1)

        # ST-GCN forward
        g_in = y.permute(0, 3, 1, 2).reshape(batch_size, channel, timesteps, num_keypoints, 1)

        g_out = self.stgcn(g_in)
        if self.dropout:
            c_out = self.drop(c_out)
        # LSTM forward
        fused_feature = self.feature_fusion(c_out, g_out)

        r_in = fused_feature  # 输出 (N, T, 256)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, self.num_classes)

        return out

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    x = torch.randn(1, 64, 3, 160, 160).to("cuda:1")
    y = torch.randn(1, 64, 17, 3).to("cuda:1")
    model = EventDetector(pretrain=False,
                          width_mult=1.,
                          lstm_layers=2,
                          lstm_hidden=256,
                          num_classes=9,
                          fused_dim=256,
                          device="cuda:1"
                          )
    model.to("cuda:1")
    out = model(x, y)
    print(out.shape)
