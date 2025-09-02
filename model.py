import torch
import torch.nn as nn
from thop import profile
from module.image_encode import MobileNetV2
from module.semantic_encoder import STGCN


class PeopleAttentionFuse(nn.Module):
    """
    输入: x (N, M, T, C)
    掩码: mask (N, M, 1, 1)  1=有效人, 0=无效/占位
    输出: out (N, T, C), attn (N, M, T)
    """
    def __init__(self, c_out, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(c_out),
            nn.Linear(c_out, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)  # 每个(N,M,T)位置产出一个logit
        )

    def forward(self, x, mask=None):
        # x: (N, M, T, C)
        logits = self.score(x)                  # (N, M, T, 1)
        if mask is not None:
            # 无效人位置打 -inf，避免分到权重
            logits = logits.masked_fill(~mask.bool(), float('-inf'))
        attn = torch.softmax(logits, dim=1)     # 在人维M做softmax -> (N, M, T, 1)
        out = (attn * x).sum(dim=1)             # (N, T, C)
        return out, attn.squeeze(-1)            # (N, T, C), (N, M, T)
    

class FeatureFusionModule(nn.Module):
    def __init__(self, cnn_dim, stgcn_dim, fused_dim, fusion_type='add'):
        super(FeatureFusionModule, self).__init__()
        assert fusion_type in ['add', 'concat', 'gate', 'mul', 'mlp'], \
            "fusion_type must be one of ['add', 'concat', 'gate', 'mul', 'mlp']"
        self.fusion_type = fusion_type

        self.cnn_proj = nn.Linear(cnn_dim, fused_dim)
        self.stgcn_proj = nn.Linear(stgcn_dim, fused_dim)

        if fusion_type == 'concat' or fusion_type == 'mlp':
            self.concat_proj = nn.Linear(fused_dim * 2, fused_dim)

        if fusion_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(fused_dim * 2, fused_dim * 2),
                nn.ReLU(),
                nn.Linear(fused_dim * 2, fused_dim)
            )

        if fusion_type == 'gate':
            self.gate_fc = nn.Sequential(
                nn.Linear(fused_dim * 2, fused_dim),
                nn.Sigmoid()
            )

        self.attention = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 4),
            nn.ReLU(),
            nn.Linear(fused_dim // 4, fused_dim),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, stgcn_feat):
        cnn_feat = self.cnn_proj(cnn_feat)        # (B, T, D)
        stgcn_feat = self.stgcn_proj(stgcn_feat)  # (B, T, D)

        if self.fusion_type == 'add':
            fused_feat = cnn_feat + stgcn_feat

        elif self.fusion_type == 'concat':
            fused_feat = torch.cat([cnn_feat, stgcn_feat], dim=-1)
            fused_feat = self.concat_proj(fused_feat)

        elif self.fusion_type == 'gate':
            gate_input = torch.cat([cnn_feat, stgcn_feat], dim=-1)
            gate = self.gate_fc(gate_input)
            fused_feat = gate * cnn_feat + (1 - gate) * stgcn_feat

        elif self.fusion_type == 'mul':
            fused_feat = cnn_feat * stgcn_feat

        elif self.fusion_type == 'mlp':
            fused_feat = torch.cat([cnn_feat, stgcn_feat], dim=-1)
            fused_feat = self.mlp(fused_feat)

        # channel-attention
        attention_weights = self.attention(fused_feat) # (B, T, D)
        fused_feat = fused_feat * attention_weights

        return fused_feat


class LockedDropout(nn.Module):
    """时间维共享掩码的 dropout：对 [B, T, D] 更友好。"""
    def __init__(self, p: float):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # x: [B, T, D]
        B, T, D = x.shape
        mask = x.new_empty(B, 1, D).bernoulli_(1 - self.p).div_(1 - self.p)
        return x * mask


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, num_classes, fused_dim, fusion_type,
                 t_kernel_size=9,
                 bidirectional=True,
                 dropout=True,
                 multi_person=False):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.num_classes = num_classes
        self.fused_dim = fused_dim
        self.fusion_type = fusion_type
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.use_dropout = dropout
        self.multi_person = multi_person

        net = MobileNetV2(width_mult=width_mult)
        if pretrain:
            state_dict_mobilenet = torch.load('./weights/mobilenet_v2.pth.tar', map_location='cpu')
            net.load_state_dict(state_dict_mobilenet, strict=False)

        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.stgcn = STGCN(in_channels=3, graph_args={'layout': 'coco', 'strategy': 'spatial'},
                           edge_importance_weighting=True, t_kernel_size=t_kernel_size)
        
        self.feature_fusion = FeatureFusionModule(cnn_dim=1280, stgcn_dim=256, fused_dim=self.fused_dim, fusion_type=self.fusion_type)

        lstm_in_dim = self.fused_dim
        lstm_dropout = 0.3 if (self.use_dropout and lstm_layers > 1) else 0.0
        self.rnn = nn.LSTM(lstm_in_dim, self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional, dropout=lstm_dropout)
        
        feat_dim = 2 * self.lstm_hidden if bidirectional else self.lstm_hidden
        self.lin = nn.Linear(feat_dim, num_classes)

        self.lockdrop_in = LockedDropout(0.2 if self.use_dropout else 0.0) 
        self.drop_head = nn.Dropout(0.5 if self.use_dropout else 0.0)

        self.people_fuser = PeopleAttentionFuse(c_out=256, hidden=128) if multi_person else None


    def forward(self, x, y):
        # CNN forward
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_feat = self.cnn(c_in)                 
        c_feat = self.gap(c_feat).flatten(1)    
        c_out = c_feat.reshape(batch_size, timesteps, -1) 

        # ST-GCN forward
        if self.multi_person:
            batch_size, timesteps, num_keypoints, channel, people = y.size()
            g_in = y.permute(0, 3, 1, 2, 4).reshape(batch_size, channel, timesteps, num_keypoints, people)
            g_out = self.stgcn(g_in)
            G_out = g_out.size(-1)
            x_mp  = g_out.view(batch_size, people, timesteps, G_out)  # (N, M, T, C')

            with torch.no_grad():
                person_mask_bool = (y.abs().sum(dim=(1,2,3)) > 0)      # (N, M)
            mask = person_mask_bool.unsqueeze(-1).unsqueeze(-1)        # (N, M, 1, 1)

            g_out, attn = self.people_fuser(x_mp, mask=mask)

        else:
            batch_size, timesteps, num_keypoints, channel = y.size()
            g_in = y.permute(0, 3, 1, 2).reshape(batch_size, channel, timesteps, num_keypoints, 1)
            g_out = self.stgcn(g_in)

        # fused module
        fused_feature = self.feature_fusion(c_out, g_out)

        # LSTM forward
        fused_feature = self.lockdrop_in(fused_feature)
        r_in = fused_feature 
        r_out, _ = self.rnn(r_in)

        # Classifer
        r_out = self.drop_head(r_out)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, self.num_classes)

        return out

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_fusion():
    B, T = 1, 64 
    cnn_dim, stgcn_dim, fused_dim = 1280, 256, 256

    dummy_cnn = torch.randn(B, T, cnn_dim)
    dummy_stgcn = torch.randn(B, T, stgcn_dim)

    fusion_types = ['add', 'concat', 'gate', 'mul', 'mlp']
    print("Fusion Type |   Params   |   FLOPs")
    print("----------------------------------------")

    for fusion_type in fusion_types:
        model = FeatureFusionModule(cnn_dim, stgcn_dim, fused_dim, fusion_type=fusion_type)
        model.eval()
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_cnn, dummy_stgcn), verbose=False)
            print(f"{fusion_type:>10} | {params/1e3:9.1f}K | {flops/1e6:7.2f} MFLOPs")

if __name__ == '__main__':
    count_fusion()
    x = torch.randn(1, 64, 3, 160, 160).to("cuda:1")
    y = torch.randn(1, 64, 17, 3, 2).to("cuda:1")
    model = EventDetector(pretrain=False,
                          width_mult=1.,
                          lstm_layers=2,
                          lstm_hidden=256,
                          num_classes=9,
                          fused_dim=256,
                          fusion_type="gate",
                          multi_person=True
                          )
    model.to("cuda:1")
    out = model(x, y)
    sum = count_parameters(model)
    print(sum)
    print(out.shape)
