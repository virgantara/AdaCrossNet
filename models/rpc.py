import torch
import torch.nn as nn
import torch.nn.functional as F
# from util_lib.GDANet_util import GDM, local_operator, SGCAM
from models.util_lib.GDANet_util import GDM, local_operator, SGCAM


class RPC(nn.Module):
    def __init__(self, args, output_channels=40):
        super(RPC, self).__init__()
        self.args = args

        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(256, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=True),
                                    self.bn11)
        self.SGCAM_1s = SGCAM(128)
        self.SGCAM_1g = SGCAM(128)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.bn_feat = nn.BatchNorm1d(1024)
        self.inv_head = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256)
        )

        # self.linear1 = nn.Linear(1024, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size, _, _ = x.size()

        x1 = local_operator(x, k=30)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv11(x1))
        x1 = x1.max(dim=-1, keepdim=False)[0]

        # Geometry-Disentangle Module:
        x1s, x1g = GDM(x1, M=256)

        # Sharp-Gentle Complementary Attention Module:
        y1s = self.SGCAM_1s(x1, x1s.transpose(2, 1))
        y1g = self.SGCAM_1g(x1, x1g.transpose(2, 1))
        feature_1 = torch.cat([y1s, y1g], 1)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)

        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (B, 1024)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (B, 1024)

        feat = torch.cat((x_max, x_avg), dim=1)  # (B, 2048)

        inv_feat = self.inv_head(feat)  # (B, 256) → for contrastive head

        return inv_feat, feat 

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

