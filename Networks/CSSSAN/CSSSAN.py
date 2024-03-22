import torch
from torch import nn

def get_drop_mask1(X, drop_thr):
    # print(X)
    max_val, _ = torch.max(X, axis=2, keepdims=True)
    thr_val = max_val * drop_thr
    a = (X >= thr_val)
    return a


class RspaA1(nn.Module):
    def __init__(self):
        super(RspaA1, self).__init__()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X):
        N, in_channel, h, w = X.shape
        x1 = X.reshape(N, in_channel, h * w)
        x2 = x1.permute(0, 2, 1)
        c = self.softmax(x2 @ x1)
        out = (c @ x2).permute(0, 2, 1).reshape(N, in_channel, h, w)
        return out
class CSSpeA(nn.Module):
    def __init__(self, inchannels=64):
        super(CSSpeA, self).__init__()
        self.inchannels = inchannels
        self.soft = nn.Softmax(dim=2)
        self.fc = nn.Linear(self.inchannels, self.inchannels)
        self.act = nn.Sigmoid()

    def forward(self, X):
        N, in_channel, h, w = X.shape
        x_center = X[:, :, int(h / 2), int(h / 2)].reshape(N, in_channel, 1).permute(0, 2, 1)
        # print(X[:, :, int(h / 2), int(h / 2)]@X[:, :, int(h / 2), int(h / 2)].permute(1,0))
        X_temp = X.reshape(N, in_channel, h * w).permute(0, 2, 1)
        R = x_center @ X_temp.permute(0, 2, 1)
        a = get_drop_mask1(R, drop_thr=0.4)
        a[:, :, int(h * w / 2)] = True
        R = R * a + (~a) * (-100.)
        R = self.soft(R)
        R[:, :, int(h * w / 2)] = 1.
        out = R @ X_temp
        W = self.act(self.fc(out))
        W = W.permute(0, 2, 1).reshape(N, in_channel, 1, 1)
        result = X * W
        return result
class RIAN(nn.Module):
    def __init__(self, inchannel=64, kernel_size=9, num_classes=16):
        super(RIAN, self).__init__()
        self.kernel_size = kernel_size

        self.CSpeA = CSSpeA(inchannels=204)
        self.conv1 = nn.Conv2d(in_channels=204, out_channels=inchannel, kernel_size=1)
        self.bN = nn.BatchNorm2d(inchannel)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),

            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),

            RspaA1(),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),

            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),

            RspaA1(),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),

        )
        self.relu = nn.ReLU()
        self.gap = nn.AvgPool2d(kernel_size=kernel_size)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(inchannel, num_classes)
        self.bN1 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        # 光谱特征抑制
        x1 = self.CSpeA(x)
        x1 = self.relu(self.bN(self.conv1(x1)))
        # print('before',x1)
        # print('after', x1)
        temp = x1
        x1 = self.block1(x1)
        temp = self.relu(x1 + temp)
        # # # block2
        # # # temp = temp + self.SpaA(temp)
        x1 = self.block2(temp)
        x1 = self.relu(x1 + temp)
        x1 = self.flatten(self.gap(x1))
        out = self.relu(self.bN1(self.head(x1)))
        return out
