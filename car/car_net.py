import numpy as np
import torch
import torch.nn as nn


class CarNet(nn.Module):
    def __init__(self, input_shape=(1, 81, 81)) -> None:

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=128,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.train()

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        return self.policy(x), self.value(x)


class CarNet2(nn.Module):
    def __init__(self, input_shape=(1, 81, 81)) -> None:

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=96,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=128,
                      kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 9)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.train()

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
            print("out_size", np.prod(o.size()))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        return self.policy(x), self.value(x)


class CarNet3(nn.Module):
    def __init__(self, input_shape=(1, 81, 81)) -> None:

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.p2 = nn.Flatten()

        conv_out_size = self._get_conv_out(input_shape)

        self.p1 = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU()
        )

        self.p3 = nn.Linear(256+9+256, 9)

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.train()

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv1(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x[:, :, 39:42, 39:42]
        x3 = x[:, :, 35:46, 35:46]

        y1 = self.p1(x1)
        y2 = self.p2(x2)
        y3 = self.conv2(x3)

        policy = self.p3(torch.cat((y1, y2, y3), axis=1))

        return policy, self.value(x1)


class CarNet5(nn.Module):
    def __init__(self, input_shape=(1, 81, 81)) -> None:

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=6, stride=2, padding=2),
            nn.ReLU(),
        )

        self.fl = nn.Flatten()

        conv_out_size = self._get_conv_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 9),
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 1),
        )

        self.train()

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = torch.cat((self.conv1(torch.zeros(1, *shape)),
                          self.conv2(torch.zeros(1, *shape))), axis=1)
            print("out_size", np.prod(o.size()), o.size())
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.fl(torch.cat((self.conv1(x), self.conv2(x)), axis=1))
        return self.policy(x), self.value(x)
