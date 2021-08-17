import torch
import torch.nn as nn
import torch.nn.functional as F

'''My ConvNet in PyTorch.'''
class my_ConvNet(nn.Module):
    def __init__(self):
        super(my_ConvNet, self).__init__()
        self.ConvPart = self._make_layers()

    def forward(self, x):
        out = self.ConvPart(x)
        out = out.squeeze()
        return out

    def _make_layers(self):

        layers = [nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), padding=1),
                  nn.BatchNorm2d(16),
                  nn.ReLU(inplace=True),

                  nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=1),
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),

                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True),

                  nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),

                  nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1),
                  nn.BatchNorm2d(32),
                  nn.ReLU(inplace=True),

                  nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0),


                  nn.AdaptiveAvgPool2d(output_size=1)
                  ]

        return nn.Sequential(*layers)

def test():
    net = my_ConvNet()
    print(net)
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
