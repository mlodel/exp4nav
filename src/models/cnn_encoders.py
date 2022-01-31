import torch
import torch.nn as nn
from torchvision import models


class CNN_Encoder_base(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = None

    def forward(self, img):
        seq_len, batch_size, C, H, W = img.size()
        img = img.reshape(batch_size * seq_len, C, H, W)
        cnn_out = self.network(img)
        cnn_out = cnn_out.reshape(seq_len, batch_size, -1)
        return cnn_out


class CNN3Layer(CNN_Encoder_base):

    def __init__(self, img_size, img_ch, out_ftrs):
        super().__init__()

        out_size = (img_size // 8) ** 2 * 64

        self.network = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(img_ch, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(out_size, out_ftrs)
        )


class CNN3Layer_old(CNN_Encoder_base):

    def __init__(self, img_size, img_ch, out_ftrs):
        super().__init__()

        out_size = (img_size // 8) ** 2 * 16

        self.network = nn.Sequential(
            nn.Conv2d(img_ch, 64, 5, stride=2, padding=1),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(out_size, out_ftrs)
        )


class ResNetEnc(CNN_Encoder_base):
    def __init__(self, img_size, img_ch, out_ftrs):
        super().__init__()

        self.network = models.resnet18(pretrained=True)
        num_ftrs = self.network.fc.in_features

        self.network.avgpool = nn.AvgPool2d(3, stride=1)
        self.network.fc = nn.Linear(num_ftrs, out_ftrs)
