import torch.nn as nn
import functools
import torch
import numpy as np

from .shared import BackboneRegistry

@BackboneRegistry.register("snrnet")
class SNRNet(nn.Module):

    @staticmethod
    def add_argparse_args(parser):
        return parser

    def __init__(self
    ):
        super().__init__()

        self.convt_out = 32

        self.conv5x5_1 = nn.Conv2d(2, 32, 5, padding=2)
        self.maxpool2x2_1 = nn.MaxPool2d(2)
        self.conv3x3_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.maxpool2x1_1 = nn.MaxPool2d((2, 1))

        self.convt_1 = nn.Conv2d(32, self.convt_out, (64, 1), padding=0)
        self.convt_2 = nn.Conv2d(32, self.convt_out, (64, 2), padding=0)
        self.convt_3 = nn.Conv2d(32, self.convt_out, (64, 4), padding=0)
        self.convt_4 = nn.Conv2d(32, self.convt_out, (64, 8), padding=0)

        self.maxpoolt_1 = nn.MaxPool2d((1, 8))
        self.maxpoolt_2 = nn.MaxPool2d((1, 7))
        self.maxpoolt_3 = nn.MaxPool2d((1, 5))
        self.maxpoolt_4 = nn.MaxPool2d((1, 1))

        self.blstm = nn.LSTM(self.convt_out*4, 128, 1, batch_first=True, bidirectional=True)

        # self.fc_128 = nn.Linear(1024, 128)
        # self.fc_32 = nn.Linear(128, 32)
        # self.fc_1 = nn.Linear(32, 1)

        self.fc = nn.Linear(1024, 1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        
        # x = [B, 2, 256, T] : [B, C, F, T]
        time_clusters = x.shape[3]//16

        x_btcf = x.permute(0, 3, 1, 2) # [B, T, 2, 256]
        x_split = x_btcf.reshape(-1, 16, 2, 256) # [B*(T/16), 16, 2, 256] , batch dim 상위 (T/16)개가 1st batch, 그다음 (T/16)개가 2nd batch, ...
        x_split = x_split.permute(0, 2, 3, 1) # [B*(T/16), 2, 256, 16] B*(T/16) batch, 1 channel, 256 frequency bins, 16 time frames

        

        features = self.conv5x5_1(x_split) # [B*(T/16), 32, 256, 16]
        # features = self.act(features)
        features = self.maxpool2x2_1(features) # [B*(T/16), 32, 128, 8]
        features = self.conv3x3_1(features) # [B*(T/16), 32, 128, 8]
        # features = self.act(features)
        features = self.maxpool2x1_1(features) # [B*(T/16), 32, 64, 8]

        features_1 = self.convt_1(features) # [B*(T/16), 384, 1, 8]
        features_2 = self.convt_2(features) # [B*(T/16), 384, 1, 7]
        features_3 = self.convt_3(features) # [B*(T/16), 384, 1, 5]
        features_4 = self.convt_4(features) # [B*(T/16), 384, 1, 1]
 
        features_1 = self.maxpoolt_1(features_1) # [B*(T/16), 384, 1, 1]
        features_2 = self.maxpoolt_2(features_2) # [B*(T/16), 384, 1, 1]
        features_3 = self.maxpoolt_3(features_3) # [B*(T/16), 384, 1, 1]
        features_4 = self.maxpoolt_4(features_4) # [B*(T/16), 384, 1, 1]

        features = torch.cat((features_1, features_2, features_3, features_4), dim=1) # [B*(T/16), 1536, 1, 1]
        features = features.squeeze(3).squeeze(2) # [B*(T/16), 1536]

        features = features.reshape(-1, time_clusters, self.convt_out*4) # [B, T/16, 1536]
        features, _ = self.blstm(features) # [B, T/16, 256=128*2]

        features_avg = torch.mean(features, dim=1) # [B, 256]
        features_std = torch.std(features, dim=1) # [B, 256]
        features_min, _ = torch.min(features, dim=1) # [B, 256]
        features_max, _ = torch.max(features, dim=1) # [B, 256]

        features = torch.cat((features_avg, features_std, features_min, features_max), dim=1) # [B, 1024]
    
        # features = self.fc_128(features) # [B, 128]
        # features = self.fc_32(features) # [B, 32]
        # features = self.fc_1(features) # [B, 1]

        features = self.fc(features)
        
        out = self.sigmoid(features) # [B, 1], groudtruth = normalized noise size i.e. SNR -inf->1, SNR 0->0.7, SNR inf->0


        return out

if __name__ == '__main__':
    model = SNRNet()
    print('parameters: {0}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    x = torch.zeros([4, 2, 256, 384])
    y = model(x)
    print(y.shape)