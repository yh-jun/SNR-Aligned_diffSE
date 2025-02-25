import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.other import pad_spec
import numpy as np
import matplotlib.pyplot as plt
import copy

from sgmse.backbones.snrnet import SNRNet


class SNRModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        return parser

    def __init__(
        self, backbone, lr=1e-4, ema_decay=0.999,
        num_eval_files=10, loss_type='mse', data_module_cls=None, **kwargs
    ):
        super().__init__()
        dnn_cls = SNRNet
        self.dnn = dnn_cls()

        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, x, x_hat):
        loss = torch.nn.functional.mse_loss(x, x_hat.squeeze(1))
        return loss

    def calculate_normfac_direct(self, s, n, fixed_snr):
        normfac = (2.040166) * (0.240253 + 0.759747 * fixed_snr**2)**0.5 / ((1 + (n/s)**2)**0.5)
        return normfac

    def _step(self, batch, batch_idx, valid=False):

        if valid == False:
            x, y = batch
            gt = torch.rand(x.shape[0], device=x.device) * 0.999
            SNR = gt / (1-gt)
            SNR = SNR.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            y = x + (y-x) * 0.56234 * SNR
            normfac_ = self.calculate_normfac_direct(1, SNR, 1)
            y = y * normfac_
        else:
            x, y, s, n = batch
            gt = n / (s+n)
            real_SNR = 20 * torch.log10((1-gt)/gt)
            self.real_SNRs.append(real_SNR)
        
        y = torch.view_as_real(y)
        y = y.permute(0, 4, 2, 3, 1)
        y = y.squeeze(4)

        est_gt = self(y)
        
        if valid == True:
            est_real_SNR = 20 * torch.log10((1-est_gt)/(est_gt))
            self.est_real_SNRs.append(est_real_SNR)

        loss = self._loss(gt, est_gt)
        return loss


    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.real_SNRs = []
        self.est_real_SNRs = []

        loss = self._step(batch, batch_idx, valid=True)

        self.real_SNRs = torch.FloatTensor(self.real_SNRs)
        self.est_real_SNRs = torch.FloatTensor(self.est_real_SNRs)

        self.snr_error = torch.mean(torch.abs(self.real_SNRs - self.est_real_SNRs)).item()

        self.log('snr_error', self.snr_error, on_step=False, on_epoch=True)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pass
        return loss

    def forward(self, y):
        dnn_output = self.dnn(y)

        return dnn_output

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()
    
    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
