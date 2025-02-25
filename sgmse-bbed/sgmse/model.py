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
from sgmse.util.inference import evaluate_model
from sgmse.util.deep_inference import deep_evaluate_model
from sgmse.util.other import pad_spec
import numpy as np
import matplotlib.pyplot as plt
import copy
from sgmse.snr_estimator import SNRModel

from sgmse.util.other import snr_dB, pad_spec_16

i_30 = np.arange(1, 30+1)
t_30 = (0.001 ** (1 / 7) + (i_30 - 1) / (30 - 1) * (1 ** (1 / 7) - 0.001 ** (1 / 7))) ** 7

snr_model = SNRModel.load_from_checkpoint(
            './sgmse-bbed/sgmse/snr_estimator.ckpt', base_dir="",
            batch_size=1, num_workers=0
        )
snr_model.eval()
snr_model.to('cuda')

class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default= 0.5,  help="magnitude transformation in the loss term")
        return parser

    def __init__(
        self, backbone, sde, model_type='sebridge', snr_conditioned='false', fixed_snr=1.0, lr=1e-4, ema_decay=0.999, t_eps=3e-2, loss_abs_exponent=0.5, 
        num_eval_files=10, loss_type='mse', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        if snr_conditioned == 'false' or snr_conditioned == 'fixed':
            dnn_cls = BackboneRegistry.get_by_name(backbone)
            self.dnn = dnn_cls(**kwargs)
        elif snr_conditioned == 'true':
            # dnn_cls = BackboneRegistry.get_by_name('ncsnpp_snr')
            dnn_cls = BackboneRegistry.get_by_name(backbone)
            self.dnn = dnn_cls(**kwargs)

        
        # Initialize SDE
        if sde == 'bbve':
            #change parameters, if the old class bbve is used. Needed for loading the provided checkpoint
            #as that checkpoint was trained with the old class.
            sde = 'bbed'
            kwargs['k'] = kwargs['sigma_max']
            del kwargs['sigma_max']
            del kwargs['sigma_min']
        
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        self.sigma_max = kwargs['sigma_max']
        self.model_type = model_type
        self.snr_conditioned = snr_conditioned
        self.fixed_snr = fixed_snr
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, fixed_snr=self.fixed_snr, gpu=kwargs.get('gpus', 0) > 0)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
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

    
    def _loss(self, score, sigmas, z):    
        if self.loss_type == 'mse':
            err = sigmas*score + z 
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def noise_mag(self, s, s_hat, mode='mean'):
        with torch.no_grad():
            if mode == 'mean':
                mag = torch.abs(torch.mean(torch.sqrt(torch.square(s-s_hat))))
            elif mode == 'max':
                mag = torch.max(torch.abs(s-s_hat))
            else:
                mag = 0
        return mag


    def _step(self, batch, batch_idx, valid=False):
        
        if valid:
            x, y, _, _ = batch
        else:
            x, y = batch

        # print('SNR:{0:.1f}'.format(snr_dB(x.detach().cpu().numpy(), y.detach().cpu().numpy())))


        if self.snr_conditioned == 'false':

            if self.model_type == "bbed":
                rdm = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
                t = torch.min(rdm, torch.tensor(self.sde.T))
                mean, std = self.sde.marginal_prob(x, t, y)
                z = torch.randn_like(x)
                sigmas = std[:, None, None, None]
                perturbed_data = mean + sigmas * z
                score = self(perturbed_data, t, y)

                if self.loss_type == 'mse' or self.loss_type == 'mae':
                    loss = self._loss(score, sigmas, z)
                elif self.loss_type == 'sqrt_mse':
                    mean_hat = perturbed_data + (sigmas ** 2) * score
                    sqrt_mean_hat = mean_hat.abs()**0.5 * torch.exp(1j * mean_hat.angle())
                    sqrt_mean = mean.abs()**0.5 * torch.exp(1j * mean.angle())
                    err = (sqrt_mean_hat - sqrt_mean) / sigmas
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
                
                return loss
            
            elif self.model_type == "sebridge":
                
                N, roh, eps, T = 30, 7, 0.001, 0.999

                n = torch.randint(1, N, size=[x.shape[0]], device=x.device)
                n = n.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                z = torch.randn_like(x) * self.sigma_max
                
                t_n = (eps**(1/roh) + ((n-1)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh
                t_n1 = (eps**(1/roh) + ((n)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh

                x_t_n = y*t_n + x*(1-t_n) + ((t_n*(1-t_n))**0.5) * z
                x_t_n1 = y*t_n1 + x*(1-t_n1) + ((t_n1*(1-t_n1))**0.5) * z

                f_theta = self(x_t_n1, t_n1, y)
                f_theta_minus = self(x_t_n, t_n, y)

                if self.loss_type == 'mse':
                    err = f_theta - f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

                elif self.loss_type == 'sqrt_mse':
                    sqrt_f_theta = f_theta.abs()**0.5 * torch.exp(1j * f_theta.angle())
                    sqrt_f_theta_minus = f_theta_minus.abs()**0.5 * torch.exp(1j * f_theta_minus.angle())
                    err = sqrt_f_theta - sqrt_f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
                
                return loss
            
            elif self.model_type == 'sebridge_v2':
                N, roh, eps, T = 30, 7, 0.001, 1

                n = torch.randint(1, N, size=[x.shape[0]], device=x.device)
                n = n.unsqueeze(1).unsqueeze(2).unsqueeze(3)

                z_mag = self.sigma_max

                z = torch.randn_like(x) * z_mag
                
                t_n = (eps**(1/roh) + ((n-1)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh
                t_n1 = (eps**(1/roh) + ((n)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh

                x_t_n = y*t_n + x*(1-t_n) + (t_n) * z
                x_t_n1 = y*t_n1 + x*(1-t_n1) + (t_n1) * z

                mu_t_n = y*t_n + x*(1-t_n)
                mu_t_n1 = y*t_n1 + x*(1-t_n1)

                f_theta = self(x_t_n1, t_n1, mu_t_n1)
                f_theta_minus = self(x_t_n, t_n, mu_t_n)

                if self.loss_type == 'mse':
                    err = f_theta - f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
                
                elif self.loss_type == 'sqrt_mse':
                    sqrt_f_theta = f_theta.abs()**0.5 * torch.exp(1j * f_theta.angle())
                    sqrt_f_theta_minus = f_theta_minus.abs()**0.5 * torch.exp(1j * f_theta_minus.angle())
                    err = sqrt_f_theta - sqrt_f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        
        elif self.snr_conditioned == 'fixed':

            if self.model_type == 'sebridge_v2':
                N, roh, eps, T = 30, 7, 0.001, 0.999

                n = torch.randint(1, N, size=[x.shape[0]], device=x.device)
                n = n.unsqueeze(1).unsqueeze(2).unsqueeze(3)

                noise_size = self.noise_mag(x, y, mode='max')
                y = x + (y-x) / noise_size * self.fixed_snr

                z_mag = self.sigma_max

                z = torch.randn_like(x) * z_mag
                
                t_n = (eps**(1/roh) + ((n-1)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh
                t_n1 = (eps**(1/roh) + ((n)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh

                x_t_n = y*t_n + x*(1-t_n) + (t_n) * z
                x_t_n1 = y*t_n1 + x*(1-t_n1) + (t_n1) * z

                f_theta = self(x_t_n1, t_n1, y)
                f_theta_minus = self(x_t_n, t_n, y)

                if self.loss_type == 'mse':
                    err = f_theta - f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
                
                elif self.loss_type == 'sqrt_mse':
                    sqrt_f_theta = f_theta.abs()**0.5 * torch.exp(1j * f_theta.angle())
                    sqrt_f_theta_minus = f_theta_minus.abs()**0.5 * torch.exp(1j * f_theta_minus.angle())
                    err = sqrt_f_theta - sqrt_f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
            
            if self.model_type == 'sebridge_v3':
                N, roh, eps, T = 30, 7, 0.001, 1

                n = torch.randint(1, N, size=[x.shape[0]], device=x.device)
                n = n.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                t_n = (eps**(1/roh) + ((n-1)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh
                t_n1 = (eps**(1/roh) + ((n)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh

                z_mag = self.sigma_max
                z = torch.randn_like(x) * z_mag

                x_ori = self._backward_transform(x)
                y0_ori = self._backward_transform(y) - x_ori
                y0_snr = y0_ori * self.fixed_snr

                mu_t_n = self._forward_transform(x_ori + y0_snr * t_n)
                mu_t_n1 = self._forward_transform(x_ori + y0_snr * t_n1)

                x_t_n = mu_t_n + (t_n) * z
                x_t_n1 = mu_t_n1 + (t_n1) * z

                f_theta = self(x_t_n1, t_n1, mu_t_n1)
                f_theta_minus = self(x_t_n, t_n, mu_t_n)

                if self.loss_type == 'mse':
                    err = f_theta - f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
                elif self.loss_type == 'sqrt_mse':
                    sqrt_f_theta = f_theta.abs()**0.5 * torch.exp(1j * f_theta.angle())
                    sqrt_f_theta_minus = f_theta_minus.abs()**0.5 * torch.exp(1j * f_theta_minus.angle())
                    err = sqrt_f_theta - sqrt_f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.snr_conditioned == 'true':
            if self.model_type == 'sebridge_v2':
                z_mag = self.sigma_max
                
                N, roh, eps, T = 30, 7, 0.001, 1

                n = torch.randint(1, N, size=[x.shape[0]], device=x.device)
                n = n.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                z = torch.randn_like(x) * z_mag
                
                t_n = (eps**(1/roh) + ((n-1)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh
                t_n1 = (eps**(1/roh) + ((n)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh

                x_t_n = y*t_n + x*(1-t_n) + (t_n) * z
                x_t_n1 = y*t_n1 + x*(1-t_n1) + (t_n1) * z

                mu_t_n = y*t_n + x*(1-t_n)
                mu_t_n1 = y*t_n1 + x*(1-t_n1)

                f_theta = self(x_t_n1, t_n1, mu_t_n1)
                f_theta_minus = self(x_t_n, t_n, mu_t_n)

                if self.loss_type == 'mse':
                    err = f_theta - f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
                elif self.loss_type == 'sqrt_mse':
                    sqrt_f_theta = f_theta.abs()**0.5 * torch.exp(1j * f_theta.angle())
                    sqrt_f_theta_minus = f_theta_minus.abs()**0.5 * torch.exp(1j * f_theta_minus.angle())
                    err = sqrt_f_theta - sqrt_f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
            
            if self.model_type == 'sebridge_v3':
                N, roh, eps, T = 30, 7, 0.001, 1

                n = torch.randint(1, N, size=[x.shape[0]], device=x.device)
                n = n.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                t_n = (eps**(1/roh) + ((n-1)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh
                t_n1 = (eps**(1/roh) + ((n)/(N-1)) * (T**(1/roh)-eps**(1/roh))) ** roh

                z_mag = self.sigma_max
                z = torch.randn_like(x) * z_mag

                mu_t_n = self._forward_transform(self._backward_transform(x) * (1-t_n) + self._backward_transform(y) * t_n)
                mu_t_n1 = self._forward_transform(self._backward_transform(x) * (1-t_n1) + self._backward_transform(y) * t_n1)

                x_t_n = mu_t_n + (t_n) * z
                x_t_n1 = mu_t_n1 + (t_n1) * z

                f_theta = self(x_t_n1, t_n1, mu_t_n1)
                f_theta_minus = self(x_t_n, t_n, mu_t_n)

                if self.loss_type == 'mse':
                    err = f_theta - f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
                elif self.loss_type == 'sqrt_mse':
                    sqrt_f_theta = f_theta.abs()**0.5 * torch.exp(1j * f_theta.angle())
                    sqrt_f_theta_minus = f_theta_minus.abs()**0.5 * torch.exp(1j * f_theta_minus.angle())
                    err = sqrt_f_theta - sqrt_f_theta_minus
                    losses = torch.square(err.abs())
                    loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
                
                
        
        return loss


    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, valid=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, valid=True)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        deep_inference_every_epoch = 10

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            if self.snr_conditioned == 'false':
                if self.model_type == 'bbed':
                    pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files, model_type='bbed')
                if self.model_type == 'sebridge':
                    pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files, model_type='sebridge')
                if self.model_type == 'sebridge_v2':
                    pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files, model_type='sebridge_v2')
            elif self.snr_conditioned == 'fixed':
                if self.model_type == 'sebridge_v2':
                    pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files, model_type='sebridge_v2_fixed', fixed_snr=self.fixed_snr)
                if self.model_type == 'sebridge_v3':
                    pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files, model_type='sebridge_v3_fixed', fixed_snr=self.fixed_snr)
            elif self.snr_conditioned == 'true':
                if self.model_type == 'sebridge_v2':
                    pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files, model_type='sebridge_v2_snr')
                if self.model_type == 'sebridge_v3':
                    pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files, model_type='sebridge_v3_snr', fixed_snr=self.fixed_snr)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

            if self.current_epoch % deep_inference_every_epoch == 0 and self.current_epoch >= deep_inference_every_epoch:
                if self.snr_conditioned == 'false':
                    if self.model_type == 'bbed':
                        si_sdr_0, si_sdr_5, si_sdr_10, si_sdr_15, si_sdr_20, si_sdr_25, si_sdr_30, si_sdr_35, si_sdr_40, pesq_0, pesq_5, pesq_10, pesq_15, pesq_20, pesq_25, pesq_30, pesq_35, pesq_40, estoi_0, estoi_5, estoi_10, estoi_15, estoi_20, estoi_25, estoi_30, estoi_35, estoi_40 = deep_evaluate_model(self, self.num_eval_files, model_type='bbed')
                    if self.model_type == 'sebridge':
                        si_sdr_0, si_sdr_5, si_sdr_10, si_sdr_15, si_sdr_20, si_sdr_25, si_sdr_30, si_sdr_35, si_sdr_40, pesq_0, pesq_5, pesq_10, pesq_15, pesq_20, pesq_25, pesq_30, pesq_35, pesq_40, estoi_0, estoi_5, estoi_10, estoi_15, estoi_20, estoi_25, estoi_30, estoi_35, estoi_40 = deep_evaluate_model(self, self.num_eval_files, model_type='sebridge')
                    if self.model_type == 'sebridge_v2':
                        si_sdr_0, si_sdr_5, si_sdr_10, si_sdr_15, si_sdr_20, si_sdr_25, si_sdr_30, si_sdr_35, si_sdr_40, pesq_0, pesq_5, pesq_10, pesq_15, pesq_20, pesq_25, pesq_30, pesq_35, pesq_40, estoi_0, estoi_5, estoi_10, estoi_15, estoi_20, estoi_25, estoi_30, estoi_35, estoi_40 = deep_evaluate_model(self, self.num_eval_files, model_type='sebridge_v2')
                elif self.snr_conditioned == 'fixed':
                    if self.model_type == 'sebridge_v2':
                        si_sdr_0, si_sdr_5, si_sdr_10, si_sdr_15, si_sdr_20, si_sdr_25, si_sdr_30, si_sdr_35, si_sdr_40, pesq_0, pesq_5, pesq_10, pesq_15, pesq_20, pesq_25, pesq_30, pesq_35, pesq_40, estoi_0, estoi_5, estoi_10, estoi_15, estoi_20, estoi_25, estoi_30, estoi_35, estoi_40 = deep_evaluate_model(self, self.num_eval_files, model_type='sebridge_v2_fixed', fixed_snr=self.fixed_snr)
                elif self.snr_conditioned == 'true':
                    if self.model_type == 'sebridge_v2':
                        si_sdr_0, si_sdr_5, si_sdr_10, si_sdr_15, si_sdr_20, si_sdr_25, si_sdr_30, si_sdr_35, si_sdr_40, pesq_0, pesq_5, pesq_10, pesq_15, pesq_20, pesq_25, pesq_30, pesq_35, pesq_40, estoi_0, estoi_5, estoi_10, estoi_15, estoi_20, estoi_25, estoi_30, estoi_35, estoi_40 = deep_evaluate_model(self, self.num_eval_files, model_type='sebridge_v2_snr')
                    if self.model_type == 'sebridge_v3':
                        si_sdr_0, si_sdr_5, si_sdr_10, si_sdr_15, si_sdr_20, si_sdr_25, si_sdr_30, si_sdr_35, si_sdr_40, pesq_0, pesq_5, pesq_10, pesq_15, pesq_20, pesq_25, pesq_30, pesq_35, pesq_40, estoi_0, estoi_5, estoi_10, estoi_15, estoi_20, estoi_25, estoi_30, estoi_35, estoi_40 = deep_evaluate_model(self, self.num_eval_files, model_type='sebridge_v3_snr', fixed_snr=self.fixed_snr)
            
            if self.snr_conditioned != 'fixed' and self.current_epoch % deep_inference_every_epoch == 0 and self.current_epoch >= deep_inference_every_epoch:
                self.log('pesq_-5', pesq_0, on_step=False, on_epoch=True)
                self.log('pesq_00', pesq_5, on_step=False, on_epoch=True)
                self.log('pesq_05', pesq_10, on_step=False, on_epoch=True)
                self.log('pesq_10', pesq_15, on_step=False, on_epoch=True)
                self.log('pesq_15', pesq_20, on_step=False, on_epoch=True)
                self.log('pesq_20', pesq_25, on_step=False, on_epoch=True)
                self.log('pesq_25', pesq_30, on_step=False, on_epoch=True)
                self.log('pesq_30', pesq_35, on_step=False, on_epoch=True)
                self.log('pesq_35', pesq_40, on_step=False, on_epoch=True)

                self.log('si_sdr_-5', si_sdr_0, on_step=False, on_epoch=True)
                self.log('si_sdr_00', si_sdr_5, on_step=False, on_epoch=True)
                self.log('si_sdr_05', si_sdr_10, on_step=False, on_epoch=True)
                self.log('si_sdr_10', si_sdr_15, on_step=False, on_epoch=True)
                self.log('si_sdr_15', si_sdr_20, on_step=False, on_epoch=True)
                self.log('si_sdr_20', si_sdr_25, on_step=False, on_epoch=True)
                self.log('si_sdr_25', si_sdr_30, on_step=False, on_epoch=True)
                self.log('si_sdr_30', si_sdr_35, on_step=False, on_epoch=True)
                self.log('si_sdr_35', si_sdr_40, on_step=False, on_epoch=True)

                self.log('estoi_-5', estoi_0, on_step=False, on_epoch=True)
                self.log('estoi_00', estoi_5, on_step=False, on_epoch=True)
                self.log('estoi_05', estoi_10, on_step=False, on_epoch=True)
                self.log('estoi_10', estoi_15, on_step=False, on_epoch=True)
                self.log('estoi_15', estoi_20, on_step=False, on_epoch=True)
                self.log('estoi_20', estoi_25, on_step=False, on_epoch=True)
                self.log('estoi_25', estoi_30, on_step=False, on_epoch=True)
                self.log('estoi_30', estoi_35, on_step=False, on_epoch=True)
                self.log('estoi_35', estoi_40, on_step=False, on_epoch=True)

        return loss

    def forward(self, x, t, y, s=None):
        # Concatenate y as an extra channel

        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        if self.snr_conditioned == 'false':
            if self.model_type == "bbed":
                score = -self.dnn(dnn_input, t)

            elif self.model_type == 'sebridge':
                eps, sigma_data = 0.001, 0.5
                c_skip =  sigma_data**2 / ((t-eps)**2 + sigma_data**2)
                c_out = (sigma_data*(t-eps))/((sigma_data**2+t**2)**0.5)
                t = t.squeeze(3).squeeze(2).squeeze(1)
                score = c_skip * x + c_out * self.dnn(dnn_input, t)


            elif self.model_type == 'sebridge_v2':
                eps, sigma_data = 0.001, 0.5
                
                # c_skip = 1 / ((t-eps) + 1)
                # c_out = (t-eps) / ((t-eps) + 1) # deprecated

                c_skip =  sigma_data**2 / ((t-eps)**2 + sigma_data**2)
                c_out = (sigma_data*(t-eps))/((sigma_data**2+t**2)**0.5)

                t = t.squeeze(3).squeeze(2).squeeze(1) 
                score = c_skip * x + c_out * self.dnn(dnn_input, t)
        
        elif self.snr_conditioned == 'fixed':
            
            if self.model_type == 'sebridge_v2':
                eps = 0.001
                c_skip = 1 / ((t-eps) + 1)
                c_out = (t-eps) / ((t-eps) + 1)

                t = t.squeeze(3).squeeze(2).squeeze(1)
                score = c_skip * x + c_out * self.dnn(dnn_input, t)
            
            if self.model_type == 'sebridge_v3':
                eps, sigma_data = 0.001, 0.5
                c_skip =  sigma_data**2 / ((t-eps)**2 + sigma_data**2)
                c_out = (sigma_data*(t-eps))/((sigma_data**2+t**2)**0.5)
                t = t.squeeze(3).squeeze(2).squeeze(1)
                score = c_skip * x + c_out * self.dnn(dnn_input, t)   
        
        elif self.snr_conditioned == 'true':
            if self.model_type == 'sebridge_v2':
                eps, sigma_data = 0.001, 0.5
                c_skip =  sigma_data**2 / ((t-eps)**2 + sigma_data**2)
                c_out = (sigma_data*(t-eps))/((sigma_data**2+t**2)**0.5)
                t = t.squeeze(3).squeeze(2).squeeze(1)
                score = c_skip * x + c_out * self.dnn(dnn_input, t)     
            
            if self.model_type == 'sebridge_v3':
                eps, sigma_data = 0.001, 0.5
                c_skip =  sigma_data**2 / ((t-eps)**2 + sigma_data**2)
                c_out = (sigma_data*(t-eps))/((sigma_data**2+t**2)**0.5)
                t = t.squeeze(3).squeeze(2).squeeze(1)
                score = c_skip * x + c_out * self.dnn(dnn_input, t)   

        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, Y_prior=None, N=None, minibatch=None, timestep_type=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, Y=y, Y_prior=Y_prior,
             timestep_type=timestep_type, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    y_prior_mini = Y_prior[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, Y=y_mini, y_prior=y_prior_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y,  Y_prior=None, N=None, minibatch=None, timestep_type=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, Y_prior=Y_prior,
             timestep_type=timestep_type, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()
    
    def val_dataloader_2(self):
        return self.data_module.val_dataloader_2()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def calculate_snr_direct(self, s, n, fixed_snr):
        snr = n / s
        return snr / (10**0.25 * fixed_snr) # for snr -5

    def calculate_normfac_direct(self, s, n, fixed_snr):
        #for snr -5
        normfac = (2.040166) * (0.240253 + 0.759747 * fixed_snr**2)**0.5 / ((1 + (n/s)**2)**0.5)
        return normfac



    def enhance_debug(self, y, x=None, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False, timestep_type=None,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        x = x / norm_factor
        
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        
        X = torch.unsqueeze(self._forward_transform(self._stft(x.cuda())), 0)
        X = pad_spec(X)
               
        if len(Y.shape)==4:
            Y = Y*self.preemp[None, None, :, None].to(device=Y.device)
            X = X*self.preemp[None, None, :, None].to(device=X.device)
        elif len(Y.self.shape)==3:
            Y = Y*self.preemp[None, :, None].to(device=Y.device)
            X = X*self.preemp[None, :, None].to(device=X.device)
        else:
            Y = Y*self.preemp[:, None].to(device=Y.device)
            X = X*self.preemp[:, None].to(device=X.device)
        
        Y_prior = self.sde._mean(X, torch.tensor([self.sde.T]).cuda(), Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), Y_prior = Y_prior.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False, timestep_type=timestep_type,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        
        
        sample = sample.squeeze()
        if len(sample.shape)==4:
            sample = sample*self.deemp[None, None, :, None].to(device=sample.device)
        elif len(sample.shape)==3:
            sample = sample*self.deemp[None, :, None].to(device=sample.device)
        else:
            sample = sample*self.deemp[:, None].to(device=sample.device)
        
        x_hat = self.to_audio(sample, T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
    
    def cos_sim(self, A, B):
         return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

    def enhance(self, x, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False, oracle=False, clean_rms=1, noise_rms=1,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        
        if self.snr_conditioned =='true':
            if oracle == False:
                y_snrcheck = y / y.abs().max().item()
                Y_snrcheck = torch.stft(y_snrcheck, n_fft=510, hop_length=128, center='True', window=torch.hann_window(510, periodic=True), return_complex=True).to('cuda')
                Y_snrcheck = torch.view_as_real(Y_snrcheck)
                Y_snrcheck = Y_snrcheck.permute(0, 3, 1, 2)
                Y_snrcheck = pad_spec_16(Y_snrcheck)
                est_gt = snr_model(Y_snrcheck)
                est_snr = est_gt / (1 - est_gt)
            elif oracle == True:
                est_snr = noise_rms / clean_rms
                est_snr = torch.FloatTensor([est_snr]).cuda()

        norm_factor = y.abs().max().item()

        if self.snr_conditioned == 'true':
            # normfac = self.calculate_normfac_direct(1, est_snr, self.fixed_snr)
            # norm_factor = norm_factor * normfac

            t_ = self.calculate_snr_direct(1, est_snr, self.fixed_snr)
            t_ = t_.detach().cpu().numpy()
            closest_t_index = np.abs(t_30 - t_).argmin()
            t_ = t_30[closest_t_index]

            est_snr_ = 10 ** 0.25 * self.fixed_snr * t_
            est_snr_ = torch.FloatTensor([est_snr_]).cuda()
            normfac_ = self.calculate_normfac_direct(1, est_snr_, self.fixed_snr)
            norm_factor = norm_factor * normfac_

        x = x.cuda()
        y = y.cuda()

        y = y / norm_factor
        x = x / norm_factor
    

        Y = torch.unsqueeze(self._forward_transform(self._stft(y)), 0)
        X = torch.unsqueeze(self._forward_transform(self._stft(x)), 0)
        Y = pad_spec(Y)
        X = pad_spec(X)

        

        if self.snr_conditioned == 'false':
            if self.model_type == 'bbed':
        
                if sampler_type == "pc":
                    sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                        corrector_steps=corrector_steps, snr=snr, intermediate=False,
                        **kwargs)
                elif sampler_type == "ode":
                    sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
                else:
                    print("{} is not a valid sampler type!".format(sampler_type))
                sample, nfe = sampler()
                sample = sample.squeeze()


            elif self.model_type == 'sebridge':
                vec_t = torch.ones(Y.shape[0], device=Y.device) * 0.999
                vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

                sample = self(Y.cpu(), vec_t.cpu(), Y.cpu())
                sample = sample.detach()

            elif self.model_type == 'sebridge_v2':
                vec_t = torch.ones(Y.shape[0], device=Y.device) * 0.999
                vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                z_mag = self.sigma_max

                Z = torch.randn_like(Y) * z_mag * 0.999
                X_T = Y + Z
                sample = self(X_T, vec_t, Y)
                sample = sample.detach()

                Y_X_diff = (Y.reshape(-1).detach() - X.reshape(-1).detach()).abs()
                sample_X_diff = (sample.reshape(-1) - X.reshape(-1).detach()).abs()
                Z_detach = Z.abs().reshape(-1).detach()
            
        elif self.snr_conditioned == 'fixed':
            raise NotImplementedError("snr fixed is only for experiment purpose, not real inference.")
        
        elif self.snr_conditioned == 'true':
            if self.model_type == 'sebridge_v2':
                z_scale = self.sigma_max
                z_mag = self.noise_mag(X, Y, mode='max') * z_scale

                vec_t = torch.ones(Y.shape[0], device=Y.device) * 0.999
                vec_s = torch.ones(Y.shape[0], device=Y.device) * z_mag * 0.999
                vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                vec_s = vec_s.unsqueeze(1).unsqueeze(2).unsqueeze(3)

                Z = torch.randn_like(Y) * z_mag * 0.999
                X_T = Y + Z
                sample = self(X_T, vec_t, Y, vec_s)
                sample = sample.detach()

            if self.model_type == 'sebridge_v3':
                z_scale = self.sigma_max
                t = self.calculate_snr_direct(1, est_snr, self.fixed_snr)

                t = t.detach().cpu().numpy()

                closest_t_index = np.abs(t_30 - t).argmin()
                t = t_30[closest_t_index]

                vec_t = torch.ones(Y.shape[0], device=Y.device) * t
                vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

                Z = torch.randn_like(Y) * z_scale * t
                X_T = Y + Z
                sample = self(X_T.cpu(), vec_t.cpu(), Y.cpu())
                sample = sample.detach()
        
        
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        if self.snr_conditioned == 'true':
            x_hat = x_hat * norm_factor.cpu()
        else:
            x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().detach().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat


    def prior_tests2(self, y, x, n):
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        x = x / norm_factor
        n = n / norm_factor
        
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        X = torch.unsqueeze(self._forward_transform(self._stft(x.cuda())), 0)
        X = pad_spec(X)
        
        #Ns = torch.unsqueeze(self._forward_transform(self._stft(n.cuda())), 0)
        #Ns = pad_spec(Ns)
        Ns = Y-X
        
        
        if len(Y.shape)==4:
            Y = Y*self.preemp[None, None, :, None].to(device=Y.device)
            Ns = Ns*self.preemp[None, None, :, None].to(device=Y.device)
            X = X*self.preemp[None, None, :, None].to(device=Y.device)
        elif len(Y.self.shape)==3:
            Y = Y*self.preemp[None, :, None].to(device=Y.device)
        else:
            Y = Y*self.preemp[:, None].to(device=Y.device)
            X = X*self.preemp[:, None].to(device=X.device)
            Ns = Ns*self.preemp[:, None].to(device=Ns.device)
        
        
        Yt, z = self.sde.prior_sampling(Y.shape, Y)
        Yt = Yt.to(Y.device)
        z = z.to(Y.device)
        
        vec_t = torch.ones(Y.shape[0], device=Y.device) * torch.tensor([1.0], device=Y.device)
        
        with torch.no_grad():
            
            grad = self(Yt, vec_t, Y)
            std = self.sde._std(vec_t)

            mp = Yt + grad*(std**2)
            mp_np = mp.squeeze().detach().cpu().numpy()
            
            z = z #/std
            z_np = z.squeeze().detach().cpu().numpy()
            
            Y_np = Y.squeeze().detach().cpu().numpy()
            Ns_np = Ns.squeeze().detach().cpu().numpy()
            X_np = X.squeeze().detach().cpu().numpy()
            
            Yt_np = Yt.squeeze().detach().cpu().numpy()
            grad_np = grad.squeeze().detach().cpu().numpy()
            
            res = z_np+grad_np
            err = np.exp(-1.5)*res/np.max(np.abs(res)) - np.exp(-1.5)*Ns_np/np.max(np.abs(Ns_np))
            #mean_res = (mp_np - Y_np) + 1e-8
            #err = (Ns_np)/(mean_res + 1e-8)
            
            fig, axs = plt.subplots(3, 3, figsize=(10,9), sharex=True, sharey=True)
            
            axs[1,0].imshow(20*np.log10(np.abs(grad_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[1,0].set_title('predicted score')
            axs[1,0].set_xlabel('time [s]')
            axs[1,0].set_ylabel('frequency [kHz]')
            
            axs[1,1].imshow(20*np.log10(np.abs(Yt_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[1,1].set_title('yT = y + z*sigma(T)')
            axs[1,1].set_xlabel('time [s]')
            axs[1,1].set_ylabel('frequency [kHz]')
            
            im = axs[1,2].imshow(20*np.log10(np.abs(mp_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[1,2].set_title('mean = yT + score*sigma(T)^2')
            axs[1,2].set_xlabel('time [s]')
            axs[1,2].set_ylabel('frequency [kHz]')
            
            
            im = axs[2,0].imshow(20*np.log10(np.abs(res/np.max(np.abs(res)))), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[2,0].set_title('score + z/sigma(T)')
            axs[2,0].set_xlabel('time [s]')
            axs[2,0].set_ylabel('frequency [kHz]')
            
            
            im = axs[0,2].imshow(20*np.log10(np.abs(Y_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[0,2].set_title('noisy mixture')
            axs[0,2].set_xlabel('time [s]')
            axs[0,2].set_ylabel('frequency [kHz]')
            
            
            im = axs[2,1].imshow(20*np.log10(np.abs(mp_np - Y_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[2,1].set_title('recon mean - noisy mixture')
            axs[2,1].set_xlabel('time [s]')
            axs[2,1].set_ylabel('frequency [kHz]')

            
            axs[0,0].imshow(20*np.log10(np.abs(X_np)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[0,0].set_title('Clean')
            axs[0,0].set_xlabel('time [s]')
            axs[0,0].set_ylabel('frequency [kHz]')
            
            axs[0,1].imshow(20*np.log10(np.abs(Ns_np/np.max(np.abs(Ns_np)))), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[0,1].set_title('environmental noise')
            axs[0,1].set_xlabel('time [s]')
            axs[0,1].set_ylabel('frequency [kHz]')
            
            
            axs[2,2].imshow(20*np.log10(np.abs(err)), aspect='auto', vmin=-30, vmax=30, origin='lower', cmap='magma')
            axs[2,2].set_title('err')
            axs[2,2].set_xlabel('time [s]')
            axs[2,2].set_ylabel('frequency [kHz]')
            
            fig.tight_layout()
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
            plt.show()
            plt.savefig('blub.png')
            a=2


    def get_prior(self, y, x, n, T=1):
            norm_factor = y.abs().max().item()
            y = y / norm_factor
            x = x / norm_factor
            n = n / norm_factor

            Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
            Y = pad_spec(Y)
            
            X = torch.unsqueeze(self._forward_transform(self._stft(x.cuda())), 0)
            diff_pad = Y.shape[-1] - X.shape[-1]
            X = pad_spec(X)
            
            Ns = Y - X

            if len(Y.shape)==4:
                Y = Y*self.preemp[None, None, :, None].to(device=Y.device)
                Ns = Ns*self.preemp[None, None, :, None].to(device=Y.device)
                X = X*self.preemp[None, None, :, None].to(device=Y.device)
            elif len(Y.self.shape)==3:
                Y = Y*self.preemp[None, :, None].to(device=Y.device)
            else:
                Y = Y*self.preemp[:, None].to(device=Y.device)
                X = X*self.preemp[:, None].to(device=X.device)
                Ns = Ns*self.preemp[:, None].to(device=Ns.device)
            
            if self.sde.__class__.__name__ == 'BBVE':
                self.sde.T = T
            Yt, z = self.sde.prior_sampling(Y.shape, Y)
            Yt = Yt.to(Y.device)
            z = z.to(Y.device)
            
            vec_t = torch.ones(Y.shape[0], device=Y.device) * torch.tensor([T], device=Y.device)
   
            grad = self(Yt, vec_t, Y, vec_t[:, None, None, None])
            std = self.sde._std(vec_t)

            mp = Yt + grad*(std**2)
            mp_np = mp.squeeze().detach().cpu().numpy()
            
            z = z/std
            z_np = z.squeeze().detach().cpu().numpy()
            
            Y_np = Y.squeeze().detach().cpu().numpy()
            X_np = X.squeeze().detach().cpu().numpy()
            Ns_np = Ns.squeeze().detach().cpu().numpy()
            
            Yt_np = Yt.squeeze().detach().cpu().numpy()
            grad_np = grad.squeeze().detach().cpu().numpy()
            
            res = z_np+grad_np

            return mp_np[:, :-diff_pad], X_np[:, :-diff_pad], Y_np[:, :-diff_pad], res[:, :-diff_pad], z_np[:, :-diff_pad], grad_np[:, :-diff_pad], Ns_np[:, :-diff_pad]
        

                
             
        
