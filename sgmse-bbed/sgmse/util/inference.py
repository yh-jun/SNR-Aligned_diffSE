import torch
from torchaudio import load
import torch.nn.functional as F
from pesq import pesq
from pystoi import stoi
import numpy as np
from ..snr_estimator import SNRModel

from .other import si_sdr, pad_spec, pad_spec_16

# Settings
sr = 16000
snr = 0.5
N = 30
corrector_steps = 1

i_30 = np.arange(1, 30+1)
t_30 = (0.001 ** (1 / 7) + (i_30 - 1) / (30 - 1) * (1 ** (1 / 7) - 0.001 ** (1 / 7))) ** 7

def noise_mag(s, s_hat, mode='mean'):
        with torch.no_grad():
            if mode == 'mean':
                mag = torch.abs(torch.mean(torch.sqrt(torch.square(s-s_hat))))
            elif mode == 'max':
                mag = torch.max(torch.abs(s-s_hat))
            else:
                mag = 0
        return mag

def active_rms(clean, noise, fs=16000, energy_thresh=-50):

    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100 # in ms
    window_samples = int(fs*window_size/1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    clean = clean.squeeze()
    noise = noise.squeeze()

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = (noise_win**2).mean()**0.5

        # Considering frames with energy
        if noise_seg_rms > 10 ** (energy_thresh / 20) * (max(abs(noise))+np.finfo(float).eps): # if noise_seg_rms > energy_thresh: 에서 고쳤음
            noise_active_segs = np.append(noise_active_segs, noise_win)
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs)!=0:
        noise_rms = (noise_active_segs**2).mean()**0.5
    else:
        noise_rms = np.finfo(float).eps
        
    if len(clean_active_segs)!=0:
        clean_rms = (clean_active_segs**2).mean()**0.5
    else:
        clean_rms = np.finfo(float).eps

    return clean_rms, noise_rms

def calculate_snr(signal, noise):
    s, n = active_rms(signal, noise)
    snr = n / s
    return snr

def calculate_normfac(signal, noise):
    s, n = active_rms(signal, noise)
    normfac = (2**0.5) / ((1 + (n/s)**2)**0.5)
    return normfac

def calculate_snr_direct(s, n, fixed_snr):
    snr = n / s
    return snr / (10**0.25 * fixed_snr) # for snr -5

def calculate_normfac_direct(s, n, fixed_snr):
    #for snr -5 dataset
    normfac = (2.040166) * (0.240253 + 0.759747 * fixed_snr**2)**0.5 / ((1 + (n/s)**2)**0.5)
    return normfac

def evaluate_model(model, num_eval_files, model_type='bbed', fixed_snr=1.0):

    if model_type == 'sebridge_v3_fixed':
        clean_files = model.data_module.valid_set_2.clean_files
        noisy_files = model.data_module.valid_set_2.noisy_files
    else:
        clean_files = model.data_module.valid_set.clean_files
        noisy_files = model.data_module.valid_set.noisy_files

    clean_rms = model.data_module.valid_set.clean_rms
    noise_rms = model.data_module.valid_set.noise_rms
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)

    if num_eval_files == -1:
        num_eval_files = total_num_files

    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)

    clean_rms = list(clean_rms[i] for i in indices)
    noise_rms = list(noise_rms[i] for i in indices)


    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files

    if model_type == 'sebridge_v2_snr' or model_type == 'sebridge_v3_snr':
        snr_model = SNRModel.load_from_checkpoint(
            './sgmse-bbed/sgmse/snr_estimator.ckpt', base_dir="",
            batch_size=1, num_workers=0
        )
        snr_model.eval()
        snr_model.to('cuda')

    model.eval(no_ema=False)

    for (clean_file, noisy_file, clean_rms_, noise_rms_) in zip(clean_files, noisy_files, clean_rms, noise_rms):
        # Load wavs
        x, _ = load(clean_file)
        y, _ = load(noisy_file)
        T_orig = x.size(1)

        s = clean_rms_
        n = noise_rms_

        real_snr = n/s

        if model_type == 'sebridge_v2_snr' or model_type == 'sebridge_v3_snr':
            y_snrcheck = y / y.abs().max()
            Y_snrcheck = torch.stft(y_snrcheck, n_fft=510, hop_length=128, center='True', window=torch.hann_window(510, periodic=True), return_complex=True).to('cuda')
            Y_snrcheck = torch.view_as_real(Y_snrcheck)
            Y_snrcheck = Y_snrcheck.permute(0, 3, 1, 2)
            Y_snrcheck = pad_spec_16(Y_snrcheck)
            est_gt = snr_model(Y_snrcheck)
            est_snr = est_gt / (1 - est_gt)

        s = 1
        
        # print('real:{0:.3f}/est:{1:.3f}'.format(real_snr, est_snr.item()))
        # print('snr difference:{0:.1f}'.format(torch.abs(torch.log10(real_snr/est_snr)*20).item()))


        # Normalize per utterance
        norm_factor = y.abs().max()
        if model_type == 'sebridge_v2_snr' or model_type == 'sebridge_v3_snr':
            # normfac = calculate_normfac_direct(s, n, fixed_snr)
            normfac = calculate_normfac_direct(1, est_snr, fixed_snr)
            norm_factor = norm_factor * normfac
        
        y = y.to('cuda')
        x = x.to('cuda')

        y = y / norm_factor
        x = x / norm_factor

        if model_type == 'sebridge_v3_fixed':
            y = x + (y - x) * fixed_snr

        # Y_temp = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
    

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y)), 0)
        Y = pad_spec(Y)

        X = torch.unsqueeze(model._forward_transform(model._stft(x)), 0)
        X = pad_spec(X)

        y = y * norm_factor
        x = x * norm_factor

        # Reverse sampling
        if model_type == 'bbed':
            sampler = model.get_pc_sampler(
                'reverse_diffusion', 'ald', Y.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr)

            sample, _ = sampler()

            sample = sample.squeeze()

   
            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()

        elif model_type == 'sebridge':
            # dnn_input = torch.cat([y, y], dim=1)
            vec_t = torch.ones(Y.shape[0], device=Y.device) * 0.999
            vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            sample = model(Y, vec_t, Y)

            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()

        elif model_type == 'sebridge_v2':
            # dnn_input = torch.cat([y, y], dim=1)
            vec_t = torch.ones(Y.shape[0], device=Y.device) * 1
            vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            z_mag = model.sigma_max

            Z = torch.randn_like(Y) * z_mag * 1
            X_T = Y + Z
            sample = model(X_T, vec_t, Y)

            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
        
        elif model_type == 'sebridge_v2_fixed':
            # dnn_input = torch.cat([y, y], dim=1)
            vec_t = torch.ones(Y.shape[0], device=Y.device) * 0.999
            vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            noise_size = noise_mag(X, Y, mode='max')
            Y = X + (Y-X) / noise_size * fixed_snr

            z_mag = model.sigma_max

            Z = torch.randn_like(Y) * z_mag * 0.999
            X_T = Y + Z
            sample = model(X_T, vec_t, Y)

            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
        
        elif model_type == 'sebridge_v3_fixed':
            z_mag = model.sigma_max

            vec_t = torch.ones(Y.shape[0], device=Y.device)
            vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            Z = torch.randn_like(Y) * z_mag
            X_T = Y + Z
            sample = model(X_T, vec_t, Y)

            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
        
        
        elif model_type == 'sebridge_v2_snr':
            z_mag = model.sigma_max
            
            t = calculate_snr_direct(s, n)

            # print('z_mag inference:{0}'.format(z_mag))

            vec_t = torch.ones(Y.shape[0], device=Y.device) * t
            vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            Z = torch.randn_like(Y) * z_mag * t
            X_T = Y + Z
            sample = model(X_T, vec_t, Y)

            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor 

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
        
        elif model_type == 'sebridge_v3_snr':
            z_mag = model.sigma_max
            t = calculate_snr_direct(1, est_snr, fixed_snr)
            t = t.detach().cpu().numpy()
            
            closest_t_index = np.abs(t_30 - t).argmin()
            t = t_30[closest_t_index]

            vec_t = torch.ones(Y.shape[0], device=Y.device) * t
            vec_t = vec_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            Z = torch.randn_like(Y) * z_mag * t
            X_T = Y + Z #######
            sample = model(X_T, vec_t, Y)

            x_hat = model.to_audio(sample.squeeze(), T_orig)
            # norm_factor = norm_factor.to('cuda')
            x_hat = x_hat * norm_factor

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
    
        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb')
        _estoi += stoi(x, x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files