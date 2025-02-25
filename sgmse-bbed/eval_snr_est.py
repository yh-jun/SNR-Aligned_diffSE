import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from torchaudio import load
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd

from sgmse.data_module import SpecsDataModule
from sgmse.snr_estimator import SNRModel
import torch.nn.functional as F


from utils import energy_ratios, ensure_dir, print_mean_std

def spec_fwd(spec):
    e = 0.5
    spec = spec.abs()**e * torch.exp(1j * spec.angle())
    spec = spec * 0.15
    return spec

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--destination_folder", type=str, help="Name of destination folder.")
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')

    args = parser.parse_args()

    clean_dir = join(args.test_dir, "clean")
    noisy_dir = join(args.test_dir, "noisy")

    checkpoint_file = args.ckpt
    
    #please change this directory
    target_dir = "./sgmse-bbed/inference/snr_estimator"

    # Settings
    sr = 16000


    # Load score model
    model = SNRModel.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=1, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cpu()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))


    num_frames = 256
    hop_length = 128
    shuffle_spec = False
    normalize = "noisy"
    
    real_SNR = []
    est_real_SNR = []

    for cnt, noisy_file in tqdm(enumerate(noisy_files)):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file)

        target_len = (num_frames - 1) * hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')
        
        SNR = np.random.rand()*40
        real_SNR.append(SNR - 5)

        y = x + (y-x) * (10**(-SNR/20))

        if normalize == "noisy":
            normfac = y.abs().max()
        elif normalize == "clean":
            normfac = x.abs().max()
        elif normalize == "not":
            normfac = 1.0
        
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, n_fft=510, hop_length=hop_length, center='True', window=torch.hann_window(510, periodic=True), return_complex=True)
        Y = torch.stft(y, n_fft=510, hop_length=hop_length, center='True', window=torch.hann_window(510, periodic=True), return_complex=True)

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.

        Y = torch.view_as_real(Y)
        Y = Y.permute(0, 3, 1, 2)


        est_gt = model(Y)
        est_SNR = 20 * torch.log10((1-est_gt)/(est_gt))
        est_real_SNR.append(est_SNR)
        
        print('real:{0:.1f}/est:{1:.1f}'.format((SNR-5), est_SNR.item()))
        # Convert to numpy


        # Write enhanced wav file
        # write(target_dir + "/files/" + filename, x_hat, 16000)

        # Append metrics to data frame


    # Save results as DataFrame

    # # Save average results
