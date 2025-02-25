SNR-Aligned Consistent Diffusion for Adaptive Speech Enhancement
================================================================

This repository builds upon the code from https://github.com/sp-uhh/sgmse-bbed [1] by Signal Processing (SP), Universität Hamburg, which is licensed under MIT License.

Installation
------------
- Create a new virtual environment with Python 3.8
- Install the package dependencies via pip install -r requirements.txt
- Set up a wandb.ai account
- Log in via wandb login before running our code.

Training SNR Estimator
--------
You should train SNR estimator before training/testing the main model.
If you just want to train only, SNR estimator is not necessary but without it you can't track validation results while training.

Training SNR estimator can be run with 
```
python ./sgmse_bbed/train_snr_est.py --base_dir ../dataset/VBD_SNR-5/ --gpus=1  --num_eval_files 10 --transform_type none
```

Place the the trained checkpoint file of the SNR estimator into `./sgmse-bbed/sgmse/snr_estimator.ckpt`

Training
--------
Training is done by executing `./sgmse_bbed/train.py`.

Our training process can be run with
```
python ./sgmse_bbed/train.py --base_dir ../dataset/VBD_SNR-5/ --modeltype sebridge_v3 --transform_type exponent --loss_type mse --gpus=1 --sigma-max 1.0 --fixed_snr 0.17783 --snr_conditioned true --num_eval_files -1
```

`fixed_snr` can be 0.17783 for μ=10, 0.31623 for μ=5, or 0.56234 for μ=0.


Evaluation
----------
To evaluate on a test set, run
```
python ./sgmse_bbed/eval.py --test_dir ./dataset/VBD_SNR-5/valid --destination_folder {your destination folder here} --ckpt {your checkpoint file here}
```

If you want to evaluate SNR-specific test results, run
```
python ./sgmse_bbed/deep_eval.py --test_dir ./dataset/VBD_SNR-5/valid2 --destination_folder {your destination folder here} --ckpt {your checkpoint file here}
```

References
----------
[1] Bunlong Lay, Simon Welker, Julius Richter and Timo Gerkmann. Reducing the Prior Mismatch of Stochastic Differential Equations for Diffusion-based Speech Enhancement, ISCA Interspeech, 2023.
