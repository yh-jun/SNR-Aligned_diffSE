SNR-Aligned Consistent Diffusion for Adaptive Speech Enhancement
================================================================

This repository contains the official PyTorch implementations for the Interspeech 2025 paper:
- *SNR-Aligned Consistent Diffusion for Adaptive Speech Enhancement* [1]

It builds upon the code from https://github.com/sp-uhh/sgmse-bbed [2] by Signal Processing (SP), Universität Hamburg, which is licensed under MIT License.

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

`fixed_snr` can be 0.17783 for η=10, 0.31623 for η=5, or 0.56234 for η=0.


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

We offer pretrained checkpoints for M4, M5, M6, SE-Bridge baseline, and our SNR estimator. (Aside from the use of oracle SNR values, M1–M3 and M4–M6 are identical)
Our checkpoints can be downloaded [here](https://drive.google.com/drive/folders/12xatVSNG1mhjGSW9vppM8AGibo-HKCwK?usp=sharing).

The other baselines, [SGMSE+](https://github.com/sp-uhh/sgmse)[3] and [StoRM](https://github.com/sp-uhh/storm)[4], were evaluated using their official implementations.

References
----------
[1] Y. Jun, B. J. Woo, M. Jeong and N. S. Kim. SNR-Aligned Consistent Diffusion for Adaptive Speech Enhancement, ISCA Interspeech, 2025.

[2] B. Lay, S. Welker, J. Richter and T. Gerkmann. Reducing the Prior Mismatch of Stochastic Differential Equations for Diffusion-based Speech Enhancement, ISCA Interspeech, 2023.

[3] J. Richter, S. Welker, J.-M. Lemercier, B. Lay, and T. Gerkmann. Speech enhancement and dereverberation with diffusion-based generative models. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:2351–2364, 2023.

[4] J.-M. Lemercier, J. Richter, S. Welker, and T. Gerkmann. StoRM: A diffusion-based stochastic regeneration model for speech enhancement and dereverberation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.

