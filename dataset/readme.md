Dataset setting
=

Path info
-
- VBD/train: VoiceBank-DEMAND train set (original)


- VBD_SNR-5/train: VoiceBank-DEMAND train set (all data SNR -5 using single_SNRize.ipynb)
- VBD_SNR-5/train2: VoiceBank-DEMAND valid set (original)
- VBD_SNR-5/valid: VoiceBank-DEMAND test set (original)
    - VBD_SNR-5/valid/active_rms.txt : pre-calculated average speech power/average noise power of VoiceBank-DEMAND test set (provided)
- VBD_SNR-5/valid2: VoiceBank-DEMAND test set (all data SNR -5 using single_SNRize.ipynb)

Setting guide
-
1. Download VoiceBank-DEMAND dataset
2. Copy train set into ./VBD/train
3. Copy valid/test set into ./VBD_SNR-5/train2, ./VBD_SNR-5/valid
4. Run ./single_SNRize.ipynb

Notes
-
- In our experiment, we excluded data with a `clean` and `noisy-clean` correlation above 0.02 during the training process and SNR-specific tests to ensure that only the noise component changes in the diffusion process.  
- However, for fairness, we did not remove them from the original test set and conducted the testing as usual.