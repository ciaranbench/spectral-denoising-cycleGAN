# spec-super-res-gan

The numpy files `noisy_data.npy` and `denoised_data.npy` can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1owS0jEbU93z9XDw_owVr5Fti1AVfQzL0?usp=sharing).

# TO DO:
- Ensure there is no data leakage in parsing 
- Expand implementation of classical denoising techniques and evaluation (see denoising notebook) (CB)
- Code/train denoisining autoencoder - relevant paper should be in the project overleaf. \cite{abdolghader2021unsupervised} (MA)
- Implement cycleGAN in pytorch (MA)
- Clean/comment tf cycleGAN implementation (CB)
- Train a supervised net for comparison
- Denoising but for images ?