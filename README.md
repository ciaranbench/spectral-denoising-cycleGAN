# spec-super-res-gan

The numpy files `noisy_data.npy` and `denoised_data.npy` can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1owS0jEbU93z9XDw_owVr5Fti1AVfQzL0?usp=sharing).

# TO DO:
- Code/train denoisining autoencoder - relevant paper should be in the project overleaf. \cite{abdolghader2021unsupervised} perhaps do single sample and multi sample version. Add this to eval notebook
- Implement cycleGAN in pytorch (MA)
- compare k means with fixed group size on clean, noisy, and adapted spectra as an eval metric (add to eval notebook)
- Train a supervised net for comparison ?

# ClirPATH to do
- ask around for a relevant conference to attend
- Propose application to get funding for webapp
- perhaps find a compelling clinical use-case for our denoising algo... apply for project funding if indeed we work something out (though now this seems less appealing than the above)
- discuss suitable journals to submit to if it all goes well


# Important notes
- Indeed, our method is unsupervised, but the network will still see the of the same sorts of samples in both domains. It just isn't directly paired. 
