# siamtrans
SiamTrans: Zero-Shot Multi-Frame Image Restoration with Pre-Trained Siamese Transformers AAAI 2022

The code will come soon !

We propose a novel zero-shot multi-frame image restoration method for removing unwanted obstruction elements (such as rains, snow, and moire patterns) that vary in successive frames.It has three stages: transformer pre-training, zero-shot restoration, and hard patch refinement. Using the pre-trained transformers, our model is able to tell the motion difference between the true image information and the obstructing elements. For zero-shot image restoration, we design a novel model, termed SiamTrans, which is constructed by Siamese transformers, encoders, and decoders. Each transformer has a temporal attention layer and several self-attention layers, to capture both temporal and spatial information of multiple frames. Only unsupervisedly pre-trained on the denoising task, SiamTrans is tested on three different low-level vision tasks (deraining, demoireing, and desnowing).
Compared with related methods, SiamTrans achieves the best performances, even outperforming those with supervised learning.
