# Out of distribution detection using a corpus of latent representation examples

This repo contains a python notebook of experiments investigating if by collecting the latent representations of samples at training time out of distribution (OOD) samples can be detected at train time. The answer is yes!

## The idea

At inference time an approximation of the final hidden layer classification token created either by Simplex (see [Explaining Latent Representations with a Corpus of Examples](https://arxiv.org/abs/2110.15355) or distance weighted k nearest neighbours. The euclidean distance between this approximation and the real token used to evaluate if the sample is OOD. The intution being if the sample comes from the training distribiution (so is in scope) it's final latent state can be accurately approximated from the training corpus.

TODO: ADD PIC

By thresholding based on the 95% percentiles of the validation the following metrics are acheived for OOD detection on the clinic150 dataset:

TODO: ADD METRICS

Full details of the approach are given in [report.pdf](TODO)

## To Run

Clone repo and run notebook in a GPU enabled enviorment (running time ~TODO)

The clinic150 dataset needs to be download from [UCI](https://archive.ics.uci.edu/ml/datasets/CLINC150) and unzipped into a direcotry named data

If a GPU is not available the model weights for the transformer used for the in the report are available on [Google Drive](https://drive.google.com/drive/folders/1SkcFS-a2Ocs9vb-urJiiicaIf8JIEnFu?usp=sharing)
