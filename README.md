# Out of Scope Detection Using a Corpus of Latent Representation Examples

This repo contains a python notebook of experiments investigating if by comparing the [simplex](https://proceedings.neurips.cc/paper/2021/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf) approximations test time latent states to those of the validation set out-of-scope samples can be detected. The answer is yes!

## The Idea

At inference time an approximation of the final hidden layer classification token created either simplex (see [Explaining Latent Representations with a Corpus of Examples](https://arxiv.org/abs/2110.15355). The euclidean distance between this approximation and the real token is called the corpus residual and is used to evaluate if the sample is OOD. The intution being if the sample comes from the training distribiution (so is in scope) it's final latent state can be accurately approximated from the training corpus.

![alt text](https://github.com/simonEllershaw/latent_variable_OOD/blob/main/figures/corpus_residuals.png)

By thresholding based on the 95% percentiles of the validation the following metrics are acheived for OOD detection on the clinic150 dataset (compared to baseline max proabability and kNN methods):

| Method          | Precision | Recall | F1    | AUC   |
|-----------------|-----------|--------|-------|-------|
| Max Probability | 0.803     | 0.826  | 0.814 | 0.886 |
| kNN             | 0.816     | 0.855  | 0.835 | 0.902 |
| Simplex         | 0.814     | 0.892  | 0.851 | 0.929 |

Full details of the approach are given in [OOS_Detection_using_Latent_States.pdf](https://github.com/simonEllershaw/latent_variable_OOD/blob/main/OOS_Detection_using_Latent_States.pdf)

## To Run

Clone repo and run notebook in a GPU enabled enviorment (running time ~25mins)

The clinic150 dataset needs to be download from [UCI](https://archive.ics.uci.edu/ml/datasets/CLINC150) and unzipped into a directory named data

If a GPU is not available the model weights for the transformer used for the in the report are available on [Google Drive](https://drive.google.com/file/d/1zO8r-P6CERgfr2f3eyawuhatArsuihU8/view?usp=sharing)

## Future Improvements
This repo is a proof concept and so corners have been cut on some implementation. Below is a non-exhaustive list of implementation details that could/should be improved in the future
- get_prob_pdf_has_greater_equal method is naively implemented and so v slow
- Transformer training and OOD detection should be seperated out
- Modularise simplex OOD dectection so easily extensible into any model framework (if it has linear last layer)
