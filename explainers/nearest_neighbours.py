"""Adapted from https://github.com/JonathanCrabbe/Simplex/blob/main/explainers/nearest_neighbours.py
All credit to JonathanCrabbe
"""

import torch
from sklearn.neighbors import KNeighborsRegressor


class NearNeighLatent:
    def __init__(self, corpus_latent_reps: torch.Tensor, weights_type: str = 'uniform') \
            -> None:
        """
        Initialize a latent nearest neighbours explainer
        :param corpus_latent_reps: corpus latent representations
        :param weights_type: type of KNN weighting (uniform or distance)
        """
        self.corpus_latent_reps = corpus_latent_reps
        self.weights_type = weights_type
        self.test_latent_reps = None
        self.regressor = None

    def fit(self, test_latent_reps: torch.Tensor, n_keep: int = 5) -> None:
        """
        Fit the nearest neighbour explainer on test examples
        :param test_latent_reps: test example latent representations
        :param n_keep: number of neighbours used to build a latent decomposition
        :return:
        """
        regressor = KNeighborsRegressor(n_neighbors=n_keep, weights=self.weights_type)
        regressor.fit(self.corpus_latent_reps.clone().detach().cpu().numpy(),
                      self.corpus_latent_reps.clone().detach().cpu().numpy())
        self.regressor = regressor
        self.test_latent_reps = test_latent_reps

    def latent_approx(self) -> torch.Tensor:
        """
        Returns the latent approximation of test_latent_reps with the nearest corpus neighbours
        :return: approximate latent representations as a tensor
        """
        approx_reps = self.regressor.predict(self.test_latent_reps.clone().detach().cpu().numpy())
        return torch.from_numpy(approx_reps).type(torch.float32)