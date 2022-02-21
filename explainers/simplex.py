"""Adapted from https://github.com/JonathanCrabbe/Simplex/blob/main/explainers/simplex.py
All credit to JonathanCrabbe
"""

import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt


class Simplex:
    def __init__(self, corpus_latent_reps: torch.Tensor) -> None:
        """
        Initialize a SimplEx explainer
        :param corpus_examples: corpus input features
        :param corpus_latent_reps: corpus latent representations
        """
        self.corpus_latent_reps = corpus_latent_reps
        self.corpus_size = corpus_latent_reps.shape[0]
        self.dim_latent = corpus_latent_reps.shape[-1]
        self.weights = None
        self.n_test = None
        self.hist = None
        self.test_examples = None
        self.test_latent_reps = None
        self.jacobian_projections = None

    def fit(self, test_latent_reps: torch.Tensor,
            n_epoch: int = 10000, reg_factor: float = 1.0, n_keep: int = 5, reg_factor_scheduler=None) -> None:
        """
        Fit the SimplEx explainer on test examples
        :param test_examples: test example input features
        :param test_latent_reps: test example latent representations
        :param n_keep: number of neighbours used to build a latent decomposition
        :param n_epoch: number of epochs to fit the SimplEx
        :param reg_factor: regularization prefactor in the objective to control the number of allowed corpus members
        :param n_keep: number of corpus members allowed in the decomposition
        :param reg_factor_scheduler: scheduler for the variation of the regularization prefactor during optimization
        :return:
        """
        n_test = test_latent_reps.shape[0]
        preweights = torch.zeros((n_test, self.corpus_size), device=test_latent_reps.device, requires_grad=True)
        optimizer = torch.optim.Adam([preweights])
        hist = np.zeros((0, 2))
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            weights = F.softmax(preweights, dim=-1)
            corpus_latent_reps = torch.einsum('ij,jk->ik', weights, self.corpus_latent_reps)
            error = ((corpus_latent_reps - test_latent_reps) ** 2).sum()
            weights_sorted = torch.sort(weights)[0]
            regulator = (weights_sorted[:, : (self.corpus_size - n_keep)]).sum()
            loss = error + reg_factor * regulator
            loss.backward()
            optimizer.step()
            if (epoch+1) % (n_epoch / 5) == 0:
                print(f'Weight Fitting Epoch: {epoch+1}/{n_epoch} ; Error: {error.item():.3g} ;'
                      f' Regulator: {regulator.item():.3g} ; Reg Factor: {reg_factor:.3g}')
            if reg_factor_scheduler:
                reg_factor = reg_factor_scheduler.step(reg_factor)
            hist = np.concatenate((hist,
                                   np.array([error.item(), regulator.item()]).reshape(1, 2)),
                                  axis=0)
        self.weights = torch.softmax(preweights, dim=-1).detach()
        self.test_latent_reps = test_latent_reps
        self.n_test = n_test
        self.hist = hist

    def to(self, device: torch.device) -> None:
        """
        Transfer the tensors to device
        :param device: the device where the tensors should be transferred
        :return:
        """
        self.corpus_latent_reps = self.corpus_latent_reps.to(device)
        self.test_examples = self.test_examples.to(device)
        self.test_latent_reps = self.test_latent_reps.to(device)
        self.weights = self.weights.to(device)
        if self.jacobian_projections is not None:
            self.jacobian_projections = self.jacobian_projections.to(device)

    def latent_approx(self) -> torch.Tensor:
        """
        Returns the latent approximation of test_latent_reps with SimplEx
        :return: approximate latent representations as a tensor
        """
        approx_reps = self.weights @ self.corpus_latent_reps
        return approx_reps

    def decompose(self, test_id: int, return_id: bool = False) -> list or tuple:
        """
        Returns a complete corpus decomposition of the test example identified with test_id
        :param test_id: batch index of the test example
        :param return_id: specify the batch index of each corpus example in the decomposition
        :return: contribution of each corpus example in the decomposition in the form
                 [weight, features, jacobian projections]
        """
        assert test_id < self.n_test
        weights = self.weights[test_id].cpu().numpy()
        sort_id = np.argsort(weights)[::-1]
        if return_id:
            return [(weights[i], self.corpus_examples[i], self.jacobian_projections[i]) for i in sort_id], sort_id
        else:
            return [(weights[i], self.corpus_examples[i], self.jacobian_projections[i]) for i in sort_id]

    def plot_hist(self) -> None:
        """
        Plot the histogram that describes SimplEx fitting over the epochs
        :return:
        """
        sns.set()
        fig, axs = plt.subplots(2, sharex=True)
        epochs = [e for e in range(self.hist.shape[0])]
        axs[0].plot(epochs, self.hist[:, 0])
        axs[0].set(ylabel='Error')
        axs[1].plot(epochs, self.hist[:, 1])
        axs[1].set(xlabel='Epoch', ylabel='Regulator')
        plt.show()
