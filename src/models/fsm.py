"""Forgery Style Mixture (FSM).

Implements the Forgery Style Mixture module described in Sec. III-C of the OSDFD
paper, Fig. 6, Fig. 7 and Eqs. (6)-(9).

Key properties (paper):
  * Active only during training; deactivated at validation / inference
    ("In the inference stage, the forgery-style-mixture is deactivated").
  * Applied only to *fake* samples, grouped by their forgery source domain
    (manipulation type). Real features are left unchanged.
  * Mixes AdaIN-style feature statistics (channel-wise spatial mean / std)
    between the original ordering ``F_sort`` and a domain-shuffled ordering
    ``F~_sort`` where each fake sample is paired with a fake sample from a
    *different* forgery domain.
  * The mixing weight ``delta`` is sampled from ``Beta(0.1, 0.1)`` (Eqs. 7-8).
  * Activated per forward pass with probability ``prob`` (default 0.5),
    following MixStyle (cited as [73]).

AdaIN (Eq. 6) transfers the style of ``q`` onto content ``p``:
    AdaIN(p) = sigma(q) * (p - mu(p)) / sigma(p) + mu(q).

FSM mixes the statistics of the two orderings before applying them (Eq. 9):
    gamma_mix = delta * sigma(F_sort) + (1 - delta) * sigma(F~_sort)     (7)
    eta_mix   = delta * mu(F_sort)    + (1 - delta) * mu(F~_sort)        (8)
    F_mix'    = gamma_mix * (F_sort - mu(F_sort)) / sigma(F_sort) + eta_mix   (9)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _token_stats(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-wise spatial mean / std over the token dimension.

    Args:
        x: Patch-token features of shape ``(B, N, D)``.
        eps: Numerical stabiliser inside the square root.

    Returns:
        ``(mu, sigma)`` each of shape ``(B, 1, D)``.
    """
    mu = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    sigma = (var + eps).sqrt()
    return mu, sigma


class ForgeryStyleMixture(nn.Module):
    """Forgery Style Mixture over patch-token features (Eqs. 6-9).

    Args:
        prob: Probability of activating FSM on a given training forward pass.
        alpha: Both parameters of the ``Beta(alpha, alpha)`` mixing prior.
        eps: Numerical stabiliser for the standard deviation.
    """

    def __init__(self, prob: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.prob = prob
        self.eps = eps
        # Beta(alpha, alpha) prior on the mixing weight delta (Eqs. 7-8).
        self.beta = torch.distributions.Beta(alpha, alpha)

    @staticmethod
    def _domain_shuffle(domains: torch.Tensor) -> torch.Tensor:
        """Pair every fake sample with a fake sample of a *different* domain.

        Args:
            domains: Forgery-domain id per fake sample, shape ``(F,)``. Distinct
                integers denote distinct manipulation types.

        Returns:
            A permutation index tensor ``perm`` of shape ``(F,)`` such that
            ``domains[perm[i]] != domains[i]`` wherever the constraint is
            satisfiable. If fewer than two distinct domains are present the
            identity is returned (mixing is a no-op).
        """
        f = domains.numel()
        perm = torch.arange(f, device=domains.device)
        if torch.unique(domains).numel() < 2:
            return perm
        for i in range(f):
            # Candidate partners come from a different forgery domain.
            candidates = (domains != domains[i]).nonzero(as_tuple=True)[0]
            j = candidates[torch.randint(len(candidates), (1,), device=domains.device)]
            perm[i] = j
        return perm

    def forward(
        self,
        tokens: torch.Tensor,
        is_fake: torch.Tensor,
        domains: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FSM to the fake subset of a batch of token features.

        Args:
            tokens: Patch-token features, shape ``(B, N, D)``.
            is_fake: Boolean mask, shape ``(B,)`` (True for forgery samples).
            domains: Forgery-domain id per sample, shape ``(B,)`` (value ignored
                for real samples).

        Returns:
            Token features with FSM-mixed statistics on the fake samples; real
            samples and the token ordering are preserved. During ``eval`` or
            when the Bernoulli(prob) draw fails the input is returned unchanged.
        """
        if not self.training or self.prob <= 0.0:
            return tokens
        if torch.rand(1).item() >= self.prob:
            return tokens

        fake_idx = is_fake.nonzero(as_tuple=True)[0]
        if fake_idx.numel() < 2:
            return tokens

        fake_tokens = tokens[fake_idx]              # F_sort (fake subset)
        fake_domains = domains[fake_idx]
        perm = self._domain_shuffle(fake_domains)
        if torch.equal(perm, torch.arange(perm.numel(), device=perm.device)):
            return tokens  # could not satisfy the different-domain constraint

        shuffled = fake_tokens[perm]                # F~_sort

        mu, sigma = _token_stats(fake_tokens, self.eps)
        mu_s, sigma_s = _token_stats(shuffled, self.eps)

        delta = self.beta.sample((fake_tokens.size(0), 1, 1)).to(fake_tokens.device)
        gamma_mix = delta * sigma + (1.0 - delta) * sigma_s   # Eq. 7
        eta_mix = delta * mu + (1.0 - delta) * mu_s           # Eq. 8

        normalized = (fake_tokens - mu) / sigma
        mixed = gamma_mix * normalized + eta_mix              # Eq. 9

        # Restore original index order: scatter the mixed fake features back.
        out = tokens.clone()
        out[fake_idx] = mixed.to(tokens.dtype)
        return out
