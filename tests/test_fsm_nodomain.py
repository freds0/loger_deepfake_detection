"""Tests for the domain-free FSM variant (PLAN_v0.2 V11)."""

from __future__ import annotations

import torch

from src.models.fsm import ForgeryStyleMixture


def _batch():
    torch.manual_seed(0)
    tokens = torch.randn(6, 16, 8)
    is_fake = torch.tensor([0, 1, 1, 1, 1, 0]).bool()
    domains = torch.ones(6, dtype=torch.long)  # a single forgery domain
    return tokens, is_fake, domains


def test_nodomain_mixes_fakes_even_with_a_single_domain():
    tokens, is_fake, domains = _batch()
    fsm = ForgeryStyleMixture(prob=1.0, require_distinct_domains=False)
    fsm.train()
    out = fsm(tokens, is_fake=is_fake, domains=domains)
    assert not torch.allclose(out[is_fake], tokens[is_fake])  # fakes restyled
    assert torch.allclose(out[~is_fake], tokens[~is_fake])    # reals untouched


def test_default_requires_distinct_domains_and_noops_single_domain():
    tokens, is_fake, domains = _batch()
    fsm = ForgeryStyleMixture(prob=1.0)  # require_distinct_domains=True (default)
    fsm.train()
    out = fsm(tokens, is_fake=is_fake, domains=domains)
    assert torch.allclose(out, tokens)  # cannot pair distinct domains -> no-op


def test_derangement_is_a_permutation_without_fixed_points():
    for f in range(2, 25):
        perm = ForgeryStyleMixture._derangement(f, torch.device("cpu"))
        assert (perm != torch.arange(f)).all()
        assert sorted(perm.tolist()) == list(range(f))


def test_nodomain_still_inactive_in_eval_mode():
    tokens, is_fake, domains = _batch()
    fsm = ForgeryStyleMixture(prob=1.0, require_distinct_domains=False)
    fsm.eval()
    out = fsm(tokens, is_fake=is_fake, domains=domains)
    assert torch.allclose(out, tokens)
