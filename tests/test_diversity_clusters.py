"""diversity's group standardisation and cluster cut on constructed descriptor matrices."""

from __future__ import annotations

import numpy as np

from vla_tools.diversity import clusters
from vla_tools.diversity import standardise


def test_standardise_weighs_each_group_as_one_unit() -> None:
    rng = np.random.default_rng(0)
    z = standardise(rng.normal(5.0, 3.0, (20, 4)))
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose((z**2).sum(axis=1).mean(), 1.0, rtol=1e-9)


def test_two_separated_blobs_make_two_clusters() -> None:
    rng = np.random.default_rng(1)
    blobs = np.concatenate([rng.normal(0.0, 0.05, (6, 3)), rng.normal(3.0, 0.05, (6, 3))])
    labels = clusters(blobs, max_k=6)
    print(labels)
    assert len(set(labels[:6])) == 1 and len(set(labels[6:])) == 1 and labels[0] != labels[6]


def test_identical_or_too_few_episodes_make_one_cluster() -> None:
    assert clusters(np.ones((8, 3)), max_k=6).tolist() == [1] * 8
    assert clusters(np.arange(9.0).reshape(3, 3), max_k=6).tolist() == [1] * 3
