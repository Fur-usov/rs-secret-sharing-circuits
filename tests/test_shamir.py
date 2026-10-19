from collections import Counter
from itertools import combinations, product
from random import Random

import pytest

from field import PrimeField
from polynomial import Polynomial
from shamir import ShamirScheme


@pytest.mark.parametrize("p,n,t", [(5, 4, 0), (5, 4, 1), (7, 6, 2), (11, 6, 3), (17, 7, 2)])
def test_all_reconstruction_subsets(p, n, t, rng):
    scheme = ShamirScheme(PrimeField(p), n, t)
    for secret in range(p):
        shares = scheme.share(secret, rng)
        for subset in combinations(shares, scheme.k):
            assert scheme.reconstruct(subset) == secret
        assert scheme.reconstruct(reversed(shares)) == secret


def test_custom_points_and_uniform_zero_leading():
    class ZeroRng:
        def randrange(self, *args):
            return 0

    scheme = ShamirScheme(PrimeField(7), 3, 2, [-1, 2, 4])
    assert scheme.share(9, ZeroRng()) == ((6, 2), (2, 2), (4, 2))
    assert scheme.reconstruct(scheme.share(0)) == 0


@pytest.mark.parametrize(
    "n,t,points",
    [
        (0, 0, None),
        (5, 1, None),
        (3, -1, None),
        (3, 3, None),
        (3, 1, [0, 1, 2]),
        (3, 1, [1, 2, 6]),
        (3, 1, [1, 2]),
    ],
)
def test_invalid_parameters(n, t, points):
    with pytest.raises(ValueError):
        ShamirScheme(PrimeField(5), n, t, points)


@pytest.mark.parametrize(
    "shares",
    [[], [(1, 2)], [(1, 2), (6, 3)], [(0, 1), (1, 1)], [(4, 1), (1, 1)], [(1, 0), (2, 0), (3, 1)]],
)
def test_invalid_shares(shares):
    with pytest.raises(ValueError):
        ShamirScheme(PrimeField(5), 3, 1).reconstruct(shares)


@pytest.mark.parametrize("p,n,t", [(5, 4, 1), (5, 4, 2), (7, 3, 1)])
def test_exact_privacy(p, n, t):
    field = PrimeField(p)
    for size in range(t + 1):
        for coalition in combinations(range(1, n + 1), size):
            distributions = []
            for secret in range(p):
                counts = Counter()
                for randomness in product(range(p), repeat=t):
                    f = Polynomial(field, [secret, *randomness])
                    counts[tuple(f(x) for x in coalition)] += 1
                distributions.append(counts)
            assert all(d == distributions[0] for d in distributions)
            assert len(distributions[0]) == p**size


def test_statistical_privacy_smoke():
    # Fixed-seed frequency smoke test, not a proof of privacy.
    scheme = ShamirScheme(PrimeField(5), 4, 2)
    for secret in range(5):
        rng = Random(330 + secret)
        counts = Counter(scheme.share(secret, rng)[0][1] for _ in range(5000))
        assert all(abs(counts[value] - 1000) < 150 for value in range(5))
