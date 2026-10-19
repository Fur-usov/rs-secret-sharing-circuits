from itertools import combinations, product

import pytest

from field import PrimeField
from linear_algebra import rank
from multiplication import (
    is_multiplicative,
    is_strongly_multiplicative,
    multiply_shares,
    product_reconstruction_weights,
    reconstruct_product,
)
from polynomial import Polynomial
from shamir import ShamirScheme


@pytest.mark.parametrize("p,n,t", [(5, 4, 0), (5, 4, 1), (7, 6, 2), (17, 8, 2), (101, 10, 3)])
def test_products_all_subsets_and_weights(p, n, t, rng):
    scheme = ShamirScheme(PrimeField(p), n, t)
    secrets = (
        list(product(range(p), repeat=2))
        if p <= 7
        else [(rng.randrange(p), rng.randrange(p)) for _ in range(30)] + [(0, 4), (5, 0)]
    )
    for a, b in secrets:
        local = multiply_shares(scheme, scheme.share(a, rng), reversed(scheme.share(b, rng)))
        assert reconstruct_product(scheme, local) == a * b % p
        for subset in combinations(local, 2 * t + 1):
            assert reconstruct_product(scheme, subset) == a * b % p
    weights = product_reconstruction_weights(scheme)
    for exponent in range(2 * t + 1):
        assert sum(w * pow(x, exponent, p) for w, x in zip(weights, scheme.points)) % p == (
            1 if exponent == 0 else 0
        )
    assert is_multiplicative(scheme)


@pytest.mark.parametrize("n,t", [(3, 1), (4, 1), (6, 2), (7, 2), (1, 0)])
def test_strong_multiplication_by_all_deletions(n, t, rng):
    scheme = ShamirScheme(PrimeField(11), n, t)
    local = multiply_shares(scheme, scheme.share(4, rng), scheme.share(7, rng))
    succeeds_for_all = True
    for size in range(t + 1):
        for removed in combinations(scheme.points, size):
            available = [(x, y) for x, y in local if x not in removed]
            if len(available) >= 2 * t + 1:
                assert reconstruct_product(scheme, available) == 6
            else:
                succeeds_for_all = False
                with pytest.raises(ValueError):
                    reconstruct_product(scheme, available)
    assert is_strongly_multiplicative(scheme) == succeeds_for_all


@pytest.mark.parametrize("t,m", [(1, 1), (1, 2), (2, 2), (2, 3), (2, 4)])
def test_insufficiency_is_mathematical_not_only_api(t, m):
    f = PrimeField(11)
    xs = list(range(1, m + 1))
    # No lambda satisfying sum lambda_i x_i**j = delta_{j0} for j=0,...,2t.
    rows = [[f.pow(x, j) for j in range(2 * t + 1)] for x in xs]
    target = [1] + [0] * (2 * t)
    assert rank(f, [*rows, target]) > rank(f, rows)
    # Explicit ambiguity among actual products of two degree <= t polynomials.
    a, b = Polynomial(f, [1]), Polynomial(f, [1])
    for i, x in enumerate(xs):
        if i < t:
            a = a * Polynomial(f, [-x, 1])
        else:
            b = b * Polynomial(f, [-x, 1])
    assert a.degree <= t and b.degree <= t
    assert all(a(x) * b(x) % f.p == 0 for x in xs)
    assert a(0) * b(0) % f.p != 0  # indistinguishable from product 0*0 on xs


def test_reconstruction_depends_only_on_product_shares(monkeypatch):
    scheme = ShamirScheme(PrimeField(7), 5, 1)
    # h=(2+3x)(4+x)=1+0x+3x^2 modulo 7; no original secrets supplied.
    local = [(x, (1 + 3 * x * x) % 7) for x in scheme.points]

    def forbidden(*args, **kwargs):
        raise AssertionError("must not create or reconstruct original sharings")

    monkeypatch.setattr(ShamirScheme, "share", forbidden)
    monkeypatch.setattr(ShamirScheme, "reconstruct", forbidden)
    assert reconstruct_product(scheme, local) == 1


def test_invalid_product_inputs(rng):
    scheme = ShamirScheme(PrimeField(7), 5, 1)
    shares = scheme.share(2, rng)
    for operation in [
        lambda: multiply_shares(scheme, shares, shares[:-1]),
        lambda: reconstruct_product(scheme, shares[:2]),
        lambda: product_reconstruction_weights(scheme, [1, 1, 2]),
        lambda: product_reconstruction_weights(scheme, [1, 2, 6]),
        lambda: reconstruct_product(scheme, [(1, 0), (2, 0), (3, 0), (4, 1)]),
    ]:
        with pytest.raises(ValueError):
            operation()
    assert not is_multiplicative(ShamirScheme(PrimeField(7), 2, 1))
