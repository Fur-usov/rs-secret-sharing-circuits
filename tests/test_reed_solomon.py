from itertools import combinations, product

import pytest

from field import PrimeField
from linear_algebra import rank
from polynomial import Polynomial
from reed_solomon import ReedSolomonCode


@pytest.mark.parametrize("p,n,k", [(5, 5, 1), (5, 4, 2), (7, 6, 3), (11, 7, 4), (17, 9, 5)])
def test_encoding_linearity_generator_and_decoding(p, n, k, rng):
    code = ReedSolomonCode(PrimeField(p), n, k, rng.sample(range(p), n))
    assert rank(code.field, code.generator_matrix()) == k
    for _ in range(10):
        a, b = [[rng.randrange(p) for _ in range(k)] for _ in range(2)]
        scalar = rng.randrange(p)
        combined = [(x + scalar * y) % p for x, y in zip(a, b)]
        ca, cb = code.encode(a), code.encode(b)
        assert code.encode(combined) == tuple((x + scalar * y) % p for x, y in zip(ca, cb))
        assert ca == tuple(
            sum(c * row[j] for c, row in zip(a, code.generator_matrix())) % p for j in range(n)
        )
        for indices in combinations(range(n), k):
            assert code.decode([(i, ca[i]) for i in indices]) == Polynomial(code.field, a)
        assert code.decode(enumerate(ca)) == Polynomial(code.field, a)


@pytest.mark.parametrize("p,n,k", [(5, 4, 1), (5, 4, 2), (7, 5, 3), (5, 5, 5)])
def test_distance_exhaustively(p, n, k):
    code = ReedSolomonCode(PrimeField(p), n, k)
    words = {code.encode(c) for c in product(range(p), repeat=k)}
    assert len(words) == p**k
    expected = min(sum(v != 0 for v in word) for word in words if any(word))
    assert code.minimum_distance() == expected == code.d


@pytest.mark.parametrize(
    "n,k,points", [(6, 2, None), (3, 0, None), (3, 4, None), (3, 2, [1, 1, 2]), (3, 2, [1, 2])]
)
def test_invalid_code(n, k, points):
    with pytest.raises(ValueError):
        ReedSolomonCode(PrimeField(5), n, k, points)


def test_invalid_encoding_decoding_and_budget():
    code = ReedSolomonCode(PrimeField(5), 4, 2)
    for samples in [
        [],
        [(0, 1)],
        [(0, 1), (0, 2)],
        [(-1, 0), (1, 0)],
        [(0, 0), (4, 0)],
        [(0, 0), (1, 0), (2, 1)],
    ]:
        with pytest.raises(ValueError):
            code.decode(samples)
    for operation in [
        lambda: code.encode([1, 2, 0]),
        lambda: code.evaluate_polynomial(Polynomial(PrimeField(7), [1])),
        lambda: code.evaluate_polynomial(Polynomial(code.field, [0, 0, 1])),
        lambda: code.minimum_distance(max_words=24),
    ]:
        with pytest.raises(ValueError):
            operation()
    assert code.encode([]) == (0, 0, 0, 0)
