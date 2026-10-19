from itertools import product

import pytest

from code_square import CodeSquare, componentwise_product, generator_products
from field import PrimeField
from linear_algebra import minimum_distance, rank, row_basis, rref
from reed_solomon import ReedSolomonCode


def span(field, rows, n):
    return {
        tuple(sum(c * row[j] for c, row in zip(coefficients, rows)) % field.p for j in range(n))
        for coefficients in product(range(field.p), repeat=len(rows))
    }


@pytest.mark.parametrize(
    "p,n,k", [(5, 4, 1), (5, 4, 2), (5, 4, 3), (7, 6, 2), (7, 6, 4), (11, 9, 3), (17, 10, 7)]
)
def test_rs_square_computed_spaces(p, n, k):
    f = PrimeField(p)
    code = ReedSolomonCode(f, n, k)
    square = CodeSquare(f, code.generator_matrix())
    expected = ReedSolomonCode(f, n, min(n, 2 * k - 1))
    assert square.dimension == expected.k
    assert rank(f, [*square.basis, *expected.generator_matrix()]) == expected.k
    for a in code.generator_matrix():
        for b in code.generator_matrix():
            assert rank(f, [*square.basis, componentwise_product(f, a, b)]) == square.dimension


@pytest.mark.parametrize("p,n,k", [(5, 4, 1), (5, 4, 2), (5, 4, 3)])
def test_exhaustive_square_closure_and_distance(p, n, k):
    f = PrimeField(p)
    code = ReedSolomonCode(f, n, k, [4, 2, 3, 1])
    square = CodeSquare(f, code.generator_matrix())
    words = span(f, square.basis, n)
    expected = ReedSolomonCode(f, n, min(n, 2 * k - 1), code.points)
    assert words == {expected.encode(c) for c in product(range(p), repeat=expected.k)}
    # Enumerate every original codeword pair, including linear combinations.
    originals = {code.encode(c) for c in product(range(p), repeat=k)}
    assert all(componentwise_product(f, a, b) in words for a in originals for b in originals)
    for a in square.basis:
        for b in words:
            for scalar in range(p):
                assert tuple((scalar * x + y) % p for x, y in zip(a, b)) in words
    assert square.minimum_distance() == min(sum(v != 0 for v in w) for w in words if any(w))
    assert square.minimum_distance() == expected.d


def test_gaussian_elimination_and_generic_square():
    f = PrimeField(5)
    matrix = [[0, 1, 2], [0, 2, 4], [1, 0, 1], [1, 1, 3]]
    reduced, pivots = rref(f, matrix)
    assert pivots == (0, 1)
    assert reduced == [[1, 0, 1], [0, 1, 2], [0, 0, 0], [0, 0, 0]]
    assert rank(f, []) == rank(f, [[], []]) == 0
    assert rank(f, [[5, 0], [0, -5]]) == 0
    assert len(row_basis(f, matrix)) == 2
    generator = [[1, 1, 0, 0], [0, 1, 1, 0]]
    square = CodeSquare(f, generator)
    words = span(f, square.basis, 4)
    assert square.dimension == 3 and square.minimum_distance() == 1
    assert set(generator_products(f, generator)).issubset(words)
    for a, b in product(span(f, generator, 4), repeat=2):
        assert componentwise_product(f, a, b) in words
    assert CodeSquare(f, [[0, 0]]).dimension == 0


def test_matrix_errors_and_enumeration_guard():
    f = PrimeField(5)
    for operation in [
        lambda: rref(f, [[1], [1, 2]]),
        lambda: componentwise_product(f, [1], []),
        lambda: CodeSquare(f, []),
        lambda: CodeSquare(f, [[]]),
        lambda: minimum_distance(f, [[0, 0]]),
        lambda: minimum_distance(f, [[1]], max_words=0),
        lambda: minimum_distance(f, [[1]], max_words=4),
    ]:
        with pytest.raises(ValueError):
            operation()
