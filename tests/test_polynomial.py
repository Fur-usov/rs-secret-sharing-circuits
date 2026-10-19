import pytest

from field import PrimeField
from polynomial import Polynomial, interpolate, lagrange_weights


@pytest.mark.parametrize("p", [5, 7, 11, 17])
def test_polynomial_identities_and_interpolation(p, rng):
    f = PrimeField(p)
    for _ in range(20):
        a, b, c = [Polynomial.random(f, 2, rng) for _ in range(3)]
        assert a + b == b + a
        assert (a + b) + c == a + (b + c)
        assert a * (b + c) == a * b + a * c
        assert a - a == Polynomial(f, [])
        for x in range(p):
            assert a(x) == sum(v * x**i for i, v in enumerate(a.coefficients)) % p
            assert (a * b)(x) == a(x) * b(x) % p
        xs = rng.sample(range(p), 3)
        assert interpolate(f, [(x, a(x)) for x in xs]) == a
        for at in range(p):
            assert sum(w * a(x) for w, x in zip(lagrange_weights(f, xs, at), xs)) % p == a(at)


def test_zero_constant_and_normalization(rng):
    f = PrimeField(7)
    zero = Polynomial(f, [0, 7, 0])
    assert zero.coefficients == (0,) and zero.degree == -1
    assert zero * Polynomial(f, [1, 2]) == zero
    assert Polynomial(f, [-1, 8, 0]).coefficients == (6, 1)
    assert interpolate(f, [(3, 4)]) == Polynomial(f, [4])
    assert interpolate(f, [(1, 0), (2, 0)]) == zero
    for degree in [-1, 0, 1, 5]:
        assert Polynomial.random(f, degree, rng).degree == degree


def test_invalid_polynomial_inputs():
    f = PrimeField(5)
    for points in [[], [(1, 2), (6, 3)]]:
        with pytest.raises(ValueError):
            interpolate(f, points)
    with pytest.raises(ValueError):
        lagrange_weights(f, [])
    with pytest.raises(ValueError):
        Polynomial.random(f, -2)
    for op in [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b]:
        with pytest.raises(ValueError):
            op(Polynomial(f, [1]), Polynomial(PrimeField(7), [1]))
