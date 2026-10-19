from itertools import product

import pytest

from field import PrimeField


@pytest.mark.parametrize("p", [2, 5, 7, 11, 17])
def test_field_axioms(p):
    f = PrimeField(p)
    for a in range(p):
        assert f.add(a, 0) == f.mul(a, 1) == a
        assert f.sub(a, a) == f.add(a, -a) == 0
        assert f.pow(a, 0) == 1
        assert f.contains(a)
        if a:
            assert f.mul(a, f.inv(a)) == f.div(a, a) == 1
            assert f.pow(a, -2) == f.mul(f.inv(a), f.inv(a))
    for a, b in product(range(p), repeat=2):
        assert f.add(a, b) == f.add(b, a)
        assert f.mul(a, b) == f.mul(b, a)
        assert f.sub(a, b) == (a - b) % p
    for a, b, c in product(range(p), repeat=3):
        assert f.add(f.add(a, b), c) == f.add(a, f.add(b, c))
        assert f.mul(f.mul(a, b), c) == f.mul(a, f.mul(b, c))
        assert f.mul(a, f.add(b, c)) == f.add(f.mul(a, b), f.mul(a, c))


@pytest.mark.parametrize("p", [-11, 0, 1, 4, 9, 15, 25, 49, 121])
def test_composite_moduli(p):
    with pytest.raises(ValueError):
        PrimeField(p)


@pytest.mark.parametrize("p", [True, 5.0, "7"])
def test_noninteger_moduli(p):
    with pytest.raises(TypeError):
        PrimeField(p)


def test_representatives_and_errors():
    f = PrimeField(17)
    assert (f.add(12, 9), f.mul(5, 7), f.inv(5), f.div(10, 5)) == (4, 1, 7, 2)
    assert f.normalize(-18) == 16
    assert not any(f.contains(x) for x in [-1, 17, True, 1.0])
    for operation in [lambda: f.inv(17), lambda: f.div(1, 0), lambda: f.pow(0, -1)]:
        with pytest.raises(ZeroDivisionError):
            operation()
    for operation in [lambda: f.add(True, 1), lambda: f.pow(2, 1.5)]:
        with pytest.raises(TypeError):
            operation()
