"""Polynomials with coefficients in ascending degree order."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from secrets import SystemRandom
from typing import Protocol

from field import PrimeField, require_integer


class RandomSource(Protocol):
    """Minimal injectable RNG interface; defaults use SystemRandom."""

    def randrange(self, start: int, stop: int | None = None) -> int:
        """Sample uniformly from a half-open integer interval."""
        ...


@dataclass(frozen=True, init=False)
class Polynomial:
    """Immutable polynomial; zero is (0,) and has degree -1."""

    field: PrimeField
    coefficients: tuple[int, ...]

    def __init__(self, field: PrimeField, coefficients: Iterable[int]) -> None:
        values = [field.normalize(c) for c in coefficients] or [0]
        while len(values) > 1 and values[-1] == 0:
            values.pop()
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "coefficients", tuple(values))

    @property
    def degree(self) -> int:
        """Largest nonzero coefficient index, or -1 for zero."""
        return -1 if self.coefficients == (0,) else len(self.coefficients) - 1

    def _check(self, other: "Polynomial") -> None:
        if not isinstance(other, Polynomial):
            raise TypeError("operand must be a Polynomial")
        if self.field != other.field:
            raise ValueError("polynomials belong to different fields")

    def __add__(self, other: "Polynomial") -> "Polynomial":
        """Coefficientwise sum over the common field."""
        self._check(other)
        result = [0] * max(len(self.coefficients), len(other.coefficients))
        for source in (self.coefficients, other.coefficients):
            for i, c in enumerate(source):
                result[i] = self.field.add(result[i], c)
        return Polynomial(self.field, result)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        """Coefficientwise difference over the common field."""
        self._check(other)
        return self + Polynomial(self.field, [-c for c in other.coefficients])

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        """Convolution product over the common field."""
        self._check(other)
        result = [0] * (len(self.coefficients) + len(other.coefficients) - 1)
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                result[i + j] = self.field.add(result[i + j], self.field.mul(a, b))
        return Polynomial(self.field, result)

    def evaluate(self, x: int) -> int:
        """Evaluate with Horner's method in GF(p)."""
        x = self.field.normalize(x)
        value = 0
        for c in reversed(self.coefficients):
            value = self.field.add(self.field.mul(value, x), c)
        return value

    __call__ = evaluate

    @classmethod
    def random(
        cls, field: PrimeField, degree: int, rng: RandomSource | None = None
    ) -> "Polynomial":
        """Sample an exact-degree polynomial (leading coefficient nonzero).

        This distribution must not be used for Shamir's random coefficients.
        Degree -1 explicitly requests the zero polynomial.
        """
        require_integer(degree, "degree")
        if degree < -1:
            raise ValueError("degree must be at least -1")
        if degree == -1:
            return cls(field, [0])
        rng = rng if rng is not None else SystemRandom()
        return cls(
            field, [rng.randrange(field.p) for _ in range(degree)] + [rng.randrange(1, field.p)]
        )


def validate_points(
    field: PrimeField, points: Iterable[int], *, nonzero: bool = False
) -> tuple[int, ...]:
    """Normalize distinct abscissae; reject collisions modulo p."""
    result = tuple(field.normalize(x) for x in points)
    if len(set(result)) != len(result):
        raise ValueError("evaluation points must be distinct modulo p")
    if nonzero and 0 in result:
        raise ValueError("participant points must be nonzero")
    return result


def lagrange_weights(field: PrimeField, points: Sequence[int], at: int = 0) -> tuple[int, ...]:
    """Return lambda_i with f(at) = sum_i lambda_i f(x_i), deg f < len(points)."""
    xs = validate_points(field, points)
    if not xs:
        raise ValueError("at least one point is required")
    at = field.normalize(at)
    weights = []
    for i, xi in enumerate(xs):
        numerator, denominator = 1, 1
        for j, xj in enumerate(xs):
            if i != j:
                numerator = field.mul(numerator, field.sub(at, xj))
                denominator = field.mul(denominator, field.sub(xi, xj))
        weights.append(field.div(numerator, denominator))
    return tuple(weights)


def interpolate(field: PrimeField, points: Iterable[tuple[int, int]]) -> Polynomial:
    """Construct the unique polynomial of degree < m through m distinct points."""
    pairs = tuple(points)
    xs = validate_points(field, [x for x, _ in pairs])
    if not pairs:
        raise ValueError("at least one point is required")
    result = Polynomial(field, [0])
    for i, (xi, yi) in enumerate(pairs):
        basis, denominator = Polynomial(field, [1]), 1
        for j, xj in enumerate(xs):
            if i != j:
                basis = basis * Polynomial(field, [-xj, 1])
                denominator = field.mul(denominator, field.sub(xi, xj))
        result = result + basis * Polynomial(field, [field.div(yi, denominator)])
    return result
