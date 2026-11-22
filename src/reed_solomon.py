"""Unweighted Reed–Solomon evaluation codes, without error correction."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from field import PrimeField, require_integer
from linear_algebra import minimum_distance
from polynomial import Polynomial, interpolate, validate_points


@dataclass(frozen=True, init=False)
class ReedSolomonCode:
    """RS(n,k) = {(f(x_1),...,f(x_n)): deg f < k}, with distinct x_i."""

    field: PrimeField
    n: int
    k: int
    points: tuple[int, ...]

    def __init__(
        self, field: PrimeField, n: int, k: int, points: Sequence[int] | None = None
    ) -> None:
        require_integer(n, "n")
        require_integer(k, "k")
        if not 1 <= k <= n <= field.p:
            raise ValueError("require 1 <= k <= n <= p")
        xs = validate_points(field, range(n) if points is None else points)
        if len(xs) != n:
            raise ValueError("exactly n evaluation points are required")
        for name, value in [("field", field), ("n", n), ("k", k), ("points", xs)]:
            object.__setattr__(self, name, value)

    @property
    def d(self) -> int:
        """Theoretical minimum distance n-k+1 (MDS property)."""
        return self.n - self.k + 1

    def encode(self, coefficients: Sequence[int]) -> tuple[int, ...]:
        """Evaluate a coefficient vector of length <= k, padding implicit zeros."""
        if len(coefficients) > self.k:
            raise ValueError("coefficient vector exceeds dimension k")
        return self.evaluate_polynomial(Polynomial(self.field, coefficients))

    def evaluate_polynomial(self, polynomial: Polynomial) -> tuple[int, ...]:
        """Evaluate a polynomial over the same field, requiring degree < k."""
        if polynomial.field != self.field:
            raise ValueError("polynomial belongs to a different field")
        if polynomial.degree >= self.k:
            raise ValueError("polynomial degree must be less than k")
        return tuple(polynomial(x) for x in self.points)

    def decode(self, coordinates: Iterable[tuple[int, int]]) -> Polynomial:
        """Interpolate from (zero-based coordinate index, value) pairs.

        Requires >= k distinct indices; surplus observations must agree. This
        is erasure reconstruction, not decoding in the presence of errors.
        """
        samples = tuple(coordinates)
        indices = [require_integer(i, "coordinate index") for i, _ in samples]
        if len(indices) != len(set(indices)) or any(not 0 <= i < self.n for i in indices):
            raise ValueError("coordinate indices must be distinct and in range")
        if len(samples) < self.k:
            raise ValueError("at least k coordinates are required")
        pairs = [(self.points[i], self.field.normalize(y)) for i, y in samples]
        polynomial = interpolate(self.field, pairs[: self.k])
        if any(polynomial(x) != y for x, y in pairs[self.k :]):
            raise ValueError("coordinates are inconsistent with degree < k")
        return polynomial

    def generator_matrix(self) -> tuple[tuple[int, ...], ...]:
        """Vandermonde generator with row j equal to evaluations of x**j."""
        return tuple(tuple(self.field.pow(x, j) for x in self.points) for j in range(self.k))

    def minimum_distance(self, *, max_words: int = 100_000) -> int:
        """Compute actual distance by bounded exhaustive codeword enumeration."""
        return minimum_distance(self.field, self.generator_matrix(), max_words=max_words)
