"""Shamir t-private sharing: n participants, reconstruction threshold k=t+1."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from secrets import SystemRandom

from field import PrimeField, require_integer
from polynomial import Polynomial, RandomSource, interpolate, validate_points

Share = tuple[int, int]


@dataclass(frozen=True, init=False)
class ShamirScheme:
    """Share f(0) at fixed nonzero points, with deg f <= t and uniform randomness."""

    field: PrimeField
    n: int
    t: int
    points: tuple[int, ...]

    def __init__(
        self, field: PrimeField, n: int, t: int, points: Sequence[int] | None = None
    ) -> None:
        require_integer(n, "n")
        require_integer(t, "t")
        if not 0 <= t < n <= field.p - 1:
            raise ValueError("require 0 <= t < n <= p-1")
        xs = validate_points(field, range(1, n + 1) if points is None else points, nonzero=True)
        if len(xs) != n:
            raise ValueError("exactly n participant points are required")
        for name, value in [("field", field), ("n", n), ("t", t), ("points", xs)]:
            object.__setattr__(self, name, value)

    @property
    def k(self) -> int:
        """Minimum number of original shares needed for reconstruction."""
        return self.t + 1

    def share(self, secret: int, rng: RandomSource | None = None) -> tuple[Share, ...]:
        """Sample t independent uniform coefficients, including leading zeros."""
        rng = rng if rng is not None else SystemRandom()
        polynomial = Polynomial(
            self.field,
            [self.field.normalize(secret)] + [rng.randrange(self.field.p) for _ in range(self.t)],
        )
        return tuple((x, polynomial(x)) for x in self.points)

    def validate_shares(self, shares: Iterable[Share]) -> tuple[Share, ...]:
        """Normalize values and check distinct configured participant identifiers."""
        pairs = tuple(shares)
        xs = validate_points(self.field, [x for x, _ in pairs], nonzero=True)
        if not set(xs).issubset(self.points):
            raise ValueError("unknown participant point")
        return tuple((x, self.field.normalize(y)) for x, (_, y) in zip(xs, pairs))

    def reconstruct(self, shares: Iterable[Share]) -> int:
        """Recover f(0), rejecting insufficient or degree-inconsistent observations.

        Degree checks detect some corruptions, but do not authenticate shares.
        """
        pairs = self.validate_shares(shares)
        if len(pairs) < self.k:
            raise ValueError("at least t+1 shares are required")
        polynomial = interpolate(self.field, pairs[: self.k])
        if any(polynomial(x) != y for x, y in pairs[self.k :]):
            raise ValueError("shares are inconsistent with degree <= t")
        return polynomial(0)
