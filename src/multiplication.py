"""Local Shamir multiplication and linear reconstruction, without degree reduction."""

from collections.abc import Iterable, Sequence

from polynomial import interpolate, lagrange_weights, validate_points
from shamir import ShamirScheme, Share


def is_multiplicative(scheme: ShamirScheme) -> bool:
    """Whether n >= 2t+1 allows universal linear product reconstruction."""
    return scheme.n >= 2 * scheme.t + 1


def is_strongly_multiplicative(scheme: ShamirScheme) -> bool:
    """Whether every deletion of <= t participants leaves >= 2t+1 shares."""
    return scheme.n >= 3 * scheme.t + 1


def multiply_shares(
    scheme: ShamirScheme, left: Iterable[Share], right: Iterable[Share]
) -> tuple[Share, ...]:
    """Multiply observations of the same participants, matching by x not order.

    Output evaluates a polynomial of degree <= 2t; it is not a fresh t-sharing.
    No threshold is needed for this local operation.
    """
    a, b = scheme.validate_shares(left), scheme.validate_shares(right)
    right_by_x = dict(b)
    if {x for x, _ in a} != set(right_by_x):
        raise ValueError("both inputs must refer to the same participants")
    return tuple((x, scheme.field.mul(y, right_by_x[x])) for x, y in a)


def product_reconstruction_weights(
    scheme: ShamirScheme, points: Sequence[int] | None = None
) -> tuple[int, ...]:
    """Weights lambda_i reconstructing h(0) for every polynomial deg h <= 2t.

    Weights correspond to the input point order. Insufficient configured point
    sets are rejected even when a specific product happens to have low degree.
    """
    xs = validate_points(scheme.field, scheme.points if points is None else points, nonzero=True)
    if not set(xs).issubset(scheme.points):
        raise ValueError("unknown participant point")
    if len(xs) < 2 * scheme.t + 1:
        raise ValueError("at least 2t+1 distinct product shares are required")
    return lagrange_weights(scheme.field, xs)


def reconstruct_product(scheme: ShamirScheme, shares: Iterable[Share]) -> int:
    """Recover only from product shares, without original secrets or polynomials.

    Require 2t+1 observations, and check extra observations for consistency.
    """
    pairs = scheme.validate_shares(shares)
    threshold = 2 * scheme.t + 1
    weights = product_reconstruction_weights(scheme, [x for x, _ in pairs])
    if len(pairs) > threshold:
        polynomial = interpolate(scheme.field, pairs[:threshold])
        if any(polynomial(x) != y for x, y in pairs[threshold:]):
            raise ValueError("product shares are inconsistent with degree <= 2t")
    return sum(scheme.field.mul(w, y) for w, (_, y) in zip(weights, pairs)) % scheme.field.p
