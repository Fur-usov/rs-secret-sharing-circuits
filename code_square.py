"""Schur squares computed from row products and finite-field elimination."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

from field import PrimeField
from linear_algebra import Matrix, minimum_distance, normalize_matrix, row_basis


def componentwise_product(
    field: PrimeField, left: Sequence[int], right: Sequence[int]
) -> tuple[int, ...]:
    """Schur product (a_i b_i)_i of equally long vectors over GF(p)."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return tuple(field.mul(a, b) for a, b in zip(left, right))


def generator_products(field: PrimeField, generator: Matrix) -> tuple[tuple[int, ...], ...]:
    """All row_i * row_j for i <= j; their span is the square of the row space."""
    rows = normalize_matrix(field, generator)
    return tuple(
        componentwise_product(field, a, b) for a, b in combinations_with_replacement(rows, 2)
    )


@dataclass(frozen=True, init=False)
class CodeSquare:
    """C^{*2}=span{c*c': c,c' in C}, stored as an actually computed row basis."""

    field: PrimeField
    n: int
    basis: tuple[tuple[int, ...], ...]

    def __init__(self, field: PrimeField, generator: Matrix) -> None:
        rows = normalize_matrix(field, generator)
        if not rows or not rows[0]:
            raise ValueError("supply at least one row of positive length")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "n", len(rows[0]))
        object.__setattr__(self, "basis", row_basis(field, generator_products(field, rows)))

    @property
    def dimension(self) -> int:
        """Computed rank of all generator row products."""
        return len(self.basis)

    def minimum_distance(self, *, max_words: int = 100_000) -> int:
        """Actual minimum nonzero weight, using bounded exhaustive enumeration."""
        return minimum_distance(self.field, self.basis, max_words=max_words)
