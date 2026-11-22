"""Minimal Gaussian elimination and bounded enumeration over GF(p)."""

from collections.abc import Sequence
from itertools import product

from field import PrimeField, require_integer

Matrix = Sequence[Sequence[int]]


def normalize_matrix(field: PrimeField, matrix: Matrix) -> list[list[int]]:
    """Copy a rectangular matrix and normalize entries modulo p."""
    rows = [[field.normalize(a) for a in row] for row in matrix]
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be rectangular")
    return rows


def rref(field: PrimeField, matrix: Matrix) -> tuple[list[list[int]], tuple[int, ...]]:
    """Reduced row echelon form and pivot column indices, using exact elimination."""
    rows = normalize_matrix(field, matrix)
    if not rows:
        return rows, ()
    pivots = []
    for col in range(len(rows[0])):
        pivot = next((i for i in range(len(pivots), len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        current = len(pivots)
        rows[current], rows[pivot] = rows[pivot], rows[current]
        inverse = field.inv(rows[current][col])
        rows[current] = [field.mul(a, inverse) for a in rows[current]]
        for i in range(len(rows)):
            if i != current and rows[i][col]:
                factor = rows[i][col]
                rows[i] = [
                    field.sub(a, field.mul(factor, b)) for a, b in zip(rows[i], rows[current])
                ]
        pivots.append(col)
        if len(pivots) == len(rows):
            break
    return rows, tuple(pivots)


def rank(field: PrimeField, matrix: Matrix) -> int:
    """Dimension of the row space, computed by Gaussian elimination."""
    return len(rref(field, matrix)[1])


def row_basis(field: PrimeField, matrix: Matrix) -> tuple[tuple[int, ...], ...]:
    """Independent nonzero rows spanning the input matrix's row space."""
    rows, pivots = rref(field, matrix)
    return tuple(tuple(row) for row in rows[: len(pivots)])


def minimum_distance(field: PrimeField, generator: Matrix, *, max_words: int = 100_000) -> int:
    """Enumerate all words in a nonzero code and minimize nonzero Hamming weight.

    Refuse enumeration when p**rank exceeds max_words, before generating words.
    The zero code has no nonzero word and is rejected.
    """
    require_integer(max_words, "max_words")
    if max_words < 1:
        raise ValueError("max_words must be positive")
    basis = row_basis(field, generator)
    if not basis:
        raise ValueError("minimum nonzero weight is undefined for the zero code")
    count = 1
    for _ in basis:
        count *= field.p
        if count > max_words:
            raise ValueError("enumeration exceeds max_words")
    n = len(basis[0])
    best = n
    for coefficients in product(range(field.p), repeat=len(basis)):
        if not any(coefficients):
            continue
        word = [sum(a * row[j] for a, row in zip(coefficients, basis)) % field.p for j in range(n)]
        best = min(best, sum(v != 0 for v in word))
        if best == 1:
            break
    return best
