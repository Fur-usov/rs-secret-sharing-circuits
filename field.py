"""Integer representatives of GF(p); exact trial division for small/medium moduli."""

from dataclasses import dataclass
from math import isqrt


def require_integer(value: int, name: str) -> int:
    """Reject booleans and non-integers rather than silently coercing them."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class PrimeField:
    """GF(p) with canonical integer elements 0,...,p-1.

    Primality testing costs O(sqrt(p)) divisions; not intended for cryptographic
    size primes. Elements carry no field tag: the owning object fixes the field.
    """

    p: int

    def __post_init__(self) -> None:
        require_integer(self.p, "p")
        if self.p < 2 or (self.p != 2 and self.p % 2 == 0):
            raise ValueError("p must be prime")
        if any(self.p % d == 0 for d in range(3, isqrt(self.p) + 1, 2)):
            raise ValueError("p must be prime")

    def normalize(self, a: int) -> int:
        """Return the canonical representative of an integer modulo p."""
        return require_integer(a, "element") % self.p

    def contains(self, a: object) -> bool:
        """Whether a is a canonical integer representative in this field."""
        return type(a) is int and 0 <= a < self.p

    def add(self, a: int, b: int) -> int:
        """Sum in GF(p)."""
        return (self.normalize(a) + self.normalize(b)) % self.p

    def sub(self, a: int, b: int) -> int:
        """Difference in GF(p)."""
        return (self.normalize(a) - self.normalize(b)) % self.p

    def mul(self, a: int, b: int) -> int:
        """Product in GF(p)."""
        return (self.normalize(a) * self.normalize(b)) % self.p

    def pow(self, a: int, exponent: int) -> int:
        """Integer power; negative powers invert first, and a**0 = 1."""
        require_integer(exponent, "exponent")
        base = self.normalize(a)
        if exponent < 0:
            base = self.inv(base)
            exponent = -exponent
        return pow(base, exponent, self.p)

    def inv(self, a: int) -> int:
        """Multiplicative inverse via the extended Euclidean algorithm."""
        a = self.normalize(a)
        if a == 0:
            raise ZeroDivisionError("zero has no multiplicative inverse")
        old_r, r, old_s, s = a, self.p, 1, 0
        while r:
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
        return old_s % self.p

    def div(self, a: int, b: int) -> int:
        """Quotient a/b in GF(p), rejecting a zero denominator."""
        return self.mul(a, self.inv(b))
