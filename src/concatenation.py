"""Explicit hierarchical composition of two Shamir schemes over the same GF(p).

This is not claimed to implement the field-extension concatenation of any paper.
Each outer evaluation is independently shared by an inner Shamir scheme.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from field import require_integer
from multiplication import is_multiplicative, reconstruct_product
from polynomial import RandomSource
from shamir import ShamirScheme

Participant = tuple[int, int]


@dataclass(frozen=True)
class CompositeShare:
    """One field element held by leaf (block, participant), using zero-based indices."""

    block: int
    participant: int
    value: int

    @property
    def identifier(self) -> Participant:
        """Public leaf identifier, independent of the evaluation points."""
        return self.block, self.participant


@dataclass(frozen=True)
class HierarchicalShamirComposition:
    """Share s by F(0)=s, then share F(x_i) independently by G_i(0)=F(x_i).

    Leaf (i,j) receives G_i(z_j). Degrees are <= outer.t and <= inner.t.
    Authorization requires outer.k blocks with at least inner.k leaves each.
    Both schemes must use the same field. No strong multiplicativity property
    for the composed access structure is asserted by this API.
    """

    outer: ShamirScheme
    inner: ShamirScheme

    def __post_init__(self) -> None:
        if self.outer.field != self.inner.field:
            raise ValueError("both levels must use the same prime field")

    @property
    def participants(self) -> tuple[Participant, ...]:
        """All final participants, grouped into disjoint outer blocks."""
        return tuple((i, j) for i in range(self.outer.n) for j in range(self.inner.n))

    @property
    def total_shares(self) -> int:
        """Number of final field-element shares, outer.n * inner.n."""
        return self.outer.n * self.inner.n

    @property
    def share_size_bits(self) -> int:
        """Fixed-width payload bits per leaf; excludes public indices and serialization."""
        return (self.outer.field.p - 1).bit_length()

    def access_parameters(self) -> dict[str, int]:
        """Summarize block thresholds and cardinality bounds, not a single threshold.

        Every unauthorized set is perfectly private with independent uniform
        coefficients. The cardinality bound alone does not describe access.
        """
        minimum_authorized = self.outer.k * self.inner.k
        maximum_unauthorized = (
            self.outer.t * self.inner.n + (self.outer.n - self.outer.t) * self.inner.t
        )
        return {
            "required_blocks": self.outer.k,
            "required_leaves_per_block": self.inner.k,
            "minimum_authorized_size": minimum_authorized,
            "maximum_unauthorized_size": maximum_unauthorized,
            "guaranteed_privacy_size": minimum_authorized - 1,
            "guaranteed_reconstruction_size": maximum_unauthorized + 1,
        }

    def _validate_identifiers(self, participants: Iterable[Participant]) -> tuple[Participant, ...]:
        ids = tuple(participants)
        for i, j in ids:
            require_integer(i, "block index")
            require_integer(j, "inner participant index")
            if not 0 <= i < self.outer.n or not 0 <= j < self.inner.n:
                raise ValueError("unknown composite participant")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate composite participant")
        return ids

    def _has_access(self, participants: Iterable[Participant], outer_k: int, inner_k: int) -> bool:
        ids = self._validate_identifiers(participants)
        counts = [0] * self.outer.n
        for i, _ in ids:
            counts[i] += 1
        return sum(count >= inner_k for count in counts) >= outer_k

    def is_authorized(self, participants: Iterable[Participant]) -> bool:
        """Exact block access rule for original secret reconstruction."""
        return self._has_access(participants, self.outer.k, self.inner.k)

    def is_private(self, participants: Iterable[Participant]) -> bool:
        """Whether the coalition is perfectly private in the honest sharing model.

        This follows from the two-level Shamir construction, not from a failed
        interpolation attempt. See the proof in README.
        """
        return not self.is_authorized(participants)

    def validate_shares(self, shares: Iterable[CompositeShare]) -> tuple[CompositeShare, ...]:
        """Check leaf identities and normalize values over the common field."""
        result = tuple(shares)
        if any(not isinstance(s, CompositeShare) for s in result):
            raise TypeError("expected CompositeShare values")
        self._validate_identifiers(s.identifier for s in result)
        return tuple(
            CompositeShare(s.block, s.participant, self.outer.field.normalize(s.value))
            for s in result
        )

    def share(self, secret: int, rng: RandomSource | None = None) -> tuple[CompositeShare, ...]:
        """Independently re-share each outer evaluation using fresh inner randomness."""
        outer_shares = self.outer.share(secret, rng)
        leaves = []
        for i, (_, value) in enumerate(outer_shares):
            leaves.extend(
                CompositeShare(i, j, y) for j, (_, y) in enumerate(self.inner.share(value, rng))
            )
        return tuple(leaves)

    def _reconstruct(self, shares: Iterable[CompositeShare], *, product: bool) -> int:
        leaves = self.validate_shares(shares)
        outer_k = 2 * self.outer.t + 1 if product else self.outer.k
        inner_k = 2 * self.inner.t + 1 if product else self.inner.k
        if not self._has_access((s.identifier for s in leaves), outer_k, inner_k):
            raise ValueError("insufficient qualified blocks for reconstruction")
        blocks: list[list[tuple[int, int]]] = [[] for _ in range(self.outer.n)]
        for leaf in leaves:
            blocks[leaf.block].append((self.inner.points[leaf.participant], leaf.value))
        outer_values = []
        for i, observations in enumerate(blocks):
            if len(observations) >= inner_k:
                value = (
                    reconstruct_product(self.inner, observations)
                    if product
                    else self.inner.reconstruct(observations)
                )
                outer_values.append((self.outer.points[i], value))
        return (
            reconstruct_product(self.outer, outer_values)
            if product
            else self.outer.reconstruct(outer_values)
        )

    def reconstruct(self, shares: Iterable[CompositeShare]) -> int:
        """Recover inner constants in qualified blocks, then the outer constant."""
        return self._reconstruct(shares, product=False)

    def is_multiplicative(self) -> bool:
        """Sufficient two-level condition: n_o >= 2t_o+1 and n_i >= 2t_i+1.

        Refers to the implemented nested linear product reconstruction only.
        """
        return is_multiplicative(self.outer) and is_multiplicative(self.inner)

    def can_reconstruct_product(self, participants: Iterable[Participant]) -> bool:
        """Whether >= 2t_o+1 blocks each contain >= 2t_i+1 product observations."""
        return self._has_access(participants, 2 * self.outer.t + 1, 2 * self.inner.t + 1)

    def multiply_shares(
        self, left: Iterable[CompositeShare], right: Iterable[CompositeShare]
    ) -> tuple[CompositeShare, ...]:
        """Leafwise products, with participant identities matched regardless of order."""
        a, b = self.validate_shares(left), self.validate_shares(right)
        right_by_id = {s.identifier: s.value for s in b}
        if {s.identifier for s in a} != set(right_by_id):
            raise ValueError("inputs must refer to the same composite participants")
        return tuple(
            CompositeShare(
                s.block, s.participant, self.outer.field.mul(s.value, right_by_id[s.identifier])
            )
            for s in a
        )

    def reconstruct_product(self, shares: Iterable[CompositeShare]) -> int:
        """Recover inner product constants, then the outer product constant.

        Uses only leaf product shares; there is no degree reduction or MPC protocol.
        """
        return self._reconstruct(shares, product=True)
