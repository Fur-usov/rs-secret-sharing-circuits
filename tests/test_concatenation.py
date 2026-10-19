from collections import Counter
from itertools import combinations, product

import pytest

from concatenation import CompositeShare, HierarchicalShamirComposition
from field import PrimeField
from shamir import ShamirScheme


def make_composition(p=5, no=3, to=1, ni=2, ti=1):
    field = PrimeField(p)
    return HierarchicalShamirComposition(ShamirScheme(field, no, to), ShamirScheme(field, ni, ti))


def subsets(items):
    for size in range(len(items) + 1):
        yield from combinations(items, size)


@pytest.mark.parametrize(
    "p,no,to,ni,ti",
    [(5, 3, 1, 2, 1), (5, 2, 0, 3, 1), (5, 3, 1, 2, 0), (3, 2, 1, 2, 1), (3, 1, 0, 1, 0)],
)
def test_exhaustive_access_and_reconstruction(p, no, to, ni, ti, rng):
    composition = make_composition(p, no, to, ni, ti)
    authorized_sizes, private_sizes = [], []
    for secret in range(p):
        shares = composition.share(secret, rng)
        assert len(shares) == composition.total_shares == no * ni
        assert tuple(s.identifier for s in shares) == composition.participants
        for subset in subsets(shares):
            ids = [s.identifier for s in subset]
            expected = sum(sum(s.block == i for s in subset) >= ti + 1 for i in range(no)) >= to + 1
            assert composition.is_authorized(ids) == expected
            assert composition.is_private(ids) != expected
            if expected:
                assert composition.reconstruct(reversed(subset)) == secret
                authorized_sizes.append(len(subset))
            else:
                with pytest.raises(ValueError):
                    composition.reconstruct(subset)
                private_sizes.append(len(subset))
    params = composition.access_parameters()
    assert min(authorized_sizes) == params["minimum_authorized_size"]
    assert max(private_sizes) == params["maximum_unauthorized_size"]
    assert params["guaranteed_privacy_size"] + 1 == min(authorized_sizes)
    assert params["guaranteed_reconstruction_size"] - 1 == max(private_sizes)


def test_same_cardinality_does_not_determine_access():
    c = make_composition()
    assert c.is_authorized([(0, 0), (0, 1), (1, 0), (1, 1)])
    assert not c.is_authorized([(0, 0), (0, 1), (1, 0), (2, 0)])


def test_exact_privacy_of_all_unauthorized_sets():
    # Enumerate actual share() random streams, including zeros, at both levels.
    c = make_composition()
    p = c.outer.field.p
    private = [ids for ids in subsets(c.participants) if c.is_private(ids)]
    positions = [[c.participants.index(identifier) for identifier in ids] for ids in private]

    class StreamRng:
        def __init__(self, values):
            self.values = iter(values)

        def randrange(self, stop):
            value = next(self.values)
            assert 0 <= value < stop
            return value

    baseline = None
    random_count = c.outer.t + c.outer.n * c.inner.t
    for secret in range(p):
        distributions = [Counter() for _ in private]
        for coefficients in product(range(p), repeat=random_count):
            leaves = c.share(secret, StreamRng(coefficients))
            for distribution, indices in zip(distributions, positions):
                distribution[tuple(leaves[j].value for j in indices)] += 1
        if baseline is None:
            baseline = distributions
        else:
            assert distributions == baseline


def test_block_isolation_and_custom_evaluation_points(rng):
    f = PrimeField(11)
    outer = ShamirScheme(f, 3, 1, [8, 3, 6])
    inner = ShamirScheme(f, 3, 1, [7, 1, 4])
    c = HierarchicalShamirComposition(outer, inner)

    class RecordingRng:
        def __init__(self):
            self.values = []

        def randrange(self, stop):
            value = rng.randrange(stop)
            self.values.append(value)
            return value

    recording = RecordingRng()
    shares = c.share(4, recording)
    a, *inner_slopes = recording.values
    assert len(inner_slopes) == outer.n
    for i in range(outer.n):
        block = [(inner.points[s.participant], s.value) for s in shares if s.block == i]
        expected = (4 + a * outer.points[i]) % f.p
        assert inner.reconstruct(block) == expected
        assert all(y == (expected + inner_slopes[i] * x) % f.p for x, y in block)
    assert c.reconstruct(shares) == 4
    assert c.share_size_bits == 4
    assert make_composition(2, 1, 0, 1, 0).share_size_bits == 1


@pytest.mark.parametrize("to,ti", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_nested_product_reconstruction(to, ti, rng):
    c = make_composition(7, 4, to, 4, ti)
    assert c.is_multiplicative()
    for a, b in product(range(7), repeat=2):
        local = c.multiply_shares(c.share(a, rng), reversed(c.share(b, rng)))
        assert c.reconstruct_product(local) == a * b % 7
        # Exercise different minimal qualified block/leaf selections.
        for blocks in combinations(range(4), 2 * to + 1):
            leaves = tuple(s for s in local if s.block in blocks and s.participant < 2 * ti + 1)
            assert c.can_reconstruct_product(s.identifier for s in leaves)
            assert c.reconstruct_product(leaves) == a * b % 7
            assert not c.can_reconstruct_product(s.identifier for s in leaves[:-1])
            with pytest.raises(ValueError):
                c.reconstruct_product(leaves[:-1])


def test_insufficient_product_parameters(rng):
    for c in [make_composition(5, 2, 1, 3, 1), make_composition(5, 3, 1, 2, 1)]:
        assert not c.is_multiplicative()
        assert not c.can_reconstruct_product(c.participants)
        products = c.multiply_shares(c.share(1, rng), c.share(2, rng))
        with pytest.raises(ValueError):
            c.reconstruct_product(products)


@pytest.mark.parametrize("ids", [[(-1, 0)], [(3, 0)], [(0, 2)], [(0, 0), (0, 0)]])
def test_invalid_identifiers(ids):
    c = make_composition()
    with pytest.raises(ValueError):
        c.is_authorized(ids)
    with pytest.raises(ValueError):
        c.reconstruct([CompositeShare(i, j, 0) for i, j in ids])


def test_invalid_structure_and_fields(rng):
    c = make_composition()
    with pytest.raises(ValueError):
        HierarchicalShamirComposition(c.outer, ShamirScheme(PrimeField(7), 2, 1))
    with pytest.raises(TypeError):
        c.validate_shares([(0, 0, 1)])
    with pytest.raises(TypeError):
        c.is_authorized([(True, 0)])
    with pytest.raises(ValueError):
        c.multiply_shares(c.share(2, rng), c.share(3, rng)[:-1])
