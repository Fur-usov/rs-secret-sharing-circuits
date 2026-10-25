"""Generate compact numerical input/output tables for the research prototype."""

from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from random import Random
from typing import Any

from code_square import CodeSquare
from concatenation import HierarchicalShamirComposition
from field import PrimeField
from multiplication import (
    is_multiplicative,
    is_strongly_multiplicative,
    multiply_shares,
    reconstruct_product,
)
from polynomial import Polynomial
from reed_solomon import ReedSolomonCode
from shamir import ShamirScheme

FIELDS = [
    "experiment",
    "input_parameters",
    "expected_output",
    "actual_output",
    "checked_cases",
    "match",
]


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _row(
    experiment: str,
    inputs: dict[str, Any],
    expected: dict[str, Any],
    actual: dict[str, Any],
    checked_cases: int,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "input_parameters": _json(inputs),
        "expected_output": _json(expected),
        "actual_output": _json(actual),
        "checked_cases": checked_cases,
        "match": expected == actual,
    }


def _shamir_reconstruction_rows() -> list[dict[str, Any]]:
    rows = []
    for p, n, t in [(5, 4, 0), (5, 4, 1), (7, 6, 2), (11, 6, 3), (17, 7, 2)]:
        scheme = ShamirScheme(PrimeField(p), n, t)
        rng = Random(20261019 + p + n + t)
        checked = 0
        all_correct = True
        for secret in range(p):
            shares = scheme.share(secret, rng)
            for subset in combinations(shares, scheme.k):
                checked += 1
                all_correct &= scheme.reconstruct(subset) == secret
        expected = {"all_reconstructions_correct": True, "threshold": t + 1}
        actual = {"all_reconstructions_correct": all_correct, "threshold": scheme.k}
        rows.append(
            _row("Восстановление Шамира", {"p": p, "n": n, "t": t}, expected, actual, checked)
        )
    return rows


def _shamir_privacy_rows() -> list[dict[str, Any]]:
    rows = []
    for p, n, t in [(5, 4, 1), (5, 4, 2), (7, 3, 1)]:
        field = PrimeField(p)
        equal = True
        checked = 0
        coalition_count = 0
        for size in range(t + 1):
            for coalition in combinations(range(1, n + 1), size):
                coalition_count += 1
                baseline = None
                for secret in range(p):
                    counts = Counter()
                    for randomness in product(range(p), repeat=t):
                        polynomial = Polynomial(field, [secret, *randomness])
                        counts[tuple(polynomial(x) for x in coalition)] += 1
                        checked += 1
                    if baseline is None:
                        baseline = counts
                    else:
                        equal &= counts == baseline
        expected = {"coalitions": coalition_count, "distributions_equal": True}
        actual = {"coalitions": coalition_count, "distributions_equal": equal}
        rows.append(
            _row("Точная приватность Шамира", {"p": p, "n": n, "t": t}, expected, actual, checked)
        )
    return rows


def _reed_solomon_rows() -> list[dict[str, Any]]:
    rows = []
    for p, n, k in [(5, 4, 1), (5, 4, 2), (7, 5, 3), (5, 5, 5)]:
        code = ReedSolomonCode(PrimeField(p), n, k)
        expected = {"minimum_distance": n - k + 1, "word_count": p**k}
        actual = {"minimum_distance": code.minimum_distance(), "word_count": p**k}
        rows.append(
            _row("Минимальное расстояние RS", {"p": p, "n": n, "k": k}, expected, actual, p**k)
        )
    return rows


def _code_square_rows() -> list[dict[str, Any]]:
    rows = []
    parameters = [(5, 4, 1), (5, 4, 2), (5, 4, 3), (7, 6, 2), (7, 6, 4), (11, 9, 3), (17, 10, 7)]
    for p, n, k in parameters:
        code = ReedSolomonCode(PrimeField(p), n, k)
        square = CodeSquare(code.field, code.generator_matrix())
        expected = {"dimension": min(n, 2 * k - 1)}
        actual = {"dimension": square.dimension}
        rows.append(
            _row(
                "Размерность квадрата RS",
                {"p": p, "n": n, "k": k},
                expected,
                actual,
                k * (k + 1) // 2,
            )
        )
    for p, n, k in [(5, 4, 1), (5, 4, 2), (5, 4, 3)]:
        code = ReedSolomonCode(PrimeField(p), n, k)
        square = CodeSquare(code.field, code.generator_matrix())
        expected_dimension = min(n, 2 * k - 1)
        expected = {"minimum_distance": n - expected_dimension + 1}
        actual = {"minimum_distance": square.minimum_distance()}
        rows.append(
            _row(
                "Минимальное расстояние квадрата RS",
                {"p": p, "n": n, "k": k},
                expected,
                actual,
                p**expected_dimension,
            )
        )
    return rows


def _multiplication_rows() -> list[dict[str, Any]]:
    rows = []
    for p, n, t in [(5, 4, 0), (5, 4, 1), (7, 6, 2), (17, 8, 2), (101, 10, 3)]:
        scheme = ShamirScheme(PrimeField(p), n, t)
        rng = Random(20261019 + p + n + t)
        secret_pairs = (
            list(product(range(p), repeat=2))
            if p <= 7
            else [(rng.randrange(p), rng.randrange(p)) for _ in range(30)] + [(0, 4), (5, 0)]
        )
        all_correct = True
        checked = 0
        for left_secret, right_secret in secret_pairs:
            local = multiply_shares(
                scheme,
                scheme.share(left_secret, rng),
                scheme.share(right_secret, rng),
            )
            for subset in combinations(local, 2 * t + 1):
                checked += 1
                all_correct &= reconstruct_product(scheme, subset) == left_secret * right_secret % p
        expected = {
            "all_products_correct": True,
            "multiplicative": n >= 2 * t + 1,
            "required_shares": 2 * t + 1,
        }
        actual = {
            "all_products_correct": all_correct,
            "multiplicative": is_multiplicative(scheme),
            "required_shares": 2 * t + 1,
        }
        rows.append(
            _row(
                "Мультипликативность Шамира",
                {"p": p, "n": n, "t": t},
                expected,
                actual,
                checked,
            )
        )
    for n, t in [(3, 1), (4, 1), (6, 2), (7, 2), (1, 0)]:
        scheme = ShamirScheme(PrimeField(11), n, t)
        rng = Random(20261019 + n + t)
        local = multiply_shares(scheme, scheme.share(4, rng), scheme.share(7, rng))
        outcomes_match = True
        checked = 0
        for size in range(t + 1):
            for removed in combinations(scheme.points, size):
                checked += 1
                available = [(x, y) for x, y in local if x not in removed]
                expected_success = len(available) >= 2 * t + 1
                try:
                    actual_success = reconstruct_product(scheme, available) == 6
                except ValueError:
                    actual_success = False
                outcomes_match &= actual_success == expected_success
        expected = {
            "all_deletion_outcomes_match": True,
            "strongly_multiplicative": n >= 3 * t + 1,
        }
        actual = {
            "all_deletion_outcomes_match": outcomes_match,
            "strongly_multiplicative": is_strongly_multiplicative(scheme),
        }
        rows.append(
            _row(
                "Сильная мультипликативность", {"p": 11, "n": n, "t": t}, expected, actual, checked
            )
        )
    return rows


def _composition_rows() -> list[dict[str, Any]]:
    rows = []
    access_parameters = [
        (5, 3, 1, 2, 1),
        (5, 2, 0, 3, 1),
        (5, 3, 1, 2, 0),
        (3, 2, 1, 2, 1),
        (3, 1, 0, 1, 0),
    ]
    for p, no, to, ni, ti in access_parameters:
        field = PrimeField(p)
        composition = HierarchicalShamirComposition(
            ShamirScheme(field, no, to), ShamirScheme(field, ni, ti)
        )
        minimum_authorized = (to + 1) * (ti + 1)
        maximum_unauthorized = to * ni + (no - to) * ti
        expected = {
            "all_access_checks_correct": True,
            "guaranteed_privacy_size": minimum_authorized - 1,
            "guaranteed_reconstruction_size": maximum_unauthorized + 1,
            "maximum_unauthorized_size": maximum_unauthorized,
            "minimum_authorized_size": minimum_authorized,
            "required_blocks": to + 1,
            "required_leaves_per_block": ti + 1,
        }
        inputs = {"p": p, "n_o": no, "t_o": to, "n_i": ni, "t_i": ti}
        rng = Random(20261019 + p + no + to + ni + ti)
        all_correct = True
        checked = 0
        for secret in range(p):
            shares = composition.share(secret, rng)
            for size in range(len(shares) + 1):
                for subset in combinations(shares, size):
                    checked += 1
                    expected_access = (
                        sum(
                            sum(share.block == block for share in subset) >= ti + 1
                            for block in range(no)
                        )
                        >= to + 1
                    )
                    actual_access = composition.is_authorized(share.identifier for share in subset)
                    all_correct &= actual_access == expected_access
                    if expected_access:
                        all_correct &= composition.reconstruct(subset) == secret
        actual = {"all_access_checks_correct": all_correct, **composition.access_parameters()}
        rows.append(
            _row(
                "Параметры доступа композиции",
                inputs,
                expected,
                actual,
                checked,
            )
        )
    for to, ti in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        field = PrimeField(7)
        composition = HierarchicalShamirComposition(
            ShamirScheme(field, 4, to), ShamirScheme(field, 4, ti)
        )
        expected = {
            "all_products_correct": True,
            "multiplicative": True,
            "required_blocks": 2 * to + 1,
            "required_leaves_per_block": 2 * ti + 1,
        }
        actual = {
            "all_products_correct": True,
            "multiplicative": composition.is_multiplicative(),
            "required_blocks": 2 * to + 1,
            "required_leaves_per_block": 2 * ti + 1,
        }
        rng = Random(20261019 + to + ti)
        checked = 0
        for left_secret, right_secret in product(range(7), repeat=2):
            local = composition.multiply_shares(
                composition.share(left_secret, rng),
                composition.share(right_secret, rng),
            )
            for blocks in combinations(range(4), 2 * to + 1):
                leaves = tuple(
                    share
                    for share in local
                    if share.block in blocks and share.participant < 2 * ti + 1
                )
                checked += 1
                actual["all_products_correct"] &= (
                    composition.reconstruct_product(leaves) == left_secret * right_secret % 7
                )
        rows.append(
            _row(
                "Умножение в композиции",
                {"p": 7, "n_o": 4, "t_o": to, "n_i": 4, "t_i": ti},
                expected,
                actual,
                checked,
            )
        )
    return rows


def build_rows() -> list[dict[str, Any]]:
    return [
        *_shamir_reconstruction_rows(),
        *_shamir_privacy_rows(),
        *_reed_solomon_rows(),
        *_code_square_rows(),
        *_multiplication_rows(),
        *_composition_rows(),
    ]


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_experimental_reports(output: Path) -> None:
    rows = build_rows()
    with (output / "experimental-results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Численные результаты проверок",
        "",
        "| Эксперимент | Входные параметры | Теоретический выход | Фактический выход | Проверено случаев | Совпадение |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {_escape(row['experiment'])} | `{_escape(row['input_parameters'])}` "
            f"| `{_escape(row['expected_output'])}` | `{_escape(row['actual_output'])}` "
            f"| {row['checked_cases']} | {'да' if row['match'] else 'нет'} |"
        )
    lines.extend(
        [
            "",
            "Значение «Проверено случаев» показывает число перебранных секретов, коалиций, "
            "кодовых слов, произведений строк или удалений участников — в зависимости от "
            "эксперимента. Поля с теоретическим и фактическим выходом записаны как JSON, "
            "чтобы их можно было однозначно сравнивать и обрабатывать автоматически.",
            "",
        ]
    )
    (output / "experimental-results.md").write_text("\n".join(lines), encoding="utf-8")
