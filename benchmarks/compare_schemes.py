"""Compare Shamir and two-level Shamir with matched length and privacy."""

import argparse
import csv
import platform
from collections import defaultdict
from functools import partial
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter_ns

from code_square import CodeSquare
from concatenation import HierarchicalShamirComposition
from field import PrimeField
from linear_algebra import rank
from multiplication import multiply_shares, reconstruct_product
from reed_solomon import ReedSolomonCode
from shamir import ShamirScheme

FIELD = PrimeField(31)
# Outer (participants, privacy), inner (participants, privacy).
CASES = [(3, 1, 3, 1), (4, 1, 4, 1), (4, 1, 5, 2), (5, 2, 4, 1)]
OPERATIONS = ("share", "reconstruct", "multiply", "reconstruct_product")


class Coefficients:
    """Inject basis vectors into the actual share-generation algorithm."""

    def __init__(self, values):
        self.values = iter(values)

    def randrange(self, stop):
        return next(self.values)


def composite_generator(scheme):
    random_count = scheme.outer.t + scheme.outer.n * scheme.inner.t
    rows = []
    for index in range(random_count + 1):
        rng = Coefficients(int(j == index) for j in range(1, random_count + 1))
        shares = scheme.share(int(index == 0), rng)
        rows.append([share.value for share in shares])
    return rows


def composite_distance(outer, inner, power):
    # Positive inner degree, and scaled degrees below block lengths in every case.
    outer_d = outer.n - power * outer.t
    inner_d = inner.n - power * inner.t
    return min(inner_d + 1, outer_d * inner_d)


def prepare(case, seed):
    no, to, ni, ti = case
    outer, inner = ShamirScheme(FIELD, no, to), ShamirScheme(FIELD, ni, ti)
    comp = HierarchicalShamirComposition(outer, inner)
    access = comp.access_parameters()
    n = comp.total_shares
    flat = ShamirScheme(FIELD, n, access["guaranteed_privacy_size"])
    code = ReedSolomonCode(FIELD, n, flat.k, flat.points)
    label = f"({no},{to}) × ({ni},{ti})"
    parameters = []
    tasks = {}
    for name, scheme in [("RS", flat), ("RS ∘ RS", comp)]:
        rng = Random(seed)
        a, b = scheme.share(7, rng), scheme.share(11, rng)
        if scheme is flat:
            multiply = partial(multiply_shares, flat)
            recover_product = partial(reconstruct_product, flat)
            secret_set = a[: flat.k]
            product_set = multiply(a, b)[: 2 * flat.t + 1]
            guaranteed = flat.k
            generator = code.generator_matrix()
            d, square_d = code.d, n - 2 * flat.t
            expected_dimensions = (flat.k, 2 * flat.t + 1)
            rule = str(flat.k)
        else:
            multiply = comp.multiply_shares
            recover_product = comp.reconstruct_product
            secret_set = tuple(s for s in a if s.block < outer.k and s.participant < inner.k)
            product_set = tuple(
                s for s in multiply(a, b) if s.block <= 2 * to and s.participant <= 2 * ti
            )
            guaranteed = access["guaranteed_reconstruction_size"]
            generator = composite_generator(comp)
            d = composite_distance(outer, inner, 1)
            square_d = composite_distance(outer, inner, 2)
            expected_dimensions = (outer.k + no * ti, 2 * to + 1 + no * 2 * ti)
            rule = f"{outer.k} × {inner.k}"
        dimensions = rank(FIELD, generator), CodeSquare(FIELD, generator).dimension
        if dimensions != expected_dimensions:
            raise ValueError(f"Unexpected code ranks for {label}, {name}: {dimensions}")
        if scheme.reconstruct(secret_set) != 7 or recover_product(product_set) != 77 % FIELD.p:
            raise ValueError(f"Failed reconstruction for {label}, {name}")
        parameters.append(
            dict(
                case=label,
                scheme=name,
                n=n,
                private=flat.t,
                rule=rule,
                min_secret=len(secret_set),
                any_secret=guaranteed,
                product_inputs=len(product_set),
                share_bits=comp.share_size_bits,
                dim=dimensions[0],
                d=d,
                square_dim=dimensions[1],
                square_d=square_d,
            )
        )
        tasks[name] = (
            partial(scheme.share, 7, rng),
            partial(scheme.reconstruct, secret_set),
            partial(multiply, a, b),
            partial(recover_product, product_set),
        )
    return parameters, tasks


def sample(call, loops):
    start = perf_counter_ns()
    for _ in range(loops):
        call()
    return (perf_counter_ns() - start) / loops / 1000


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--loops", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.rounds < 1 or args.loops < 1:
        parser.error("rounds and loops must be positive")
    parameters, observations = [], []
    timings = defaultdict(list)
    for index, case in enumerate(CASES):
        rows, tasks = prepare(case, args.seed + index)
        parameters.extend(rows)
        names = list(tasks)
        for op, operation in enumerate(OPERATIONS):
            for name in names:
                sample(tasks[name][op], args.loops)
            for batch in range(args.rounds):
                for name in names if batch % 2 == 0 else names[::-1]:
                    elapsed = sample(tasks[name][op], args.loops)
                    key = (rows[0]["case"], name, operation)
                    timings[key].append(elapsed)
                    observations.append(
                        dict(
                            case=key[0],
                            scheme=name,
                            operation=operation,
                            batch=batch + 1,
                            us_per_call=elapsed,
                        )
                    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "parameters.csv", parameters)
    write_csv(output / "timings.csv", observations)
    report = [
        "# Сравнение RS и двухуровневой схемы Шамира",
        "",
        f"Python {platform.python_version()}, {platform.system()} {platform.release()}, "
        f"{platform.machine()}. Поле GF({FIELD.p}), начальный seed {args.seed}.",
        "",
        "## Параметры",
        "",
        "В первом столбце указаны (n, t) внешнего и внутреннего уровней. В каждой паре "
        "совпадают поле, число участников N и гарантированная приватность t. Полезная "
        "нагрузка одной доли при фиксированной упаковке составляет 5 бит у обеих схем.",
        "",
        "Правило доступа показывает, сколько долей требуется для RS, либо сколько блоков "
        "должны предоставить указанное число долей для композиции. Столбец «Любые» даёт "
        "число долей, достаточное при любом распределении. Для произведения указано число "
        "входов выбранного алгоритма при минимальной подходящей конфигурации.",
        "",
        "| Уровни | Схема | N | t | Правило доступа | Любые | Для произведения | "
        "dim C | d(C) | dim C² | d(C²) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in parameters:
        report.append(
            "| {case} | {scheme} | {n} | {private} | {rule} | {any_secret} | "
            "{product_inputs} | {dim} | {d} | {square_dim} | {square_d} |".format(**row)
        )
    report += [
        "",
        "C содержит только доли участников. Размерности вычислены по рангам матриц, "
        "для композиции строки получены подачей единичных векторов секрета и случайных "
        "коэффициентов на вход алгоритму выдачи долей. Расстояния рассчитаны аналитически.",
        "",
        "Для выбранных параметров код композиции при квадрате сохраняет блочную форму "
        "с удвоенными степенями. Произведения внешних мономов порождают внешние степени "
        "от 0 до 2tₒ. В каждом блоке произведения с единицей и попарные произведения "
        "локальных мономов дают все степени от 1 до 2tᵢ. Это даёт полное пространство "
        "с указанными степенями. Для степени m = 1 или 2 положим a = nₒ − mtₒ и "
        "b = nᵢ − mtᵢ. При ненулевом внешнем слове вес не меньше ab, при нулевом "
        "внешнем слове ненулевой блок имеет дополнительный корень в нуле и вес не меньше "
        "b + 1. Обе границы достижимы выбором многочленов с нужным числом корней "
        "среди точек вычисления. Поэтому расстояние равно min(ab, b + 1).",
        "",
        "## Время операций",
        "",
        f"Медиана {args.rounds} серий по {args.loops} вызовов, мкс на вызов. Перед каждой "
        "операцией выполнена одна прогревочная серия. Порядок схем чередуется. Генерация "
        "использует random.Random для воспроизводимости. Выдача и умножение обрабатывают "
        "все N долей. Восстановление получает заранее выбранный минимальный подходящий "
        "набор. Для обеих схем проверено восстановление секрета и произведения. "
        "Сырые средние по сериям сохранены в timings.csv.",
        "",
        "| Уровни | Схема | Выдача | Восстановление | Локальное умножение всех долей | "
        "Восстановление произведения |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in parameters:
        values = [median(timings[row["case"], row["scheme"], op]) for op in OPERATIONS]
        report.append(
            f"| {row['case']} | {row['scheme']} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + " |"
        )
    (output / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
