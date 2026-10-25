"""Run the test suite and produce reproducible research-oriented reports.

The module is both a small pytest plugin and a command-line entry point.  It uses
only pytest (already a development dependency) and the Python standard library.
"""

from __future__ import annotations

import csv
import json
import platform
import shlex
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

SECTION_NAMES = {
    "test_field": "Конечное поле",
    "test_polynomial": "Многочлены и интерполяция",
    "test_shamir": "Схема Шамира",
    "test_reed_solomon": "Коды Рида—Соломона",
    "test_code_square": "Квадрат кода",
    "test_multiplication": "Мультипликативность",
    "test_concatenation": "Иерархическая композиция",
}

# Descriptions are deliberately tied to mathematical properties rather than to
# implementation details.  Unknown future tests receive a readable fallback.
PROPERTY_NAMES = {
    "test_field_axioms": "Аксиомы поля GF(p)",
    "test_composite_moduli": "Отклонение составных модулей",
    "test_noninteger_moduli": "Отклонение нецелых модулей",
    "test_representatives_and_errors": "Нормализация и ошибки арифметики",
    "test_polynomial_identities_and_interpolation": "Тождества и интерполяция",
    "test_zero_constant_and_normalization": "Канонический нулевой многочлен",
    "test_invalid_polynomial_inputs": "Проверка входов многочленов",
    "test_all_reconstruction_subsets": "Восстановление по всем минимальным коалициям",
    "test_custom_points_and_uniform_zero_leading": "Пользовательские точки и нулевые коэффициенты",
    "test_invalid_parameters": "Проверка параметров схемы",
    "test_invalid_shares": "Отклонение некорректных долей",
    "test_exact_privacy": "Точная приватность малых схем",
    "test_statistical_privacy_smoke": "Статистическая проверка приватности",
    "test_encoding_linearity_generator_and_decoding": "Линейность, ранг и декодирование RS",
    "test_distance_exhaustively": "Минимальное расстояние RS полным перебором",
    "test_invalid_code": "Проверка параметров RS-кода",
    "test_invalid_encoding_decoding_and_budget": "Ошибки кодирования, декодирования и лимитов",
    "test_rs_square_computed_spaces": "Совпадение квадрата кода с ожидаемым RS-пространством",
    "test_exhaustive_square_closure_and_distance": "Замкнутость и расстояние квадрата кода",
    "test_gaussian_elimination_and_generic_square": "Гауссово исключение и общий квадрат кода",
    "test_matrix_errors_and_enumeration_guard": "Ошибки матриц и ограничение перебора",
    "test_products_all_subsets_and_weights": "Восстановление произведений и веса Лагранжа",
    "test_strong_multiplication_by_all_deletions": "Сильная мультипликативность при всех удалениях",
    "test_insufficiency_is_mathematical_not_only_api": "Недостаточность числа долей для произведения",
    "test_reconstruction_depends_only_on_product_shares": "Независимость от исходных разделений",
    "test_invalid_product_inputs": "Проверка входов умножения",
    "test_exhaustive_access_and_reconstruction": "Полная структура доступа и восстановление",
    "test_same_cardinality_does_not_determine_access": "Непороговый характер структуры доступа",
    "test_exact_privacy_of_all_unauthorized_sets": "Точная приватность всех неавторизованных множеств",
    "test_block_isolation_and_custom_evaluation_points": "Изоляция блоков и пользовательские точки",
    "test_nested_product_reconstruction": "Вложенное восстановление произведений",
    "test_insufficient_product_parameters": "Недостаточные параметры композиции",
    "test_invalid_identifiers": "Отклонение неверных идентификаторов",
    "test_invalid_structure_and_fields": "Проверка структуры и совместимости полей",
}

STATUS_NAMES = {
    "passed": "пройдено",
    "failed": "ошибка",
    "skipped": "пропущено",
    "xfailed": "ожидаемая ошибка",
    "xpassed": "неожиданно пройдено",
    "pending": "не выполнено",
}

CSV_FIELDS = [
    "section",
    "property",
    "module",
    "test",
    "parameters",
    "status",
    "duration_seconds",
]

_REPORT_COMMAND: str | None = None


def _safe_value(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _display_name(function_name: str) -> str:
    if function_name in PROPERTY_NAMES:
        return PROPERTY_NAMES[function_name]
    return function_name.removeprefix("test_").replace("_", " ")


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _platform_name() -> str:
    system = platform.system()
    return "macOS" if system == "Darwin" else system


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("research report")
    group.addoption(
        "--research-report",
        metavar="DIR",
        help="write CSV and Markdown test reports to DIR",
    )


def pytest_configure(config: pytest.Config) -> None:
    output = config.getoption("--research-report")
    if output:
        config._research_report_state = {  # type: ignore[attr-defined]
            "output": Path(output),
            "cases": {},
            "started": datetime.now(timezone.utc),
        }


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    state = getattr(config, "_research_report_state", None)
    if state is None:
        return
    for item in items:
        module = item.path.stem
        function_name = getattr(item, "originalname", None) or item.name.split("[")[0]
        callspec = getattr(item, "callspec", None)
        parameters = {}
        if callspec is not None:
            parameters = {key: _safe_value(value) for key, value in callspec.params.items()}
        state["cases"][item.nodeid] = {
            "section": SECTION_NAMES.get(module, module),
            "property": _display_name(function_name),
            "module": module,
            "test": item.name,
            "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
            "status": "pending",
            "duration_seconds": 0.0,
        }


def pytest_deselected(items: list[pytest.Item]) -> None:
    """Exclude cases filtered by ``-k``, ``-m`` or explicit node selection."""
    for item in items:
        state = getattr(item.config, "_research_report_state", None)
        if state is not None:
            state["cases"].pop(item.nodeid, None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report = outcome.get_result()
    state = getattr(item.config, "_research_report_state", None)
    if state is None:
        return

    case = state["cases"][item.nodeid]
    case["duration_seconds"] += report.duration
    if report.failed:
        case["status"] = "failed"
    elif report.skipped and case["status"] != "failed":
        case["status"] = "xfailed" if hasattr(report, "wasxfail") else "skipped"
    elif report.when == "call" and report.passed:
        case["status"] = "xpassed" if hasattr(report, "wasxfail") else "passed"


def _write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row["duration_seconds"] = f"{case['duration_seconds']:.6f}"
            writer.writerow({key: row[key] for key in CSV_FIELDS})


def _write_markdown(
    path: Path,
    cases: list[dict[str, Any]],
    started: datetime,
    finished: datetime,
    command: str,
) -> None:
    counts = Counter(case["status"] for case in cases)
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    properties: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        sections[case["section"]].append(case)
        properties[(case["section"], case["property"])].append(case)

    lines = [
        "# Результаты автоматизированного тестирования",
        "",
        f"- Начало прогона (UTC): `{started.isoformat(timespec='seconds')}`",
        f"- Окончание прогона (UTC): `{finished.isoformat(timespec='seconds')}`",
        f"- Python: `{platform.python_version()}`",
        f"- Платформа: `{_platform_name()}`",
        f"- Команда: `{command}`",
        "",
        "## Общий результат",
        "",
        "| Всего | Пройдено | Ошибки | Пропущено | Время тестов, с |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {len(cases)} | {counts['passed']} | {counts['failed'] + counts['xpassed']} "
            f"| {counts['skipped'] + counts['xfailed']} "
            f"| {sum(case['duration_seconds'] for case in cases):.3f} |"
        ),
        "",
        "## Результаты по разделам",
        "",
        "| Раздел | Проверок | Пройдено | Ошибки | Время, с |",
        "|---|---:|---:|---:|---:|",
    ]
    for section, group in sections.items():
        group_counts = Counter(case["status"] for case in group)
        errors = group_counts["failed"] + group_counts["xpassed"]
        duration = sum(case["duration_seconds"] for case in group)
        lines.append(
            f"| {_escape_cell(section)} | {len(group)} | {group_counts['passed']} | {errors} | {duration:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Матрица проверяемых свойств",
            "",
            "| Раздел | Проверяемое свойство | Наборов параметров | Результат | Время, с |",
            "|---|---|---:|---|---:|",
        ]
    )
    for (section, property_name), group in properties.items():
        statuses = Counter(case["status"] for case in group)
        if statuses["failed"] or statuses["xpassed"]:
            result = "ошибка"
        elif statuses["pending"]:
            result = "не выполнено"
        elif statuses["skipped"] or statuses["xfailed"]:
            result = "частично пропущено"
        else:
            result = "пройдено"
        duration = sum(case["duration_seconds"] for case in group)
        lines.append(
            f"| {_escape_cell(section)} | {_escape_cell(property_name)} | {len(group)} "
            f"| {result} | {duration:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Наиболее длительные проверки",
            "",
            "| Проверка | Параметры | Результат | Время, с |",
            "|---|---|---|---:|",
        ]
    )
    for case in sorted(cases, key=lambda value: value["duration_seconds"], reverse=True)[:15]:
        lines.append(
            f"| `{_escape_cell(case['module'])}::{_escape_cell(case['test'])}` "
            f"| `{_escape_cell(case['parameters'])}` | {STATUS_NAMES[case['status']]} "
            f"| {case['duration_seconds']:.6f} |"
        )

    lines.extend(
        [
            "",
            "Полная детализация каждого параметризованного случая находится в "
            "[`test-results.csv`](test-results.csv). Время в таблицах является временем "
            "тестов по данным pytest и не предназначено для сравнения производительности "
            "алгоритмов.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    state = getattr(session.config, "_research_report_state", None)
    if state is None:
        return
    output: Path = state["output"]
    output.mkdir(parents=True, exist_ok=True)
    cases = list(state["cases"].values())
    command = _REPORT_COMMAND or shlex.join(
        [sys.executable, "-m", "pytest", "-p", "research_report", *session.config.invocation_params.args]
    )
    _write_csv(output / "test-results.csv", cases)
    _write_markdown(
        output / "test-summary.md",
        cases,
        state["started"],
        datetime.now(timezone.utc),
        command,
    )


def _option_value(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
    return None


def main(arguments: list[str] | None = None) -> int:
    global _REPORT_COMMAND

    args = list(sys.argv[1:] if arguments is None else arguments)
    _REPORT_COMMAND = shlex.join(["python", "research_report.py", *args])
    output = _option_value(args, "--research-report")
    if output is None:
        output = "results"
        args.extend(["--research-report", output])
    Path(output).mkdir(parents=True, exist_ok=True)
    return int(pytest.main(args, plugins=[sys.modules[__name__]]))


if __name__ == "__main__":
    raise SystemExit(main())
