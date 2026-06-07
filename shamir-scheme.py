from secrets import randbelow
from typing import List, Tuple

# Простое поле GF(257): удобно для байтов, так как байт = 0..255
P = 257

Share = Tuple[int, List[int]]  # (x, список y-значений для каждого байта)


def _eval_polynomial(coeffs: List[int], x: int) -> int:
    """
    Вычисляет значение многочлена в точке x по модулю P.
    coeffs[0] — свободный член, то есть байт секрета.
    """
    result = 0
    power = 1

    for c in coeffs:
        result = (result + c * power) % P
        power = (power * x) % P

    return result


def split_secret(secret: bytes, threshold: int, shares_count: int) -> List[Share]:
    """
    Делит секрет на shares_count долей.
    Для восстановления нужно минимум threshold долей.

    secret: секрет в виде bytes
    threshold: минимальное число долей для восстановления
    shares_count: общее число долей
    """
    if threshold < 2:
        raise ValueError("threshold должен быть >= 2")

    if shares_count < threshold:
        raise ValueError("shares_count должен быть >= threshold")

    if shares_count > 256:
        raise ValueError("Максимум 256 долей для поля GF(257)")

    shares: List[Share] = [(x, []) for x in range(1, shares_count + 1)]

    for byte in secret:
        # Многочлен степени threshold - 1
        # f(0) = byte
        coeffs = [byte] + [randbelow(P) for _ in range(threshold - 1)]

        for i, (x, ys) in enumerate(shares):
            y = _eval_polynomial(coeffs, x)
            ys.append(y)
            shares[i] = (x, ys)

    return shares


def _lagrange_interpolate_at_zero(points: List[Tuple[int, int]]) -> int:
    """
    Восстанавливает f(0) по точкам с помощью интерполяции Лагранжа.
    """
    total = 0

    for i, (xi, yi) in enumerate(points):
        numerator = 1
        denominator = 1

        for j, (xj, _) in enumerate(points):
            if i == j:
                continue

            numerator = (numerator * (-xj)) % P
            denominator = (denominator * (xi - xj)) % P

        inv_denominator = pow(denominator, -1, P)
        total = (total + yi * numerator * inv_denominator) % P

    return total


def recover_secret(shares: List[Share]) -> bytes:
    """
    Восстанавливает секрет из переданных долей.
    Долей должно быть не меньше threshold, который использовался при split_secret.
    """
    if not shares:
        raise ValueError("Нужна хотя бы одна доля")

    secret_length = len(shares[0][1])

    for _, ys in shares:
        if len(ys) != secret_length:
            raise ValueError("Все доли должны иметь одинаковую длину")

    recovered = bytearray()

    for byte_index in range(secret_length):
        points = [(x, ys[byte_index]) for x, ys in shares]
        value = _lagrange_interpolate_at_zero(points)

        if value > 255:
            raise ValueError("Ошибка восстановления: получено значение вне диапазона байта")

        recovered.append(value)

    return bytes(recovered)


secret = b"my super secret password"

shares = split_secret(secret, threshold=3, shares_count=5)

print("Доли:")
for share in shares:
    print(share)

restored = recover_secret([shares[0], shares[2], shares[4]])

print(restored)
print(restored == secret)