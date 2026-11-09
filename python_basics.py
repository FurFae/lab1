"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(1 for ch in text.lower() if ch in "aeiou")


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        while number:
            number, digit = divmod(number, 10)
            product *= digit
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Векторы должны иметь одинаковую длину")
    if len(predicted) == 0:
        raise ValueError("Векторы не должны быть пустыми")
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = {}
    d = 2
    while d * d <= number:
        while number % d == 0:
            factors[d] = factors.get(d, 0) + 1
            number //= d
        d += 1
    if number > 1:
        factors[number] = factors.get(number, 0) + 1
    parts = []
    for prime in sorted(factors):
        power = factors[prime]
        parts.append(f"({prime}**{power})" if power > 1 else f"({prime})")
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 0
    total = 0
    while total < cube_count:
        k += 1
        total += k * k
        if total == cube_count:
            return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(ch) for ch in str(number)]
    n = len(digits)
    left = digits[: n // 2] if n % 2 == 0 else digits[: n // 2]
    right = digits[n // 2 :] if n % 2 == 0 else digits[n // 2 + 1 :]
    return sum(left) == sum(right)
