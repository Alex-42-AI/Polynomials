"""
Module for implementing polynomials
"""

from __future__ import annotations

from collections.abc import Iterable

from cmath import sqrt

from math import sin, cos, pi, inf

def round_if_possible(n: complex, d: int = 0):
    """
    Args:
        n: Given number
        d: Decimal place precision
    Returns:
        A more closely rounded value of n
    """

    if isinstance(n, complex):
        if not n.imag:
            return round_if_possible(n.real, d)

        return complex(round_if_possible(n.real, d), round_if_possible(n.imag, d))

    if isinstance(n, int):
        return n

    if isinstance(n, float):
        return (round(n, d) if d > 0 else int(n)) if abs(n - round(n, d)) <= 5 * 10 ** -13 else n

    raise TypeError('Numbers only!')


def cbrt(n: complex) -> complex:
    """
    Args:
        n: Given number
    Returns:
        cube root of n
    """

    if isinstance(n, (complex, float, int)):
        if n in {-1, 0, 1}:
            return n

        if isinstance(n, (int, float)):
            return round_if_possible(abs(n) ** (1 / 3) * (-1) ** (n < 0))

        return n ** (1 / 3)

    raise TypeError(f"Inappropriate type! Expected a numerical value, got {type(n).__name__} instead!")


def quadratic_solver(a: complex, b: complex, c: complex) -> dict[complex, int]:
    """
    Args:
        a: coefficient of x^2
        b: coefficient of x
        c: free constant
    Returns:
        The set of roots of ax^2+bx+c=0
    """

    if a:
        b, c = b / a, c / a
        extremal, deviation = -b / 2, sqrt(abs(b ** 2 - 4 * c)) / 2

        if not deviation:
            return {round_if_possible(extremal): 2}

        if b ** 2 < 4 * c:
            return {round_if_possible(complex(extremal, deviation)): 1,
                    round_if_possible(complex(extremal, -deviation)): 1}

        return {round_if_possible(extremal + deviation): 1, round_if_possible(extremal - deviation): 1}

    try:
        return {round_if_possible(-c / b): 1}
    except ZeroDivisionError:
        raise ZeroDivisionError('False!') if c else ValueError('Indeterminable')


def cubic_solver(a: float, b: float, c: float, d: float) -> complex:
    """
    Args:
        a: coefficient of x^3
        b: coefficient of x^2
        c: coefficient of x
        d: free constant
    Returns:
        A root of ax^3+bx^2+cx+d=0
    """

    if a:
        b, c, d = b / a, c / a, d / a
        p, q = b * c / 6 - d / 2 - (b / 3) ** 3, c / 3 - (b / 3) ** 2
        v = sqrt(p ** 2 + q ** 3)
        fst, snd = cbrt(p + v), cbrt(p - v)

        return round_if_possible(fst + snd - b / 3)

    res = quadratic_solver(b, c, d)

    return round_if_possible(set(res).pop())


def quartic_solver(a: float, b: float, c: float, d: float, e: float) -> complex:
    """
    Args:
        a: coefficient of x^4
        b: coefficient of x^3
        c: coefficient of x^2
        d: coefficient of x
        e: free constant
    Returns:
        A root of ax^4+bx^3+cx^2+dx+e=0
    """

    if a:
        b, c, d, e = b / a, c / a, d / a, e / a
        p, q, r = c - 3 * b ** 2 / 8, b ** 3 / 8 - b * c / 2 + d, e - b * d / 4 + b ** 2 * c / 16 - 3 * b ** 4 / 256

        if 4 * p * r != q ** 2:
            z0 = cubic_solver(1, -p, 4 * r, 4 * p * r - q ** 2)
        elif r:
            z0 = quadratic_solver(1, -p, 4 * r)
        elif p:
            z0 = p
        else:
            return -b / 4

        R = sqrt(z0)
        D, E = sqrt(2 * q / R - 2 * p - z0), sqrt(- 2 * q / R - 2 * p - z0)

        return round_if_possible((D + E) / 2 - b / 4)

    return cubic_solver(b, c, d, e)

class Polynomial:
    def __init__(self, vals: Iterable[float]):
        """
        Args:
            vals: A list of coefficients
        Creates a polynomial of degree len(self) - 1 (for vals == [] the polynomial
        is constant 0), where self[i - 1] is the coefficient of x^(n - i)
        """

        if isinstance(vals, (int, float)):
            vals = [vals]

        self.__value = self.remove_0s(vals)

    @staticmethod
    def remove_0s(l: Iterable[float]) -> list[float]:
        """
        Args:
            l: A list of coefficients
        Returns:
            l without leading 0s
        """

        i = 0

        try:
            while not l[i]:
                i += 1
        except IndexError:
            ...
        return list(l[i:])

    @staticmethod
    def extend_0s(l: list[float], n: int) -> list[float]:
        """
        Args:
            l: A list of coefficients
            n: Wanted length
        Returns:
            l, extended with leading 0s and with a length of n
        """

        return (n - len(l)) * [0] + l

    @staticmethod
    def polynomial_from_roots(roots: dict[float, int]) -> Polynomial:
        res = Polynomial([1])

        for k, v in roots.items():
            res *= Polynomial([1, -k]) ** v

        return res

    @property
    def value(self) -> list[float]:
        """
        Returns:
           The list of coefficients
        """

        return self.__value

    @property
    def degree(self) -> int:
        """
        Returns:
            The degree of the polynomial
        """

        if not self:
            return -inf

        return len(self) - 1

    def copy(self) -> Polynomial:
        """
        Returns:
            Identical copy of the polynomial
        """

        return Polynomial(self.value)

    def evaluate(self, x: complex) -> complex:
        """
        Args:
            x: A complex number
        Returns:
            self(x)
        """

        s = 0

        for a in self:
            s = s * x + a

        return s

    def compose(self, other: Polynomial) -> Polynomial:
        """
        Args:
            other: Another polynomial
        Returns:
            self(other)
        """

        res, n = Polynomial([]), self.degree

        for i, a in enumerate(self):
            res += a * other ** (n - i)

        return res

    def differentiate(self) -> Polynomial:
        """
        Returns:
            self'(x)
        """

        res, n = [], len(self)

        for i, a in enumerate(self.value[:-1]):
            res.append(a * (n - i - 1))

        return Polynomial(res)

    def integrate(self, c: float = 0) -> Polynomial:
        """
        Args:
            c: Integration constant
        Returns:
            Indeterminate integral of self + c
        """

        res, n = [], len(self)

        for i, a in enumerate(self.value):
            res.append(a / (n - i))

        return Polynomial(res + [c])

    def definitive_integral(self, a: float, b: float) -> float:
        """
        Args:
            a: lower bound
            b: upper bound
        Returns:
            Definitive integral of self from a to b
        """

        tmp = self.integrate()

        return tmp(b) - tmp(a)

    def limit(self, negative: bool = False) -> float:
        """
        Args:
            negative: Whether x approaches negative infinity, otherwise positive infinity
        Returns:
            lim(+-inf, self)
        """

        if len(self) < 2:
            return self(0).real

        return (-1) ** (1 - (self[0] > 0) + (bool(negative) and (self.degree % 2))) * inf

    def show(self, var: str = "x") -> str:
        """
        Args:
            var: Variable name
        Returns:
            A string representation of the polynomial
        """

        try:
            return str(self.evaluate(complex(var)))
        except ValueError:
            ...

        res = ""
        n = len(self)

        for i, a in enumerate(self):
            if a == 0:
                continue

            positive = a > 0
            a = abs(a)
            exp = n - i - 1
            sign = ("+ " if positive else "- ")

            if exp:
                if a != 1:
                    res += (sign if a else "") + str(round_if_possible(a)) + var + f"^{exp}" * (exp != 1) + " "
                else:
                    res += sign + var + f"^{exp}" * (exp != 1) + " "
            else:
                res += (sign if a else "") + str(round_if_possible(a))

        if res and res[0] == "+":
            res = res[2:]

        return res.strip() if res else "0"

    def solve(self) -> dict[complex, int]:
        """
        Returns:
            A dictionary of the roots of the polynomial and how many times each number is a root
        """

        if (n := self.degree) < 1:
            return {}

        if not self[-1]:
            res = Polynomial(self[:-1]).solve()

            if 0 in res:
                res[0] += 1

            return res

        if n in {1, 2}:
            return quadratic_solver(*self)

        if not any(self[1:-1]):
            c, phi, res = self[-1] / self[0], pi, {}

            if not c:
                return {0: n}

            if c < 0:
                c *= -1
                phi = -phi

            root_c = c ** (1 / n)

            for k in range(n):
                curr = (phi + 2 * pi * k) / n
                res[root_c * complex(cos(curr), sin(curr))] = 1

            return res

        if n in {3, 4}:
            res = (cubic_solver if n == 3 else quartic_solver)(*self)
            tmp = self // Polynomial([1, -res])
            roots = tmp.solve()

            if res not in roots:
                roots[res] = 0

            roots[res] += 1

            return roots

        return ...

    def factorize(self) -> dict[Polynomial, int]:
        """
        Returns:
            A dictionary of polynomials p: n where the product of p^n for all p = self and p are simple polynomials
        """

        roots, c_roots = self.solve(), []
        res, remaining = {}, self.copy()

        for k, v in roots.items():
            if k.imag:
                c_roots.append(k)
            else:
                res[curr := Polynomial([1, -k])] = v
                remaining //= curr

        if remaining:
            res_roots = {}

            for r in c_roots:
                if r not in res_roots and r.conjugate() not in res_roots:
                    res_roots[r] = roots[r]

            for k, v in res_roots.items():
                res[Polynomial([1, -2 * k.real, abs(k) ** 2])] = v

        return res

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __call__(self, x: complex) -> complex:
        """
        Args:
            x: A complex number
        Returns:
            self(x)
        """

        return self.evaluate(x)

    def __add__(self, other: Polynomial | Iterable[float] | float) -> Polynomial:
        """
        Args:
            other: Another polynomial
        Returns:
            self + other
        """

        if isinstance(other, Polynomial):
            shorter, longer = min(self.value, other.value, key=len), max(other.value, self.value, key=len)

            return Polynomial(list(map(lambda t: t[0] + t[1], zip(self.extend_0s(shorter, len(longer)), longer))))

        return self + Polynomial(other)

    def __sub__(self, other: Polynomial | Iterable[float] | float) -> Polynomial:
        """
        Args:
            other: Another polynomial
        Returns:
            self - other
        """

        if isinstance(other, Polynomial):
            return self + -other

        return self - Polynomial(other)

    def __mul__(self, other: Polynomial | Iterable[float] | float) -> Polynomial:
        """
        Args:
            other: Another polynomial
        Returns:
            self * other
        """

        def helper(k):
            return sum(self[i] * other[k - i] for i in range(max(0, k - q + 1), min(p, k + 1)))

        if isinstance(other, Polynomial):
            p, q = len(self), len(other)

            return Polynomial(list(map(helper, list(range(p + q - 1)))))

        return self * Polynomial(other)

    def __floordiv__(self, other: Polynomial | Iterable[float] | float) -> Polynomial:
        """
        Args:
            other: Another polynomial
        Returns:
            The maximal polynomial p: p * other <= self
        """

        return (self / other)[0]

    def __truediv__(self, other: Polynomial | Iterable[float] | float) -> tuple[Polynomial, Polynomial]:
        """
        Args:
            other: Another polynomial
        Returns:
            A tuple with the maximal polynomial p: p * other <= self and the remainder self - p
        """

        if not other:
            raise ZeroDivisionError("Zero polynomial")

        if isinstance(other, (int, float)):
            return Polynomial(list(map(lambda a: a[0] / other, self))), Polynomial([0])

        if not isinstance(other, Polynomial):
            other = Polynomial(other)

        if self < other:
            return Polynomial([]), self

        p, q = len(self), len(other)
        res = [0] * (p - q + 1)
        current = self.copy()

        for i in range(len(res)):
            ratio = current[0] / other[0]
            current -= Polynomial([ratio] + [0] * (p - q - i)) * other
            res[i] = ratio

        return Polynomial(res), current

    def __pow__(self, power: int) -> Polynomial:
        """
        Args:
            power: An integer
        Returns:
            self ^ power
        """

        if power < 0:
            return Polynomial([])

        res = Polynomial([1])

        for _ in range(power):
            res *= self

        return res

    def __neg__(self) -> Polynomial:
        """
        Returns:
           The polynomial negated
        """

        return Polynomial([-x for x in self.value])

    def __bool__(self) -> bool:
        """
        Returns:
           self.show() != 0
        """

        return bool(self.value)

    def __len__(self) -> int:
        """
        Returns:
           Number of coefficients
        """

        return len(self.value)

    def __getitem__(self, item: int | slice) -> float | list[float]:
        """
        Args:
            item: An integer or slice
        Returns:
            self.value[item]
        """

        return self.value[item]

    def __setitem__(self, key: int | slice, value: float) -> None:
        """
        Args:
            key: An integer or slice
            value: A new coefficient
        Changes coefficient or sequence of coefficients self[key] to value
        """

        self.value[key] = value

    def __eq__(self, other) -> bool:
        """
        Args:
            other: Another polynomial
        Returns:
            If other is the same polynomial as self
        """

        if type(other) == Polynomial:
            return self.value == other.value

        return False

    def __lt__(self, other: Polynomial) -> bool:
        """
        Args:
            other: Another polynomial
        Returns:
            If lim(inf, self) < lim(inf, other)
        """

        if len(self) != len(other):
            return len(self) < len(other)

        for a, b in zip(self.value, other.value):
            if a != b:
                return a < b

        return False

    def __str__(self) -> str:
        """
        Returns:
            A string representation of the list of coefficients
        """

        return str(self.value)

    __repr__ = __str__
