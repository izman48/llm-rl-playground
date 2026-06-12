"""Coding tasks for the code track.

Each task ships a natural-language spec, a handful of *public* example cases
(shown to the agent), a small set of curated *held-out* edge cases, a reference
solution (the source of truth — never shown), and an **input generator**.

Two things defeat "memorize the visible answers":

* the held-out edge cases are never shown; and
* at grade time the generator synthesizes *fresh* inputs every run and the
  reference solution labels them (see ``grader.py``). Because the expected
  outputs are computed live from the reference rather than stored, there is no
  fixed answer key to overfit to — a different seed yields a different test set,
  so a solution must generalize, not memorize.

The generator (``gen_input``) runs in the trusted parent process and returns an
argument tuple; it only needs to stay inside the spec's input domain, since the
reference provides the expected output.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Case:
    args: tuple[Any, ...]
    expected: Any


# Per-task input generators. Each takes a seeded ``random.Random`` and returns a
# fresh argument tuple inside the task's input domain. The reference solution
# turns these inputs into expected outputs at grade time (differential testing),
# so generators never need to know the answer — only how to sample valid inputs.
def _rand_text(rng: random.Random, lo: int = 0, hi: int = 24) -> str:
    alphabet = string.ascii_letters + "0123456789   ,.!?:"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))


def _rand_word(rng: random.Random, lo: int = 1, hi: int = 9) -> str:
    return "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(lo, hi)))


def _gen_text(rng: random.Random) -> tuple[Any, ...]:
    return (_rand_text(rng),)


def _gen_palindrome(rng: random.Random) -> tuple[Any, ...]:
    # Half the time build a genuine palindrome so the True branch is exercised.
    if rng.random() < 0.5:
        base = _rand_text(rng, 0, 12)
        return (base + base[::-1],)
    return (_rand_text(rng),)


def _gen_small_int(rng: random.Random) -> tuple[Any, ...]:
    # Wide domain on purpose: a few memorized public values (fib 0/1/10) must be a
    # negligible fraction of the input space, else "memorize the examples" earns
    # spurious partial credit just from random collisions.
    return (rng.randint(0, 90),)


def _int_to_roman(n: int) -> str:
    table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
        (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
        (5, "V"), (4, "IV"), (1, "I"),
    ]
    out: list[str] = []
    for value, sym in table:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def _gen_roman(rng: random.Random) -> tuple[Any, ...]:
    return (_int_to_roman(rng.randint(1, 3999)),)


def _gen_int_list(rng: random.Random) -> tuple[Any, ...]:
    n = rng.randint(1, 12)  # non-empty: the spec requires a non-empty subarray
    return ([rng.randint(-9, 9) for _ in range(n)],)


def _rand_nested(rng: random.Random, depth: int) -> Any:
    if depth >= 3 or rng.random() < 0.4:
        return rng.randint(-9, 9)
    return [_rand_nested(rng, depth + 1) for _ in range(rng.randint(0, 4))]


def _gen_nested_list(rng: random.Random) -> tuple[Any, ...]:
    return ([_rand_nested(rng, 1) for _ in range(rng.randint(0, 5))],)


def _gen_sentence(rng: random.Random) -> tuple[Any, ...]:
    words = []
    for _ in range(rng.randint(0, 5)):
        w = _rand_word(rng)
        words.append("".join(rng.choice((c.upper(), c.lower())) for c in w))
    return (" ".join(words),)


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    entry_point: str
    spec: str
    public_cases: tuple[Case, ...]
    held_out_cases: tuple[Case, ...]
    reference: str  # correct solution; for sanity checks only, never shown to agents
    gen_input: Callable[[random.Random], tuple[Any, ...]] | None = None

    def prompt(self) -> str:
        lines = [
            self.spec.strip(),
            "",
            f"Define a function named `{self.entry_point}`.",
            "Example cases (you are graded on additional hidden cases):",
        ]
        for c in self.public_cases:
            lines.append(f"  {self.entry_point}{tuple(c.args)!r} == {c.expected!r}")
        return "\n".join(lines)


def _c(*args: Any, expected: Any) -> Case:
    return Case(args=tuple(args), expected=expected)


TASKS: tuple[Task, ...] = (
    Task(
        id="reverse_string",
        title="Reverse a string",
        entry_point="reverse_string",
        spec="Return the input string reversed.",
        public_cases=(_c("hello", expected="olleh"), _c("abc", expected="cba")),
        held_out_cases=(
            _c("Hello, World", expected="dlroW ,olleH"),
            _c("racecar", expected="racecar"),
            _c("Python", expected="nohtyP"),
            _c("ab", expected="ba"),
            _c("xyz!", expected="!zyx"),
        ),
        reference="def reverse_string(s):\n    return s[::-1]\n",
        gen_input=_gen_text,
    ),
    Task(
        id="is_palindrome",
        title="Palindrome check (alphanumeric, case-insensitive)",
        entry_point="is_palindrome",
        spec=(
            "Return True if the string is a palindrome considering only "
            "alphanumeric characters and ignoring case, else False."
        ),
        public_cases=(_c("racecar", expected=True), _c("hello", expected=False)),
        held_out_cases=(
            _c("A man, a plan, a canal: Panama", expected=True),
            _c("", expected=True),
            _c("ab", expected=False),
            _c("No lemon, no melon", expected=True),
            _c("12321", expected=True),
        ),
        reference=(
            "def is_palindrome(s):\n"
            "    t = [c.lower() for c in s if c.isalnum()]\n"
            "    return t == t[::-1]\n"
        ),
        gen_input=_gen_palindrome,
    ),
    Task(
        id="count_vowels",
        title="Count vowels",
        entry_point="count_vowels",
        spec="Return the number of vowels (a, e, i, o, u; case-insensitive) in the string.",
        public_cases=(_c("hello", expected=2), _c("xyz", expected=0)),
        held_out_cases=(
            _c("AEIOU", expected=5),
            _c("", expected=0),
            _c("Programming", expected=3),
            _c("rhythm", expected=0),
            _c("Queue", expected=4),
        ),
        reference=(
            "def count_vowels(s):\n"
            "    return sum(c.lower() in 'aeiou' for c in s)\n"
        ),
        gen_input=_gen_text,
    ),
    Task(
        id="fibonacci",
        title="Nth Fibonacci number (0-indexed)",
        entry_point="fibonacci",
        spec="Return the nth Fibonacci number, 0-indexed: fib(0)=0, fib(1)=1.",
        public_cases=(_c(0, expected=0), _c(1, expected=1), _c(10, expected=55)),
        held_out_cases=(
            _c(2, expected=1),
            _c(7, expected=13),
            _c(12, expected=144),
            _c(15, expected=610),
            _c(20, expected=6765),
        ),
        reference=(
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
        gen_input=_gen_small_int,
    ),
    Task(
        id="roman_to_int",
        title="Roman numeral to integer",
        entry_point="roman_to_int",
        spec="Convert a valid Roman numeral string to its integer value.",
        public_cases=(_c("III", expected=3), _c("IV", expected=4)),
        held_out_cases=(
            _c("IX", expected=9),
            _c("LVIII", expected=58),
            _c("MCMXCIV", expected=1994),
            _c("XL", expected=40),
            _c("DCCC", expected=800),
        ),
        reference=(
            "def roman_to_int(s):\n"
            "    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
            "    total = 0\n"
            "    prev = 0\n"
            "    for ch in reversed(s):\n"
            "        v = vals[ch]\n"
            "        total += -v if v < prev else v\n"
            "        prev = v\n"
            "    return total\n"
        ),
        gen_input=_gen_roman,
    ),
    Task(
        id="max_subarray",
        title="Maximum subarray sum (Kadane)",
        entry_point="max_subarray",
        spec="Return the largest sum of any contiguous non-empty subarray of the list.",
        public_cases=(
            _c([-2, 1, -3, 4, -1, 2, 1, -5, 4], expected=6),
            _c([1], expected=1),
        ),
        held_out_cases=(
            _c([5, 4, -1, 7, 8], expected=23),
            _c([-1, -2, -3], expected=-1),
            _c([2, -1, 2, -1, 2], expected=4),
            _c([-5], expected=-5),
            _c([3, -2, 5, -1], expected=6),
        ),
        reference=(
            "def max_subarray(nums):\n"
            "    best = cur = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        cur = max(x, cur + x)\n"
            "        best = max(best, cur)\n"
            "    return best\n"
        ),
        gen_input=_gen_int_list,
    ),
    Task(
        id="flatten",
        title="Deep-flatten a nested list of ints",
        entry_point="flatten",
        spec="Flatten an arbitrarily nested list of integers into a flat list (left to right).",
        public_cases=(
            _c([1, [2, 3]], expected=[1, 2, 3]),
            _c([[1], [2, [3]]], expected=[1, 2, 3]),
        ),
        held_out_cases=(
            _c([], expected=[]),
            _c([1, 2, 3], expected=[1, 2, 3]),
            _c([[[[1]]], 2], expected=[1, 2]),
            _c([1, [2, [3, [4]]]], expected=[1, 2, 3, 4]),
            _c([[], [1], []], expected=[1]),
        ),
        reference=(
            "def flatten(lst):\n"
            "    out = []\n"
            "    for x in lst:\n"
            "        if isinstance(x, list):\n"
            "            out.extend(flatten(x))\n"
            "        else:\n"
            "            out.append(x)\n"
            "    return out\n"
        ),
        gen_input=_gen_nested_list,
    ),
    Task(
        id="title_case",
        title="Title-case a sentence",
        entry_point="title_case",
        spec=(
            "Capitalize the first letter of each whitespace-separated word and "
            "lowercase the rest; join words with single spaces."
        ),
        public_cases=(
            _c("hello world", expected="Hello World"),
            _c("the QUICK fox", expected="The Quick Fox"),
        ),
        held_out_cases=(
            _c("", expected=""),
            _c("a", expected="A"),
            _c("multiple   spaces", expected="Multiple Spaces"),
            _c("ALL CAPS", expected="All Caps"),
            _c("mixED CaSe", expected="Mixed Case"),
        ),
        reference=(
            "def title_case(s):\n"
            "    return ' '.join(w[:1].upper() + w[1:].lower() for w in s.split())\n"
        ),
        gen_input=_gen_sentence,
    ),
)

_BY_ID = {t.id: t for t in TASKS}


def all_tasks() -> list[Task]:
    return list(TASKS)


def get_task(task_id: str) -> Task:
    try:
        return _BY_ID[task_id]
    except KeyError:
        raise KeyError(f"unknown task_id {task_id!r}; known: {sorted(_BY_ID)}") from None
