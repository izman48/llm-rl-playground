"""Coding tasks for the code track.

Each task ships a natural-language spec, a handful of *public* example cases
(shown to the agent), a larger set of *held-out* cases (the reward signal, never
shown), and a reference solution (used for sanity checks — also never shown).

The held-out cases are what defeat "memorize the visible answers": a solution
must generalize to inputs it has never seen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    args: tuple[Any, ...]
    expected: Any


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    entry_point: str
    spec: str
    public_cases: tuple[Case, ...]
    held_out_cases: tuple[Case, ...]
    reference: str  # correct solution; for sanity checks only, never shown to agents

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
