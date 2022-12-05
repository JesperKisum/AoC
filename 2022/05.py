#!/usr/bin/env python

from copy import deepcopy
from collections import deque

Stacks = dict[int, deque[str]]
Moves = list[tuple[int, int, int]]


def parse_data(file=0) -> tuple[Stacks, Moves]:
    def parse_initial(starting: list[str]):
        cols = {int(c): i for i, c in enumerate(starting[-1]) if c != " "}
        stacks = {}
        for ln in reversed(starting[:-1]):
            for s, i in cols.items():
                if i < len(ln) and ln[i] != " ":
                    stacks.setdefault(s, deque()).append(ln[i])
        return stacks

    def parse_moves(procedure: list[str]):
        return [
            tuple(map(int, (t[1], t[3], t[5])))
            for t in [ln.split() for ln in procedure]
        ]

    with open("05.txt") as f:
        starting, procedure = map(str.splitlines, f.read().split("\n\n"))
        return parse_initial(starting), parse_moves(procedure)


def part1(stacks: Stacks, moves: Moves) -> str:
    stacks = deepcopy(stacks)
    for n, source, target in moves:
        crates = list(stacks[source].pop() for _ in range(n))
        stacks[target].extend(crates)
    return "".join(s[-1] for s in stacks.values())


def part2(stacks: Stacks, moves: Moves) -> str:
    stacks = deepcopy(stacks)
    for n, source, target in moves:
        crates = list(stacks[source].pop() for _ in range(n))
        stacks[target].extend(reversed(crates))
    return "".join(s[-1] for s in stacks.values())


if __name__ == "__main__":
    stacks, moves = parse_data()

    print(f"P1: {part1(stacks, moves)}")
    print(f"P2: {part2(stacks, moves)}")