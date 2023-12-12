import os
from collections import Counter
from copy import deepcopy as dp

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np

day = 12
year = 2023
data = get_data(day=day, year=year).splitlines()
examples = aocd.models.Puzzle(year, day)._get_examples()
examples = [{"input_data": ex.input_data.splitlines()} for ex in examples]
# [print(ex) for ex in examples[0]["input_data"]]
# data = examples[0]["input_data"]

# part 1

L = data
G = [[c for c in row] for row in L]

# i == current position within dots
# bi == current position within blocks
# current == length of current block of '#'
# state space is len(dots) * len(blocks) * len(dots)
DP = {}


def f(dots, blocks, i, bi, current):
    key = (i, bi, current)
    if key in DP:
        return DP[key]
    if i == len(dots):
        if bi == len(blocks) and current == 0:
            return 1
        elif bi == len(blocks) - 1 and blocks[bi] == current:
            return 1
        else:
            return 0
    ans = 0
    for c in [".", "#"]:
        if dots[i] == c or dots[i] == "?":
            if c == "." and current == 0:
                ans += f(dots, blocks, i + 1, bi, 0)
            elif (
                c == "." and current > 0 and bi < len(blocks) and blocks[bi] == current
            ):
                ans += f(dots, blocks, i + 1, bi + 1, 0)
            elif c == "#":
                ans += f(dots, blocks, i + 1, bi, current + 1)
    DP[key] = ans
    return ans


for part2 in [False, True]:
    ans = 0
    for line in L:
        dots, blocks = line.split()
        if part2:
            dots = "?".join([dots, dots, dots, dots, dots])
            blocks = ",".join([blocks, blocks, blocks, blocks, blocks])
        blocks = [int(x) for x in blocks.split(",")]
        DP.clear()
        score = f(dots, blocks, 0, 0, 0)
        #   print(dots, blocks, score, len(DP))
        ans += score
    print(ans)
