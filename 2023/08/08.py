import os
from collections import Counter

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np

day = 8
year = 2023
# data = get_data(day=day, year=year).splitlines()
with open("input.txt", "r") as file:
    data = file.read().splitlines()

examples = aocd.models.Puzzle(year, day)._get_examples()
# save examples onput to txt file
for idx, ex in enumerate(examples):
    with open("day" + str(day) + "_ex" + str(idx) + ".txt", "w") as file:
        file.write(ex.input_data)

examples = [
    {
        "input_data": ex.input_data.splitlines(),
    }
    for ex in examples
]
# [print(ex) for ex in examples[0]["input_data"]]
# data = examples[0]["input_data"]
mappings = {
    line[:3]: {"L": line.split(",")[0][-3:], "R": line.split(",")[1][1:4]}
    for line in data[2:]
}
steps = [i for i in data[0]]
# print(steps)
START = "AAA"
CURRENT = "AAA"


def solve(CURRENT, mappings, steps):
    steps_run = 0
    while CURRENT != "ZZZ":
        for step in steps:
            steps_run += 1
        if CURRENT == "ZZZ":
            break
        CURRENT = mappings.get(CURRENT).get(step)
    return steps_run


print(solve(CURRENT, mappings, steps))


def get_start(mappings):
    return [key for key in mappings.keys() if key.endswith("A")]


def solve2(CURRENT, mappings, steps):
    steps_run = 0
    while not CURRENT.endswith("Z"):
        for step in steps:
            steps_run += 1
        if CURRENT.endswith("Z"):
            break
        CURRENT = mappings.get(CURRENT).get(step)
    return steps_run


starts = get_start(mappings)

results = [solve2(start, mappings, steps) for start in starts]

print(starts)
print(results)


from math import lcm

print(lcm(*results))
