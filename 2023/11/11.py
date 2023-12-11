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

day = 11
year = 2023
data = get_data(day=day, year=year).splitlines()
examples = aocd.models.Puzzle(year, day)._get_examples()
examples = [{"input_data": ex.input_data.splitlines()} for ex in examples]
# [print(ex) for ex in examples[0]["input_data"]]
# data = examples[0]["input_data"]

# part 1


def sum_of_paths(paths):
    if type(paths) == set:
        paths = list(paths)
    path_sum = 0
    for i in range(len(paths)):
        for r in range(i + 1, len(v)):
            path_sum += abs(paths[i][0] - paths[r][0]) + abs(paths[i][1] - paths[r][1])
    return path_sum


path_sum = 0

# x = data

# find all empty rows
y = []
for i in data:
    y.append(i)
    if "#" not in i:
        y.append(i)

#  transpose matrix
y = list(zip(*y))
x = dp(y)


# find all empty columns
y = []
for i in x:
    y.append(i)
    if "#" not in i:
        y.append(i)

# find all galaxies as a set of coordinates
x = dp(y)
v = set()
for i in range(len(x)):
    for r in range(len(x[0])):
        if x[i][r] == "#":
            v.add((i, r))

# find all shortest paths between galaxies
v = list(v)
print("9418609 = ", sum_of_paths(v))


# part 2
# clearing the matrix
t = 0
y = []

for i in range(len(data)):
    if "#" not in data[i]:
        y.append(i)

yy = []
for r in range(len(data[0])):
    for i in range(len(data)):
        if data[i][r] == "#":
            break
    else:
        yy.append(r)


v = set()

ii = 0
rr = 0
for i in range(len(data)):
    if i in y:
        ii += 1000000
        ii -= 1
    rr = 0
    for r in range(len(data[0])):
        if r in yy:
            rr += 1000000
            rr -= 1
        if data[i][r] == "#":
            v.add((i + ii, r + rr))

print("593821230983 = ", sum_of_paths(v))
