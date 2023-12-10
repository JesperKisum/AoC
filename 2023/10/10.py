import os
from collections import Counter

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np

day = 10
year = 2023
data = get_data(day=day, year=year).splitlines()
examples = aocd.models.Puzzle(year, day)._get_examples()
# save examples onput to txt file
#for idx, ex in enumerate(examples):
#    with open("day" + str(day) + "_ex" + str(idx) + ".txt", "w") as file:
#        file.write(ex.input_data)

examples = [
    {
        "input_data": ex.input_data.splitlines(),
    }
    for ex in examples
]
#[print(ex) for ex in examples[0]["input_data"]]
#data = examples[0]["input_data"]

lines = [line for line in data]
#[print(line) for line in lines]
start_2 = [n for n, line in enumerate(lines) if "S" in line][0]
start_1 = lines[start_2].index("S")
start = [start_1, start_2] # x, y


# part 1
ans = 1

poly = [start]

position = [start[0], start[1]-1]
print(position)
last_move = [0, -1]

while lines[position[1]][position[0]] != 'S':
    poly.append([*position])
    tile = lines[position[1]][position[0]]
    if tile == "|":
        if last_move == [0, 1]:
            position[1] += 1
        elif last_move == [0, -1]:
            position[1] -= 1
    elif tile == "-":
        if last_move == [1, 0]:
            position[0] += 1
        elif last_move == [-1, 0]:
            position[0] -= 1
    elif tile == "7":
        if last_move == [1, 0]:
            position[1] += 1
            last_move = [0, 1]
        elif last_move == [0, -1]:
            position[0] -= 1
            last_move = [-1, 0]
    elif tile == "J":
        if last_move == [1, 0]:
            position[1] -= 1
            last_move = [0, -1]
        elif last_move == [0, 1]:
            position[0] -= 1
            last_move = [-1, 0]
    elif tile == "L":
        if last_move == [-1, 0]:
            position[1] -= 1
            last_move = [0, -1]
        elif last_move == [0, 1]:
            position[0] += 1
            last_move = [1, 0]
    elif tile == "F":
        if last_move == [-1, 0]:
            position[1] += 1
            last_move = [0, 1]
        elif last_move == [0, -1]:
            position[0] += 1
            last_move = [1, 0]
    else:
        print("ERROR")
        break
        
    ans += 1
        
print(ans//2)

# part 2
print("PART 2:")

ans_2 = 0

from matplotlib.path import Path

p = Path(poly)
for y in range(len(lines)):
    for x in range(len(lines[0])):
        if [x, y] in poly:
            continue
        if p.contains_point((x, y)):
            ans_2 += 1

print(ans_2)
