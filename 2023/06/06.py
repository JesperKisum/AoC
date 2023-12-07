import os

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np

day = 6
year = 2023
data = get_data(day=day, year=year).splitlines()
examples = aocd.models.Puzzle(year, day)._get_examples()
# save examples onput to txt file
for idx, ex in enumerate(examples):
    with open("day" + str(day) + "_ex" + str(idx) + ".txt", "w") as file:
        file.write(ex.input_data)

examples = [
    {
        "input_data": ex.input_data.splitlines(),
        "part_1": ex.answer_a,
        "part_2": ex.answer_b,
    }
    for ex in examples
]
#[print(ex) for ex in examples[0]["input_data"]]
# data = examples[0]["input_data"]

import re


pattern = r'\b\d+\b'

numbers = [re.findall(pattern, i) for i in data]
times = [int(i) for i in numbers[0]]
distances = [int(i) for i in numbers[1]]
print(times)
print(distances)

def dist(t_hold, t_tot):
    velocity = t_hold
    travel_time = t_tot - t_hold
    return velocity * travel_time

wins = []
for idx, time in enumerate(times):
    win = sum([1 for n in range(time) if dist(n,int(time))>int(distances[idx])])
    wins.append(win)
print(np.prod(wins))


#part 2
pattern = r'\b\d+\b'
numbers = [''.join(re.findall(pattern, i)) for i in data]
time = int(numbers[0])
dis = int(numbers[1])
print("time:",time)
print("distance:",dis)

ways_to_beat = []

for i in range(time):
    if (dist(i,time) > dis):
        print(time - 2 * i + 1)
        break

