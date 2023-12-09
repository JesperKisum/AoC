import os
from collections import Counter

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np

day = 9
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

series = [list(map(int ,line.split(" "))) for line in data]
#[print(line) for line in series]

def subset(series):
    subserieses = []
    subserieses.append(series)
    while any([True if i!=0 else False  for i in subserieses[-1]]):
        
        subseries = []
        for i in range(len(subserieses[-1])-1):
            if i ==len(subserieses[-1])-1:
                break
            subseries.append(subserieses[-1][i+1]-subserieses[-1][i])

        subserieses.append(subseries)
    return subserieses





def extrapolate_subset(subset):
    start = 0
    for line in subset[:-1]:
        start = start+line[-1]
    return start
print("part 1: ",
sum([extrapolate_subset(subset(series[i])) for i in range(len(series))] )
)

def extrapolate_subset_backwards(subset):
    start = 0
    for line in reversed(subset[:-1]):
        start = line[0]-start
    return start
print( "part 2: ",
sum([extrapolate_subset_backwards(subset(series[i])) for i in range(len(series))] )
)