import os

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np

day = 5
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
[print(ex) for ex in examples[0]["input_data"]]
data = examples[0]["input_data"]

seeds = data[0].split(":")[1].split(" ")[1:]
data.pop(0)
data.pop(0)
#print(data)
maps = []
idx = np.where(np.array(data)=="",1,0)
#print(idx)
map_num=0

map = []
for n, line in enumerate(data):
    if idx[n]==0:
        map.append(line)
    if idx[n]==1:
        maps.append(map)
        map=[]
maps.append(map)


for m, map in enumerate(maps):
    for n, line in enumerate(map):
        if n == 0:
            maps[m][n] = line.split(" ")[0]
        else:
            maps[m][n] = [int(l) for l in line.split(" ")]

[print(m) for m in maps]

location_for_seeds = []

seeds = [int(seed) for seed in seeds]
print()
print("seeds", seeds)
for seed in seeds:
    for map in maps:
        found = False
        idx = 1
        while not found:
            if seed in list(range(map[idx][0], map[idx][0]+map[idx][2])):
               found = True
               corresponding_idx = list(range(map[idx][0], map[idx][0]+map[idx][2])).index(seed)
               print("seed", seed,"corresponding location",list(range(map[idx][1], map[idx][1]+map[idx][2]))[corresponding_idx])
               seed = list(range(map[idx][1], map[idx][1]+map[idx][2]))[corresponding_idx]
            elif idx == len(map)-1:
                print("seed", seed,"corresponding location", seed)
                found = True
                seed = seed
            else:
                idx += 1
    location_for_seeds.append(seed)

print("location_for_seeds", location_for_seeds)
            