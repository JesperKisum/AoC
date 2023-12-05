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
# [print(ex) for ex in examples[0]["input_data"]]
# data = examples[0]["input_data"]

seeds = data[0].split(":")[1].split(" ")[1:]
data.pop(0)
data.pop(0)
# print(data)
maps = []
idx = np.where(np.array(data) == "", 1, 0)
# print(idx)
map_num = 0

map = []
for n, line in enumerate(data):
    if idx[n] == 0:
        map.append(line)
    if idx[n] == 1:
        maps.append(map)
        map = []
maps.append(map)


for m, map in enumerate(maps):
    for n, line in enumerate(map):
        if n == 0:
            maps[m][n] = line.split(" ")[0]
        else:
            maps[m][n] = [int(l) for l in line.split(" ")]

# [print(m) for m in maps]

location_for_seeds = []

seeds = [int(seed) for seed in seeds]
# print()
# print("seeds", seeds)
# print()
for seed in seeds:
    # print()
    # print(seed)
    for map in maps:
        found = False
        idx = 1
        while not found:
            if seed >= map[idx][1] and seed < sum(map[idx][1:]):
                found = True
                seed = map[idx][0] + seed - map[idx][1]
            elif idx == len(map) - 1:
                found = True
                seed = seed
            else:
                idx += 1
    # print("seed", seed)
    location_for_seeds.append(seed)

print("location_for_seeds", location_for_seeds)
print("lowest location", min(location_for_seeds))

# part2


maps = [[(n[0], n[1], n[2]) for n in map if len(n) == 3] for map in maps]


def solve(range_tuple, maps):
    if not maps:
        return range_tuple[0]
    for map in maps[0]:
        destination = map[0]
        start = map[1]
        rang = map[2]
        # range withing mapping
        if (
            start <= range_tuple[0] < start + rang
            and start <= range_tuple[1] < start + rang
        ):
            seed = (
                destination + range_tuple[0] - start,
                destination + range_tuple[1] - start,
            )
            return solve(seed, maps[1:])
        # range starts inside mapping and ends outside
        elif start <= range_tuple[0] < start + rang and start + rang < range_tuple[1]:
            seed = (destination + range_tuple[0] - start, destination + rang)
            rest = (start + rang, range_tuple[1])
            return min(solve(seed, maps[1:]), solve(rest, maps))
        # range starts outside mapping and ends inside
        elif range_tuple[0] < start and start <= range_tuple[1] < start + rang:
            rest = (range_tuple[0], start - 1)
            seed = (destination, destination + range_tuple[1] - start)
            return min(solve(rest, maps), solve(seed, maps[1:]))
        # range starts and ends outside mapping[
        elif range_tuple[0] < start and range_tuple[1] > start + rang:
            rest1 = (range_tuple[0], start - 1)
            seed = (destination, destination + rang)
            rest2 = (start + rang, range_tuple[1])
            return min(solve(rest1, maps), solve(seed, maps[1:]), solve(rest2, maps))
    return solve(range_tuple, maps[1:])


seed_tuples = [(i, seeds[idx + 1] + i) for idx, i in enumerate(seeds) if idx % 2 == 0]
print("lowest location", min((solve(i, maps) for i in seed_tuples)))
