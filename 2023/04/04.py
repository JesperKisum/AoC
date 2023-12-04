import os

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np

day = 4
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
# data = examples[0]["input_data"]
# print(data)
games = [line.replace("  ", " ").split(":")[1].split("|") for line in data]
points_total = []
for game in games:
    draws = game[0].split(" ")
    while len(draws) > 10:
        if "" in draws:
            draws.pop(draws.index(""))
            # print("pop", len(draws), draws)
    draws = [int(n) for n in draws]
    own_numbers = game[1].split(" ")
    while len(own_numbers) > 25:
        if "" in own_numbers:
            own_numbers.pop(own_numbers.index(""))
            # print("pop", len(own_numbers), own_numbers)
    own_numbers = [int(n) for n in own_numbers]

    points = 0
    for num in draws:
        if num in own_numbers:
            points += 1
    points_total.append(points)
    # print("points", points)
# Part 1
result = [2 ** (i - 1) for i in points_total if i > 0]
print("Part 1:", sum(result))


# part 2

games = [line.replace("  ", " ").split(":")[1].split("|") for line in data]
cards_total = {i: 1 for i in range(1, 1 + len(games))}
# print(cards_total)
for idx, game in enumerate(games):
    draws = game[0].split(" ")
    while len(draws) > 10:
        if "" in draws:
            draws.pop(draws.index(""))
            # print("pop", len(draws), draws)
    draws = [int(n) for n in draws]
    own_numbers = game[1].split(" ")
    while len(own_numbers) > 25:
        if "" in own_numbers:
            own_numbers.pop(own_numbers.index(""))
            # print("pop", len(own_numbers), own_numbers)
    own_numbers = [int(n) for n in own_numbers]

    points = 0
    for num in draws:
        if num in own_numbers:
            points += 1
    if points > 0:
        for n in range(idx + 1, idx + points + 1):
            # print(n, idx, points)
            if n < len(cards_total):
                cards_total[n + 1] += cards_total[idx + 1]

print(sum(cards_total.values()))
