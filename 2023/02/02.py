import os

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np
day=2
year=2023
data = get_data(day=2, year=2023).splitlines()
examples = aocd.models.Puzzle(2023, 2)._get_examples()
# save examples onput to txt file
for idx, ex in enumerate(examples):
    with open("day" + str(day) + "_ex" + str(idx) + ".txt", "w") as file:
        file.write(ex.input_data)

examples = [{"input_data": ex.input_data.splitlines(), "part_1": ex.answer_a, "part_2": ex.answer_b } for ex in examples]


# Part 1
#print(data[0].split(":")[0].replace("Game ",""))
#print(
#data = examples[0]["input_data"]    
games = {game.split(":")[0].replace("Game ",""):  game.split(":")[1].split(";") for game in data}
#)

#print(
#    [games[i] for i in games]
#)
r_limit = 12
g_limit = 13
b_limit = 14
successes = [str(i+1) for i in range(len(games))]
for round in games:
    #print(games[round])
    print("new round", round)
    for draw in games[round]:
        r = 0
        g = 0
        b = 0
        print(draw.split(" ")[1:])
        res = draw.split(" ")[1:]
        for idx , color in enumerate(res):
            color = color.replace(",","")
            if color == "red":
                r += int(res[idx-1])
            elif color == "green":
                g += int(res[idx-1])
            elif color == "blue":
                b += int(res[idx-1])
        print(r,g,b)
        if r > r_limit or g > g_limit or b > b_limit:
            successes.remove(str(round))
            print("fail")
            break
        else:
            print("success")

print(sum(list(map(int, set(successes)))))

        
# part 2
print("part 2")
print("part 2")
print("part 2")
powers = []
for round in games:
    #print(games[round])
    print("new round", round)
    r = 0
    g = 0
    b = 0
    pwr = 0
    for draw in games[round]:
        print(draw.split(" ")[1:])
        res = draw.split(" ")[1:]
        for idx , color in enumerate(res):
            color = color.replace(",","")
            if color == "red":
                if int(res[idx-1])>r:
                    r = int(res[idx-1])
            elif color == "green":
                if int(res[idx-1])>g:
                    g = int(res[idx-1])
            elif color == "blue":
                if int(res[idx-1])>b:
                    b = int(res[idx-1])
    print(r,g,b)
    powers.append(r*g*b)
print(sum(powers))      

            
            