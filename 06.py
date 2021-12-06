#actual solution to the problem
with open('06.txt') as f:
    dat = f.readlines()
dat = dat[:][0]

from collections import deque

def count_fish(data, days):
    life = [0]*9
    for i in data.split(','): life[int(i)] += 1
    fish = deque(life)
    
    for i in range(days):
        spawn = fish.popleft()
        fish[-2] += spawn
        fish.append(spawn)
    return sum(fish)

#print('Part 1', count_fish(dat, 80))

print('Part 2', count_fish(dat, 1000000))
#count_fish(dat, 1000000)