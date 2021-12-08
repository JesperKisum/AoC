import numpy as np

data = [int(i) for i in open("07.txt").read().split(",")]

t = np.median(data)
f = sum([abs(x-t) for x in data])
print(f"Target pos {t}, Fuel: {f}")


cost = {i:sum([abs(x-i) * (abs(x-i)+1) / 2 for x in data]) for i in range(0, max(np.unique(data)))}
result = min(cost, key=cost.get)
print(f"Target pos {result}, Fuel: {cost[result]}")