from icecream import ic
from collections import deque


def parse_data():
    with open("16_test.txt") as f:
        s = f.read().strip()

    go_to = {}
    flows = {}
    for line in s.split("\n"):
        valve = line[6:8]
        flow = line.split("=")[1].split(";")[0]
        _, r = line.split(";")
        r = r.replace("valves", "valve")[len(" tunnels lead to valve ") :]
        go_to[valve] = deque(r.split(", "))
        flows[valve] = int(flow)

    return go_to, flows


def deque_remove(a, b):
    _ = [a.remove(i) if i in a else "" for i in b]
    return a


# DFS algorithm
def dfs(graph, start, flows, visited=None):
    if visited is None:
        visited = deque()
    visited.append(start)

    print(start)
    print(graph[start])
    res = 0
    res_name = start
    for next in deque_remove(graph[start], visited):
        if flows[next] >= res:
            res = flows[next]
            res_name = next
    if res_name != start:
        dfs(graph, res_name, flows, visited)
    else:
        return visited


def part1(go_to, flows):
    


go_to, flows = parse_data()
ic(go_to)
ic(flows)
part1(go_to, flows)

