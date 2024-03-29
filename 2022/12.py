
from collections import deque

def parse_data():
    return open("12.txt").readlines()

def get_dist(grid, start, end):
    visited = {}
    visited[start] = 0

    to_visit = deque()
    to_visit.append(start)

    deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while to_visit:
        (cx, cy) = to_visit.popleft()
        current_steps = visited[(cx, cy)]

        for (dx, dy) in deltas:
            nx, ny = (cx+dx, cy+dy)
            if (nx, ny) in visited or nx in [-1, len(grid[0])] or ny in [-1, len(grid)]:
                continue
            if ord(grid[ny][nx]) - ord(grid[cy][cx]) <= 1:
                visited[(nx, ny)] = current_steps + 1
                to_visit.append((nx, ny))

                if (nx, ny) == end:
                    return current_steps+1


def part1(data):
    grid = []
    start = (-1, -1)
    end = (-1, -1)

    for y in range(len(data)):
        grid.append(list(data[y].strip()))
        if "S" in data[y]:
            start = (data[y].index("S"), y)
        if "E" in data[y]:
            end = (data[y].index("E"), y)

    grid[start[1]][start[0]] = 'a'
    grid[end[1]][end[0]] = 'z'

    print(get_dist(grid, start, end))



def part2(data):
    grid = []
    start = (-1, -1)
    end = (-1, -1)

    possible_start = []

    for y in range(len(data)):
        grid.append(list(data[y].strip()))
        if "S" in data[y]:
            grid[y][data[y].index("S")] = 'a'
        if "E" in data[y]:
            end = (data[y].index("E"), y)

        possible_start += [(x, y) for x, p in enumerate(grid[y]) if p == 'a']

    grid[end[1]][end[0]] = 'z'

    print(min(filter(lambda x: x is not None, [get_dist(grid, a, end) for a in possible_start])))


if __name__ == "__main__":

    data = parse_data()
    part1(data)
    part2(data)