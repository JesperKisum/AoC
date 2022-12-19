def parse_data():
    with open("15.txt") as f:
        data = f.read().strip().splitlines()
    res = []
    for row in data:
        # Sensor at x=2, y=18: closest beacon is at x=-2, y=15
        s, b = row.split(": ")
        s = s.lstrip("Sensor at ")
        sx, sy = s.split(", ")
        sx, sy = int(sx[2:]), int(sy[2:])
        b = b.lstrip("closest beacon is at ")
        bx, by = b.split(", ")
        bx, by = int(bx[2:]), int(by[2:])
        res.append((sx, sy, bx, by))
    return res


def dist(sx, sy, bx, by):
    return abs(sx - bx) + abs(sy - by)


def part1(res):
    y = 2000000
    xx = set()

    for sx, sy, bx, by in res:
        x = dist(sx, sy, bx, by) - abs(sy - y)
        xx.update(range(sx - x, sx + x + 1))
        if by == y:
            xx.discard(bx)

    print(len(xx))


def part2(res):
    xx = set()
    M = 4000000
    for sx, sy, bx, by in res:
        d = dist(sx, sy, bx, by) + 1
        for x in range(sx - d, sx + d + 1):
            y = sy + (d - abs(sx - x))
            if (0 <= x <= M) and (0 <= y <= M):
                xx.add((x, y))
            y = sy - (d - abs(sx - x))
            if (0 <= x <= M) and (0 <= y <= M):
                xx.add((x, y))

    for x, y in xx:
        if all(dist(sx, sy, x, y) > dist(sx, sy, bx, by) for sx, sy, bx, by in res):
            print(x, y)
            print(x * M + y)
            break


res = parse_data()
part1(res)
part2(res)
