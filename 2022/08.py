def parse_data():
    with open("08.txt") as f:
        data = f.read().strip()
        return [[int(y) for y in x] for x in data.split("\n")]

def part1(tree_map):
    n = len(tree_map)
    m = len(tree_map[0])
    
    visible = set()
    for i in range(n):
        for j in range(m):
            is_visible = False
            for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                ni = i + dx
                nj = j + dy
                v = True
                while ni in range(n) and nj in range(m):
                    if tree_map[ni][nj] >= tree_map[i][j]:
                        v = False
                        break
                    ni += dx
                    nj += dy
                if v:
                    is_visible = True
                    break
            if is_visible:
                visible.add((i, j))

    return len(vis)

def part2(data):
    r = 0
    g = data
    n = len(g)
    m = len(g[0])

    for i in range(n):
        for j in range(m):
            vd = []
            for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                ni = i + dx
                nj = j + dy
                c = 0
                v = True
                while ni in range(n) and nj in range(m):
                    if g[ni][nj] >= g[i][j]:
                        v = False
                        break
                    ni += dx
                    nj += dy
                    c += 1
                vd.append(c + (1 if ni in range(n) and nj in range(m) else 0))
            r = max(r, vd[0]*vd[1]*vd[2]*vd[3])
    
    return r



if __name__ == "__main__":
    data = parse_data()
    print(part1(data))
    print(part2(data))
    