
def part1():
    x = 1
    cycle = 0
    strength = {}


    for l in open("10.txt").read().splitlines():
        if "addx" in l:
            cycle += 1
            strength[cycle] = (x, x * cycle)
            cycle += 1
            strength[cycle] = (x, x * cycle)
            x += int(l.split(" ")[1])
        else:
            cycle += 1
            strength[cycle] = (x, x * cycle)

    print(sum([strength[cycle][1] for cycle in [20,60,100,140,180,220]]))



def part2():
    x = 1
    cycle = 0
    strength = {}

    for l in open("10.txt").read().splitlines():
        if "addx" in l:
            cycle += 1
            strength[cycle] = (x, x * cycle)
            cycle += 1
            strength[cycle] = (x, x * cycle)
            x += int(l.split(" ")[1])
        else:
            cycle += 1
            strength[cycle] = (x, x * cycle)

    screen = [[' ' for x in range(40)] for y in range(6)]

    for cycle in strength:
        x = strength[cycle][0]
        if (cycle-1)%40 in [x-1,x,x+1]:
            screen[(cycle-1)//40][(cycle-1)%40] = "0"

    for l in screen:
        print("".join(l))


if __name__ == "__main__":

    part1()
    part2()