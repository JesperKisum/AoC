from icecream import ic
def parse_data():
    with open(r"16.txt") as f:
        s = f.read().strip()

    go_to = {}
    flows = {}
    for line in s.split("\n"):
        valve = line[6:8]
        flow = line.split("=")[1].split(";")[0]
        _, r = line.split(";")
        r = r.replace("valves","valve")[len(" tunnels lead to valve "):]
        go_to[valve] = r.split(", ")
        flows[valve] = int(flow)

    return go_to, flows


go_to, flows = parse_data()

ic(go_to, flows)
