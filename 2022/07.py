from bisect import bisect_right

def parse_data():
    with open("07.txt") as f:
        data = f.read()

    filesystem = {}
    current_path = [filesystem]
    for line in data.splitlines():
        if line[0] == "$":
            if line[2] == "c":
                if line[5] == "/":
                    current_path = [filesystem]
                elif line[5:].strip() == "..":
                    current_path.pop()
                else:
                    current_path.append(current_path[-1][line[5:]])
        elif line[0:3] == "dir":
            current_path[-1][line[4:].strip()] = {}
        else:
            size, name = line.split()
            current_path[-1][name] = int(size)

    return filesystem


def sum_directory(directory):
    global part1_size
    total_size = 0
    for item in directory.values():
        if isinstance(item, int):
            total_size += item
        else:
            total_size += sum_directory(item)
    if total_size <= 100000:
        part1_size += total_size
    directory_sizes.append(total_size)
    return total_size


# part 1 
part1_size = 0
directory_sizes = []
filesystem = parse_data()
root_size = sum_directory(filesystem)
print(part1_size)

# part 2
required_space = 30000000 - (70000000 - root_size)
directory_sizes.sort()
print(directory_sizes[bisect_right(directory_sizes, required_space)])