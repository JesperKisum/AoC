import os

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np
day=3
year=2023
data = get_data(day=day, year=year).splitlines()
examples = aocd.models.Puzzle(year, day)._get_examples()
# save examples onput to txt file
for idx, ex in enumerate(examples):
    with open("day" + str(day) + "_ex" + str(idx) + ".txt", "w") as file:
        file.write(ex.input_data)

examples = [{"input_data": ex.input_data.splitlines(), "part_1": ex.answer_a, "part_2": ex.answer_b } for ex in examples]


# Part 1

symbols = list(set([digit for line in data for digit in line ] ))
[symbols.remove(f"{i}")for i in range(10)]
symbols.remove(".")
digits= [str(i) for i in range(10)]

def find_next_symbol(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] in symbols:
                return i,j
    return None

def find_surrounding_number(matrix, i, j):
    # search for number next to symbol at [i,j] and return it as string
    number = []
    for n in [i-1,i,i+1]:
        if n < 0 or n >= len(matrix):
            continue
        for m in [j-1,j,j+1]:
            if m < 0 or m >= len(matrix[n]):
                continue
            #print("searching",n,m)
            if matrix[n][m] in digits:
                
                cursor = matrix[n][m]
                idx = m
                while cursor in digits:
                    number.append(cursor)
                    matrix[n]=  matrix[n][:idx] + "." + matrix[n][idx + 1:]
                    if idx >= len(matrix[n]):
                        break
                    idx += 1
                    if idx >= len(matrix[n]):
                        break
                    cursor = matrix[n][idx]
                
                idx = m-1
                cursor = matrix[n][idx]
                if cursor in digits:
                    while cursor in digits:
                        number.insert(0,cursor)
                        if idx < 0:
                            break
                        matrix[n]=  matrix[n][:idx] + "." + matrix[n][idx + 1:]
                        idx -= 1
                        cursor = matrix[n][idx]
                #print("found number",number)
                return "".join(number)
            
    matrix[i] =  matrix[i][:j] + "." + matrix[i][j + 1:]
    return None

#data = examples[0]["input_data"]

numbers = []
#[print(line) for line in data]
while find_next_symbol(data):
    coords = find_next_symbol(data)
    #print("coords",coords,"symbol",data[coords[0]][coords[1]])
    res = find_surrounding_number(data, *coords)
    if res:
        numbers.append(res)
    #[print(line) for line in data]
    #print(numbers)

print(sum(list(map(int,numbers))))


#part 2
data = get_data(day=day, year=year).splitlines()

def find_next_gear(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == "*":
                coords = [
                    [matrix[i-1][j-1],matrix[i-1][j],matrix[i-1][j+1]],
                    [matrix[i][j-1],matrix[i][j],matrix[i][j+1]],
                    [matrix[i+1][j-1],matrix[i+1][j],matrix[i+1][j+1]]
                ]
                
                
                for n in range(len(coords)):
                    for m in range(len(coords[n])):
                        if coords[n][m] in digits:
                            coords[n][m] = 1
                        else:
                            coords[n][m] = 0
                            

                print(coords)
                            

                        
            
                 
                

                
                return i,j
    return None


def find_next_gear(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == "*":
                return i,j
    return None

ratios = []

while find_next_gear(data):
    input=find_next_gear(data)
    nums_at_gear = []

    go_on = True
    while go_on:
        nums_at_gear.append(find_surrounding_number(data, *input))
        if nums_at_gear[-1] == None:
            go_on = False
            nums_at_gear.pop(-1)
    if len(nums_at_gear) > 1:
        ratio = np.prod(list(map(int,nums_at_gear)))
        ratios.append(ratio)



print(
    sum(ratios)
)