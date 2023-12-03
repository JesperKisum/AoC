import os

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
from aocd import get_data
import numpy as np

data = get_data(day=1, year=2023).splitlines()
test_data = ["1abc2", "pqr3stu8vwx", "a1b2c3d4e5f", "treb7uchet"]
# Part 1
# print(data)
digits = []
for line in data:
    # find the first number in string
    for i in range(len(line)):
        if line[i].isdigit():
            first = line[i]
            break
    # find the last number in string
    for i in range(len(line)):
        i = i + 1
        if line[-i].isdigit():
            last = line[-i]
            break
    digits.append(int(first + last))
print(sum(digits))


test2_data = [
    "two1nine",
    "eightwothree",
    "abcone2threexyz",
    "xtwone3four",
    "4nineeightseven2",
    "zoneight234",
    "7pqrstsixteen",
]


def replace_written_numbers(line):
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for number in numbers:
        line = line.replace(number, number + str(numbers.index(number) + 1) + number)
    return line


def find_first(line):
    for i in range(len(line)):
        if line[i].isdigit():
            return line[i]


def find_last(line):
    for i in range(len(line)):
        i = i + 1
        if line[-i].isdigit():
            return line[-i]


digits2 = []

for line in data:
    line = replace_written_numbers(line)

    # find the first number in string
    for i in range(len(line)):
        if line[i].isdigit():
            first = line[i]
            break
    # find the last number in string
    for i in range(len(line)):
        i = i + 1
        if line[-i].isdigit():
            last = line[-i]
            break
    digits2.append(int(first + last))
print(sum(digits2))
data = get_data(day=1, year=2023).splitlines()
print(
    sum(
        [
            int(
                str(find_first(replace_written_numbers(line)))
                + str(find_last(replace_written_numbers(line)))
            )
            for line in data
        ]
    )
)
