import numpy as np


def main():
    with open("01.txt", "r") as f:
        data = f.read().strip().split("\n\n")
        data = [i.split("\n") for i in data]
        data = [[int(i) for i in j] for j in data]
        sums = [sum(i) for i in data]
        max = np.max(sums)
        print(max)
        sums_sorted = np.sort(sums)
        print(sums_sorted[-1] + sums_sorted[-2] + sums_sorted[-3])


if __name__ == "__main__":
    main()
