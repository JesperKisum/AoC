rpsValue = {"A": 1, "B": 2, "C": 3}  # Rock Paper Scissors values
rpsWC = {"A": "C", "B": "A", "C": "B"}  # Win condition
rpsLC = {val: key for (key, val) in rpsWC.items()}  # Lose condition


def roundOutcome(game):
    if rpsWC[game[0]] == game[1]:
        return 0
    elif rpsWC[game[1]] == game[0]:
        return 6
    else:
        return 3


def part_1(game):
    equivalences = {"X": "A", "Y": "B", "Z": "C"}
    t_game = [game[0], equivalences[game[1]]]
    return roundOutcome(t_game) + rpsValue[t_game[1]]


def part_2(game):
    if game[1] == "X":
        chosen_outcome = rpsWC[game[0]]
    elif game[1] == "Z":
        chosen_outcome = rpsLC[game[0]]
    else:
        chosen_outcome = game[0]
    return rpsValue[chosen_outcome] + roundOutcome([game[0], chosen_outcome])


with open("02.txt") as file:
    List = [i.split(" ") for i in file.read().split("\n")[:-1]]
print(sum(map(part_1, List)))
print(sum(map(part_2, List)))
