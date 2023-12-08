import os
from collections import Counter

os.environ.setdefault(
    "AOC_SESSION",
    "53616c7465645f5f6388e91879f5f8bf2fe829d3d89bc51f27228930fbad662162f04ade3dd7fd1690ae157906831b72b44b3ee6e9de4470c781a10f68415cab",
)
import aocd
from aocd import get_data
import numpy as np

day = 7
year = 2023
data = get_data(day=day, year=year).splitlines()
examples = aocd.models.Puzzle(year, day)._get_examples()
# save examples onput to txt file
for idx, ex in enumerate(examples):
    with open("day" + str(day) + "_ex" + str(idx) + ".txt", "w") as file:
        file.write(ex.input_data)

examples = [
    {
        "input_data": ex.input_data.splitlines(),
    }
    for ex in examples
]
# [print(ex) for ex in examples[0]["input_data"]]
# data = examples[0]["input_data"]


CARD_VALUES = {str(i): i for i in range(2, 10)}
CARD_VALUES["T"] = 10
CARD_VALUES["J"] = 11
CARD_VALUES["Q"] = 12
CARD_VALUES["K"] = 13
CARD_VALUES["A"] = 14

HIGH_CARD = 1
ONE_PAIR = 2
TWO_PAIR = 3
THREE_OF_A_KIND = 4
FULL_HOUSE = 5
FOUR_OF_A_KIND = 6
FIVE_OF_A_KIND = 7


def cards_to_card_values(cards):
    return [CARD_VALUES[c] for c in cards]


def cards_to_type(cards):
    card_frequencies = [count for _, count in Counter(cards).most_common()]

    match card_frequencies:
        case [5, *_]:
            return FIVE_OF_A_KIND
        case [4, *_]:
            return FOUR_OF_A_KIND
        case [3, 2, *_]:
            return FULL_HOUSE
        case [3, *_]:
            return THREE_OF_A_KIND
        case [2, 2, *_]:
            return TWO_PAIR
        case [2, *_]:
            return ONE_PAIR
        case _:
            return HIGH_CARD


def order_hand(hand):
    return cards_to_type(hand[0]), cards_to_card_values(hand[0])


hands = []
for line in data:
    cards, bid = line.split()
    hands.append((cards, int(bid)))

hands.sort(key=order_hand)
print(sum(i * hand[1] for i, hand in enumerate(hands, 1)))


# part 2


CARD_VALUES["J"] = 0  # <-- change here


def cards_to_type_2(cards):
    counter = Counter(cards)

    joker_count = counter["J"]  # <-- change here
    if 0 < joker_count < 5:
        del counter["J"]
        counter[counter.most_common(1)[0][0]] += joker_count

    card_frequencies = [count for _, count in counter.most_common()]
    match card_frequencies:
        case [5, *_]:
            return FIVE_OF_A_KIND
        case [4, *_]:
            return FOUR_OF_A_KIND
        case [3, 2, *_]:
            return FULL_HOUSE
        case [3, *_]:
            return THREE_OF_A_KIND
        case [2, 2, *_]:
            return TWO_PAIR
        case [2, *_]:
            return ONE_PAIR
        case _:
            return HIGH_CARD


def order_hand(hand):
    return cards_to_type_2(hand[0]), cards_to_card_values(hand[0])


hands = []
for line in data:
    cards, bid = line.split()
    hands.append((cards, int(bid)))

hands.sort(key=order_hand)
print(sum(i * hand[1] for i, hand in enumerate(hands, 1)))
