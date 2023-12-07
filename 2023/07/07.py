import os

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
        "part_1": ex.answer_a,
        "part_2": ex.answer_b,
    }
    for ex in examples
]
# [print(ex) for ex in examples[0]["input_data"]]
# data = examples[0]["input_data"]

hands = [d.split(" ") for n, d in enumerate(data)]


# functions for checking hands
def five_of_a_kind(hand):
    if all(card == hand[0] for card in hand):
        return hand[0]


def four_of_a_kind(hand):
    array = np.array([card for card in hand])
    unique, counts = np.unique(array, return_counts=True)

    if unique[np.where(counts == 4)]:
        return unique[np.where(counts == 4)][0]


def full_house(hand):
    array = np.array([card for card in hand])
    unique, counts = np.unique(array, return_counts=True)
    if unique[np.where(counts == 3)] and unique[np.where(counts == 2)]:
        return unique[np.where(counts == 3)][0]


def three_of_a_kind(hand):
    array = np.array([card for card in hand])
    unique, counts = np.unique(array, return_counts=True)
    if unique[np.where(counts == 3)]:
        return unique[np.where(counts == 3)][0]


def two_pairs(hand):
    array = np.array([card for card in hand])
    unique, counts = np.unique(array, return_counts=True)
    if len(unique[np.where(counts == 2)]) == 2:
        res = unique[np.where(counts == 2)]
        return res[0]


def one_pair(hand):
    array = np.array([card for card in hand])
    unique, counts = np.unique(array, return_counts=True)
    if len(unique[np.where(counts == 2)]) == 1 and not three_of_a_kind(hand):
        return unique[np.where(counts == 2)][0]


def high_card(
    hand,
    card_values={
        "A": 14,
        "K": 13,
        "Q": 12,
        "J": 11,
        "T": 10,
        "9": 9,
        "8": 8,
        "7": 7,
        "6": 6,
        "5": 5,
        "4": 4,
        "3": 3,
        "2": 2,
    },
):
    value_cards = {
        14: "A",
        13: "K",
        12: "Q",
        11: "J",
        10: "T",
        9: "9",
        8: "8",
        7: "7",
        6: "6",
        5: "5",
        4: "4",
        3: "3",
        2: "2",
    }
    array = np.array([card_values[card] for card in hand])

    return value_cards[np.max(array)]


def duel(kind1, kind2):
    card_types = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    if card_types.index(kind1) > card_types.index(kind2):
        return kind2
    else:
        return kind1


def second_ordering(hand1, hand2):
    card_values = {
        "A": 14,
        "K": 13,
        "Q": 12,
        "J": 11,
        "T": 10,
        "9": 9,
        "8": 8,
        "7": 7,
        "6": 6,
        "5": 5,
        "4": 4,
        "3": 3,
        "2": 2,
    }
    for idx, card in enumerate(hand1):
        if card_values[card] > card_values[hand2[idx]]:
            return hand1
        elif card_values[card] < card_values[hand2[idx]]:
            return hand2


def detect_hand(hand):
    if five_of_a_kind(hand):
        return 7, five_of_a_kind(hand)
    if four_of_a_kind(hand):
        return 6, four_of_a_kind(hand)
    if full_house(hand):
        return 5, full_house(hand)
    if three_of_a_kind(hand):
        return 4, three_of_a_kind(hand)
    if two_pairs(hand):
        return 3, two_pairs(hand)
    if one_pair(hand):
        return 2, one_pair(hand)
    return 1, high_card(hand)


card_values = {
    "A": 14,
    "K": 13,
    "Q": 12,
    "J": 11,
    "T": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
}
test_strings = [
    "A2AAA",
    "AAAAJ",
    "Qaaaa",
    "aaQaa",
    "aJQJe",
    "aaJJJ",
    "aJJJJ",
    "a5234",
    "353aa",
    "AAAAA",
    "23332",
    "55aJQ",
]

# [print(detect_hand(hand.upper())) for hand in test_strings]


def sort_hands(hands):
    sorted_hands = []
    for hand in hands:
        # print("sorted_hands")
        # [print(s) for s in sorted_hands]
        # print()

        rank, card = detect_hand(hand[0])
        multiplier = hand[1]
        l = len(sorted_hands)
        if len(sorted_hands) == 0:
            sorted_hands.append((rank, card, hand))

        else:
            for idx, s_hand in enumerate(sorted_hands):
                # print(idx)
                if rank > s_hand[0]:
                    sorted_hands.insert(idx, (rank, card, hand))
                    break
                elif s_hand[0] == rank:
                    if second_ordering(hand[0], s_hand[2][0]) == hand[0]:
                        sorted_hands.insert(idx, (rank, card, hand))
                        break
                    else:
                        sorted_hands.insert(idx + 1, (rank, card, hand))
                        break
        if len(sorted_hands) == l:
            sorted_hands.append((rank, card, hand))

    return sorted_hands


res = sort_hands([d for d in hands])
sum = 0
for i, r in enumerate(res):
    sum += int(r[2][1]) * (len(res) - i)

print(sum)
