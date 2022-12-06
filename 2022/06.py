
def parse_data():
    with open("06.txt") as f:
        data = f.read()
        return data

def part1(data, word_length=4):
    """find the fist time four different characters appear in a row and return how many characters before that"""
    words = [data[i:i+word_length] for i in range(len(data)-word_length)]
    for i, word in enumerate(words):
        if len(set(word)) == word_length:
            return i+word_length

    


if __name__ == "__main__":
    data = parse_data()

    print(f"P1: {part1(data,word_length=4)}")
    print(f"P2: {part1(data,word_length=14)}")

    