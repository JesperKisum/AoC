import numpy as np

def parse_input():
    with open("08.txt") as f:
        data = f.read()
        data = [line.split() for line in data.splitlines()]
    trees=np.array([np.array([int(n) for n in line[0]]) for line in data])
    return trees

def hidden_tree(mid, up, down, left, right):
    if mid <= up and mid <= down and mid <= left and mid <= right:
        return True
    else:
        return False

def visible_from_border(tree_map,i,j):
    visible_i = True
    visible_j = True
    for n in range(i):
        if tree_map[n,j] !=1:
            visible_i = False
    for m in range(j):
        if tree_map[i,m] !=1:
            visible_j = False
    
    return visible_i or visible_j


def part0(tree_map):
    trees = np.sum([0 if hidden_tree(tree_map[i,j], tree_map[i+1,j],tree_map[i-1,j],tree_map[i,j-1],tree_map[i,j+1]) else 1 for j in range(1,tree_map.shape[1]-1) for i in range(1,tree_map.shape[0]-1)] )
    return trees + (2*tree_map.shape[0]+2*tree_map.shape[1]-4)

def part1(tree_map):
    visible_trees = set()
    for i in range(tree_map.shape[0]):
        for j in range(tree_map.shape[1]):
            if i==0 or j==0 or i==tree_map.shape[0]-1 or j==tree_map.shape[1]-1:
                visible_trees.add((i,j))
            elif not hidden_tree(tree_map[i,j], tree_map[i+1,j],tree_map[i-1,j],tree_map[i,j-1],tree_map[i,j+1]):
                visible_trees.add((i,j))
            elif not visible_from_border(tree_map,i,j):
                visible_trees.add((i,j))
    
    return tree_map.shape[0]*tree_map.shape[1]- len(visible_trees)
            


if __name__ == "__main__":
    # part 1
    tree_map = parse_input()
    print(part1(tree_map))
    