'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    October 22th, 2018

    Unit test for graph.
'''

import sys, os
sys.path.append(os.path.abspath("../utils/"))
from graph import *

def test_normalize_diagraph():
    num_node = 15
    self_link = [(i, i) for i in range(num_node)]
    neighbor_link = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 8),
                     (8, 7), (7, 6), (8, 12), (12, 9), (9, 10),
                     (10, 11), (9, 3), (12, 13), (13, 14)]
    edge = self_link + neighbor_link
    print("Edge: \n{}\n".format(edge))

    hop_dis = get_hop_distance(num_node, edge, max_hop=1)
    print("Hop_dis: \n{}\n".format(hop_dis))

    max_hop = 1
    dilation = 1
    valid_hop = range(0, max_hop + 1, dilation)
    print("Valid_hop: \n{}\n".format(valid_hop))

    adjacency = np.zeros((num_node, num_node))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    print("Adjacency matrix: \n{}\n".format(adjacency))

    normalize_adjacency = normalize_digraph(adjacency)
    print("Normalized adjacency matrix: \n{}\n".format(normalize_adjacency))
    return


if __name__ == "__main__":
    test_normalize_diagraph()
