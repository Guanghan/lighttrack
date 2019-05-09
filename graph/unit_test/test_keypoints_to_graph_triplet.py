'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    October 22th, 2018

    Unit test for data preparation
'''
import sys, os
sys.path.append(os.path.abspath("../utils/"))
from keypoints_to_graph_triplet import *
import pickle

def test_load_data_for_gcn_train():
    dataset_str = "posetrack_18"
    dataset_split_str = "train"
    graph_triplet_list_all = load_data_for_gcn(dataset_str, dataset_split_str)
    print("graph_pair_list Top 10: {}".format(graph_triplet_list_all[0:10]))
    print("number of graph triplets collected: {}".format(len(graph_triplet_list_all)))

    output_folder = "."
    data_out_path = '{}/posetrack_train_data_triplet.pickle'.format(output_folder)
    with open(data_out_path, 'wb') as handle:
        pickle.dump(graph_triplet_list_all, handle)

    with open('./posetrack_train_data_triplet.pickle', 'rb') as handle:
        restore = pickle.load(handle)
    print(restore == graph_triplet_list_all)


def test_load_data_for_gcn_val():
    dataset_str = "posetrack_18"
    dataset_split_str = "val"
    graph_triplet_list_all = load_data_for_gcn(dataset_str, dataset_split_str)
    print("graph_pair_list Top 10: {}".format(graph_triplet_list_all[0:10]))
    print("number of graph triplets collected: {}".format(len(graph_triplet_list_all)))

    output_folder = "."
    data_out_path = '{}/posetrack_val_data_triplet.pickle'.format(output_folder)
    with open(data_out_path, 'wb') as handle:
        pickle.dump(graph_triplet_list_all, handle)

    with open('./posetrack_val_data_triplet.pickle', 'rb') as handle:
        restore = pickle.load(handle)
    print(restore == graph_triplet_list_all)

if __name__ == "__main__":
    test_load_data_for_gcn_train()
    test_load_data_for_gcn_val()
