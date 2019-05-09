'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Nov 2nd, 2018

    Unit test for data preparation
'''
import sys, os
sys.path.append(os.path.abspath("../utils/"))
from keypoints_to_graph_hard_neg import *
import pickle

def test_load_data_for_gcn_train():
    dataset_str = "posetrack_18"
    dataset_split_str = "train"
    graph_pair_list_all = load_data_for_gcn(dataset_str, dataset_split_str)
    print("graph_pair_list Top 10: {}".format(graph_pair_list_all[0:10]))
    print("number of graph pairs collected: {}".format(len(graph_pair_list_all)))

    output_folder = "."
    data_out_path = '{}/posetrack_train_data_hard_neg.pickle'.format(output_folder)
    with open(data_out_path, 'wb') as handle:
        pickle.dump(graph_pair_list_all, handle)

    with open('./posetrack_train_data_hard_neg.pickle', 'rb') as handle:
        restore = pickle.load(handle)
    print(restore == graph_pair_list_all)


def test_load_data_for_gcn_val():
    dataset_str = "posetrack_18"
    dataset_split_str = "val"
    graph_pair_list_all = load_data_for_gcn(dataset_str, dataset_split_str)
    print("graph_pair_list Top 10: {}".format(graph_pair_list_all[0:10]))
    print("number of graph pairs collected: {}".format(len(graph_pair_list_all)))

    output_folder = "."
    data_out_path = '{}/posetrack_val_data_hard_neg.pickle'.format(output_folder)
    with open(data_out_path, 'wb') as handle:
        pickle.dump(graph_pair_list_all, handle)

    with open('./posetrack_val_data_hard_neg.pickle', 'rb') as handle:
        restore = pickle.load(handle)
    print(restore == graph_pair_list_all)

if __name__ == "__main__":
    test_load_data_for_gcn_train()
    test_load_data_for_gcn_val()
