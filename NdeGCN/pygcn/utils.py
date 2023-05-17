import torch
from time_process import *


def load_data(node_number, dataset, pre_order):
    print('Loading {} dataset...'.format(dataset))
    path = "../data/" + dataset
    second_sample = np.loadtxt(path + "/" + dataset + '_2.txt', delimiter=" ")
    high_sample = np.loadtxt(path + "/" + dataset + '_3.txt', delimiter=" ")
    second_train, _, high_train, high_test = preprocess_time(second_sample, high_sample)
    adj_matrix_pair = contruct_adj(second_train, node_number)
    adj_matrix_pair = adj_matrix_pair + np.eye(node_number)
    incidence_matrix_pre = np.zeros(shape=(node_number, high_train.shape[0]))
    for i in range(high_train.shape[0]):
        x = high_train[i][0:-1] - 1
        incidence_matrix_pre[x.astype(int), i] = 1
    degree_edge_pre = np.sum(incidence_matrix_pre, axis=1)
    degree_matrix_pre = np.diag(degree_edge_pre)
    adj_matrix_pre = incidence_matrix_pre.dot(incidence_matrix_pre.T) - degree_matrix_pre
    adj_matrix_2pre = adj_matrix_pre + adj_matrix_pair

    # delete the existing triangles
    # pos = high_test - 1
    # pos_train = high_train - 1
    # adj = adj_matrix_pair
    # with_raw = []
    # with_raw_train = []
    # edge_number_diff = (pos.shape[1] - 1) * (pos.shape[1] - 2) / 2
    # for i in range(pos.shape[0]):
    #     adj_ab = 0
    #     for j in range(pos.shape[1] - 1):
    #         for k in range(j + 1, pos.shape[1] - 1):
    #             adj_ab = adj_ab + adj[int(pos[i][j]), int(pos[i][k])]
    #     if adj_ab == edge_number_diff:
    #         with_raw.append(i)
    # final_pos_test = np.delete(pos, with_raw, axis=0)
    # for i in range(pos_train.shape[0]):
    #     adj_ab_train = 0
    #     for j in range(pos_train.shape[1] - 1):
    #         for k in range(j + 1, pos_train.shape[1] - 1):
    #             adj_ab_train = adj_ab_train + adj[int(pos_train[i][j]), int(pos_train[i][k])]
    #     if adj_ab_train == edge_number_diff:
    #         with_raw_train.append(i)
    # final_pos_train = np.delete(pos_train, with_raw_train, axis=0)

    # negative sampling
    # pos_train_sample, neg_train_sample = neg_sampling_random(final_pos_train.shape[0], final_pos_train[:, 0:-1],  adj_matrix_pair)
    # pos_test_sample, neg_test_sample = neg_sampling_random(final_pos_test.shape[0], final_pos_test[:, 0:-1], adj_matrix_pair)

    # save negative sample
    # np.savetxt(path + "/" + pre_order + '_order' + "/" + 'train_neg.txt', neg_train_sample, fmt="%d", delimiter=",")  # 改为保存为整数，以逗号分隔
    # np.savetxt(path + "/" + pre_order + '_order' + "/" + 'test_neg.txt', neg_test_sample, fmt="%d", delimiter=",")  # 改为保存为整数，以逗号分隔
    # np.savetxt(path + "/" + pre_order + '_order' + "/" + 'random_train_neg.txt', neg_train_sample, fmt="%d", delimiter=",")  # 改为保存为整数，以逗号分隔
    # np.savetxt(path + "/" + pre_order + '_order' + "/" + 'random_test_neg.txt', neg_test_sample, fmt="%d", delimiter=",")  # 改为保存为整数，以逗号分隔

    # load negative sample
    pos_train_sample = np.loadtxt(path + "/" + pre_order + '_order' + "/" + 'train_pos.txt', delimiter=",")   # 读入的时候也需要指定逗号分隔
    pos_test_sample = np.loadtxt(path + "/" + pre_order + '_order' + "/" + 'test_pos.txt', delimiter=",") # 读入的时候也需要指定逗号分隔
    neg_train_sample = np.loadtxt(path + "/" + pre_order + '_order' + "/" + 'random_train_neg.txt', delimiter=",")  # 读入的时候也需要指定逗号分隔
    neg_test_sample = np.loadtxt(path + "/" + pre_order + '_order' + "/" + 'random_test_neg.txt', delimiter=",")  # 读入的时候也需要指定逗号分隔

    adj_matrix_2pre = DAD(adj_matrix_2pre)
    adj_matrix = adj_matrix_2pre
    adj_matrix_p = np.zeros((node_number, node_number))
    adj_matrix_n = np.zeros((node_number, node_number))
    for i in range(node_number):
        non_num = np.array(np.nonzero(adj_matrix[i, :]))
        mean_score = np.sum(adj_matrix[i, :]) / non_num.shape[1]
        big = np.where(adj_matrix[i, :] >= mean_score)
        adj_matrix_p[i, big] = adj_matrix[i, big]
        small = np.where(adj_matrix[i, :] < mean_score)
        adj_matrix_n[i, small] = adj_matrix[i, small]
    degree_p = np.sum(adj_matrix_p, axis=1)
    degree_n = np.sum(adj_matrix_n, axis=1)
    degree_ma_p = np.diag(degree_p)
    degree_ma_n = np.diag(degree_n)
    lap_matrix_p = degree_ma_p - adj_matrix_p
    lap_matrix_n = degree_ma_n - adj_matrix_n
    lap_matrix_p = torch.tensor(lap_matrix_p)
    lap_matrix_n = torch.tensor(lap_matrix_n)
    adj_matrix = torch.tensor(adj_matrix)

    return adj_matrix, lap_matrix_p, lap_matrix_n, incidence_matrix_pre, node_number, \
            neg_train_sample, pos_train_sample, neg_test_sample, pos_test_sample


def DAD(matrix):
    D = np.diag(np.sum(matrix, axis=1) ** (-0.5))
    D[np.isinf(D)] = 0
    matrix_norm = np.dot(np.dot(D, matrix), D)
    return matrix_norm

