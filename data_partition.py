import numpy as np
import random
import os
import json
from collections import defaultdict


def partition_data(partition, n_train, n_parties, train_label, beta=0.5, skew_class=2):
    """Split dataset into multiple clients with different partition strategies."""

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == 'noniid':
        min_size = 0
        min_require_size = 10
        K = int(train_label.max() + 1)

        N = n_train
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(train_label == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([
                    p * (len(idx_j) < N / n_parties)
                    for p, idx_j in zip(proportions, idx_batch)
                ])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]

                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        class_dis = np.zeros((n_parties, K))
        for j in range(n_parties):
            for m in range(K):
                class_dis[j, m] = int((np.array(train_label[idx_batch[j]]) == m).sum())
        print(class_dis.astype(int))

    elif partition.startswith("noniid-fixed"):
        """Each client is assigned a fixed number of classes."""

        try:
            fixed_k = int(partition.split("-")[-1])
        except:
            fixed_k = skew_class

        num_classes = int(train_label.max() + 1)

        assert num_classes == 100, f"Expected CIFAR-100 (100 classes), got {num_classes}"
        assert num_classes % fixed_k == 0, "num_classes must be divisible by fixed_k"

        num_groups = num_classes // fixed_k
        assert n_parties % num_groups == 0, "n_parties must be divisible by group number"

        classes = np.arange(num_classes)
        class_groups = np.array_split(classes, num_groups)

        cls_to_indices = defaultdict(list)
        for idx, label in enumerate(train_label):
            cls_to_indices[int(label)].append(idx)

        net_dataidx_map = {i: [] for i in range(n_parties)}

        for cid in range(n_parties):
            grp = class_groups[cid % num_groups]
            for c in grp:
                net_dataidx_map[cid].extend(cls_to_indices[int(c)])
            random.shuffle(net_dataidx_map[cid])

        class_dis = np.zeros((n_parties, num_classes), dtype=int)
        for cid in range(n_parties):
            y = np.array(train_label[net_dataidx_map[cid]])
            for m in np.unique(y):
                class_dis[cid, int(m)] = int((y == m).sum())
        print(class_dis)

    elif partition.startswith('noniid-skew'):
        """Each client contains a subset of classes."""

        skew_class = int(partition.split('-')[-1]) if partition != 'noniid-skew' else 2
        num_classes = int(train_label.max() + 1)

        num_cluster = num_classes / skew_class
        client_num_per_cluster = int(n_parties / num_cluster)

        assert num_classes % skew_class == 0
        assert n_parties % num_cluster == 0

        net_dataidx_map = {i: list() for i in range(n_parties)}

        label_idx = [[] for _ in range(num_classes)]
        for i in range(n_train):
            label_idx[int(train_label[i])].append(i)

        for i in range(n_parties):
            client_cluster_id = int(i // num_cluster)
            for j in range(skew_class):
                label = int(skew_class * (i % num_cluster) + j)
                sample_num_per_client = int(len(label_idx[label]) // client_num_per_cluster)

                net_dataidx_map[i] += label_idx[label][
                    sample_num_per_client * client_cluster_id:
                    sample_num_per_client * (1 + client_cluster_id)
                ]

            random.shuffle(net_dataidx_map[i])

    elif partition == 'noniid-skew2':
        """Predefined skew partition (CIFAR-10 style)."""

        net_dataidx_map = {i: list() for i in range(n_parties)}

        label_idx = [[] for _ in range(10)]
        for i in range(n_train):
            label_idx[int(train_label[i])].append(i)

        for i in range(10):
            net_dataidx_map[i] += label_idx[(i * 2) % 10][1250 * (i // 5):1250 * (i // 5 + 1)]
            net_dataidx_map[i] += label_idx[(i * 2) % 10 + 1][1250 * (i // 5):1250 * (i // 5 + 1)]
            random.shuffle(net_dataidx_map[i])

        for i in range(3):
            for j in range(5):
                net_dataidx_map[10 + i * 2] += label_idx[j][2500 + 500 * i:2500 + 500 * (i + 1)]
                net_dataidx_map[10 + i * 2 + 1] += label_idx[j + 5][2500 + 500 * i:2500 + 500 * (i + 1)]

        for i in range(4):
            for j in range(10):
                net_dataidx_map[16 + i] += label_idx[j][4000 + 250 * i:4000 + 250 * (i + 1)]

        for i in range(20):
            random.shuffle(net_dataidx_map[i])

    # statistics
    client_num_samples = np.array([len(net_dataidx_map[i]) for i in range(n_parties)])
    traindata_cls_counts = record_net_data_stats(train_label, net_dataidx_map)
    data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:, np.newaxis]

    return net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions


def record_net_data_stats(y_train, net_dataidx_map):
    """Compute per-client class distribution."""

    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    num_classes = int(y_train.max()) + 1

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)

        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_dict[net_i] = tmp

        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]

        net_cls_counts_npy = np.concatenate((net_cls_counts_npy, tmp_npy), axis=0)

    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1, num_classes))

    data_list = []
    for _, data in net_cls_counts_dict.items():
        n_total = sum(data.values())
        data_list.append(n_total)

    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts_dict))
    print(net_cls_counts_npy.astype(int))

    return net_cls_counts_npy
