#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/11/26 15:24
@project: LucaVirusTasks
@file: dataset_process_PPC_prot
@desc: xxxx
"""
import numpy as np
import random
import os, sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import csv_reader, csv_writer, write_fasta
except ImportError:
    from src.file_operator import csv_reader, csv_writer, write_fasta

random.seed(1234)
data_filepath_list = [
    "../../../data/PPC/protein/binary_class/train/train.csv",
    "../../../data/PPC/protein/binary_class/dev/dev.csv",
    "../../../data/PPC/protein/binary_class/test/test.csv"
]
dataset = [[], [], []]
seq_ids = set()
fasta = []
positive_num = []
positive_rate = []
seq_len_stats = []
for f_idx, data_filepath in enumerate(data_filepath_list):
    print("process filename: %s" % data_filepath)
    for row in csv_reader(data_filepath):
        seq_id, seq_type, seq, label = row
        seq_len = len(seq)
        label = eval(label)
        label_len = len(label)
        cur_positive_num = sum(label)
        cur_positive_rate = cur_positive_num/label_len
        positive_num.append(cur_positive_num)
        positive_rate.append(cur_positive_rate)
        if seq_type != "prot":
            print(row)
            print(1/0)
        assert label_len == seq_len
        assert seq_type == "prot"

        dataset[f_idx].append([seq_id, seq_type, seq, label])
        seq_ids.add(seq_id)
        fasta.append([seq_id, seq])
        seq_len_stats.append(seq_len)
print("seq_ids: %d, seqs: %d" % (len(seq_ids), len(fasta)))

print("positive num stats:")
print("min: %d, max: %d, mean: %f, median: %d, 25: %d, 45: %d, 60: %d, 75: %d, 80: %d, 85: %d, 90: %d, 95: %d, 99: %d" %(
    np.min(positive_num),
    np.max(positive_num),
    np.mean(positive_num),
    np.median(positive_num),
    np.percentile(positive_num, 25),
    np.percentile(positive_num, 45),
    np.percentile(positive_num, 60),
    np.percentile(positive_num, 75),
    np.percentile(positive_num, 80),
    np.percentile(positive_num, 85),
    np.percentile(positive_num, 90),
    np.percentile(positive_num, 95),
    np.percentile(positive_num, 99)
))

print("positive rate stats:")
print("min: %d, max: %d, mean: %f, median: %d, 25: %d, 45: %d, 60: %d, 75: %d, 80: %d, 85: %d, 90: %d, 95: %d, 99: %d" %(
    np.min(positive_rate),
    np.max(positive_rate),
    np.mean(positive_rate),
    np.median(positive_rate),
    np.percentile(positive_rate, 25),
    np.percentile(positive_rate, 45),
    np.percentile(positive_rate, 60),
    np.percentile(positive_rate, 75),
    np.percentile(positive_rate, 80),
    np.percentile(positive_rate, 85),
    np.percentile(positive_rate, 90),
    np.percentile(positive_rate, 95),
    np.percentile(positive_rate, 99)
))

print("seq len stats:")
print("min: %d, max: %d, mean: %f, median: %d, 25: %d, 45: %d, 60: %d, 75: %d, 80: %d, 85: %d, 90: %d, 95: %d, 99: %d" %(
    np.min(seq_len_stats),
    np.max(seq_len_stats),
    np.mean(seq_len_stats),
    np.median(seq_len_stats),
    np.percentile(seq_len_stats, 25),
    np.percentile(seq_len_stats, 45),
    np.percentile(seq_len_stats, 60),
    np.percentile(seq_len_stats, 75),
    np.percentile(seq_len_stats, 80),
    np.percentile(seq_len_stats, 85),
    np.percentile(seq_len_stats, 90),
    np.percentile(seq_len_stats, 95),
    np.percentile(seq_len_stats, 99)
))

for _ in range(10):
    random.shuffle(dataset[0])

for dataset_type in ["train", "dev", "test"]:
    if not os.path.exists("../../../dataset/PPC/protein/binary_class/%s" % dataset_type):
        os.makedirs("../../../dataset/PPC/protein/binary_class/%s" % dataset_type)
csv_writer(
    dataset=dataset[0],
    handle="../../../dataset/PPC/protein/binary_class/train/train.csv",
    header=["seq_id", "seq_type", "seq", "label"]
)

csv_writer(
    dataset=dataset[1],
    handle="../../../dataset/PPC/protein/binary_class/dev/dev.csv",
    header=["seq_id", "seq_type", "seq", "label"]
)

csv_writer(
    dataset=dataset[2],
    handle="../../../dataset/PPC/protein/binary_class/test/test.csv",
    header=["seq_id", "seq_type", "seq", "label"]
)

csv_writer(
    dataset=[[0], [1]],
    handle="../../../dataset/PPC/protein/binary_class/label.txt",
    header=["label"]
)

write_fasta("../../../dataset/PPC/protein/binary_class/ppc_all_prot.fasta", fasta)

"""
seq_ids: 17279, seqs: 17279
positive num stats:
min: 0, max: 29, mean: 6.922854, median: 6, 25: 2, 45: 4, 60: 8, 75: 12, 80: 14, 85: 14, 90: 16, 95: 18, 99: 20
positive rate stats:
min: 0, max: 0, mean: 0.005038, median: 0, 25: 0, 45: 0, 60: 0, 75: 0, 80: 0, 85: 0, 90: 0, 95: 0, 99: 0
seq len stats:
min: 27, max: 7182, mean: 1325.485040, median: 1189, 25: 631, 45: 1096, 60: 1425, 75: 1925, 80: 2078, 85: 2222, 90: 2426, 95: 2834, 99: 3086
"""