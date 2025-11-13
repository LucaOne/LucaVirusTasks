#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/11/26 15:24
@project: LucaVirusTasks
@file: dataset_process_deepvirid_nucl
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
    "../../../data/DeepVirID_full/gene_protein/binary_class/train/train.csv",
    "../../../data/DeepVirID_full/gene_protein/binary_class/dev/dev.csv"
]
data = {}
seq_ids = set()
fasta = []
seq_len_stats = []
for data_filepath in data_filepath_list:
    print("process filename: %s" % data_filepath)
    for row in csv_reader(data_filepath):
        seq_id, seq_type, seq, label = row
        label = int(label)
        if seq_type != "gene":
            continue
        if label not in data:
            data[label] = []
        if seq_id[0] == ">":
            seq_id = seq_id[1:]
        seq = seq.strip().upper()
        data[label].append([seq_id, seq_type, seq, label])
        seq_ids.add(seq_id)
        fasta.append([seq_id, seq])
        seq_len_stats.append(len(seq))
print("seq_ids: %d, seqs: %d" % (len(seq_ids), len(fasta)))
print("total: %d, positive: %d, negative: %d, rate: %f" % (len(data[0]) + len(data[1]), len(data[1]), len(data[0]), len(data[0])/len(data[1])))

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

dev_rate = 0.1
train_set = []
dev_set = []
test_set = []
for label in [0, 1]:
    print("%s" % ("Negative" if label == 0 else "Positive"))
    cur_data = data[label]
    for _ in range(10):
        random.shuffle(cur_data)
    dev_num = int(len(cur_data) * dev_rate)
    dev = cur_data[0: dev_num]
    test = cur_data[dev_num: dev_num + dev_num]
    train = cur_data[dev_num + dev_num:]
    train_set.extend(train)
    dev_set.extend(dev)
    test_set.extend(test)
    print("total: %d, train: %d, dev: %d, test: %d" % (len(cur_data), len(train), len(dev), len(test)))
print("total: %d, train: %d, dev: %d, test: %d" % (len(train_set) + len(dev_set) + len(test_set), len(train_set), len(dev_set), len(test_set)))

for _ in range(10):
    random.shuffle(train_set)

for dataset_type in ["train", "dev", "test"]:
    if not os.path.exists("../../../dataset/DeepVirID/gene/binary_class/%s" % dataset_type):
        os.makedirs("../../../dataset/DeepVirID/gene/binary_class/%s" % dataset_type)
csv_writer(
    dataset=train_set,
    handle="../../../dataset/DeepVirID/gene/binary_class/train/train.csv",
    header=["seq_id", "seq_type", "seq", "label"]
)

csv_writer(
    dataset=dev_set,
    handle="../../../dataset/DeepVirID/gene/binary_class/dev/dev.csv",
    header=["seq_id", "seq_type", "seq", "label"]
)

csv_writer(
    dataset=test_set,
    handle="../../../dataset/DeepVirID/gene/binary_class/test/test.csv",
    header=["seq_id", "seq_type", "seq", "label"]
)

csv_writer(
    dataset=[[0], [1]],
    handle="../../../dataset/DeepVirID/gene/binary_class/label.txt",
    header=["label"]
)

write_fasta("../../../dataset/DeepVirID/gene/binary_class/DeepVirID_all_nucl.fasta", fasta)


if not os.path.exists("../../../dataset/DeepVirID/gene/binary_class/fasta_split"):
    os.makedirs("../../../dataset/DeepVirID/gene/binary_class/fasta_split")

split_num = 4
split_per_size = (len(fasta) + split_num - 1)//split_num
for idx in range(split_num):
    begin = idx * split_per_size
    end = min(len(fasta), (idx + 1) * split_per_size)
    cur_fasta = fasta[begin: end]
    print("idx: %d, size: %d" % (idx + 1, len(cur_fasta)))
    write_fasta(
        "../../../dataset/DeepVirID/gene/binary_class/fasta_split/DeepVirID_all_nucl_part_%02d_of_%02d.fasta" % (idx + 1, split_num),
        cur_fasta
    )

independent_test_filepath = "../../../data/DeepVirID_full/gene_protein/binary_class/test/test.csv"
independent_test_data = {}
independent_test_seq_ids = set()
independent_test_fasta = []
independent_test = []
independent_test_seq_len_stats = []
print("process filename: %s" % independent_test_filepath)
for row in csv_reader(independent_test_filepath):
    seq_id, seq_type, seq, label = row
    label = int(label)
    if seq_type != "gene":
        continue
    if label not in independent_test_data:
        independent_test_data[label] = []
    independent_test_data[label].append([seq_id, seq_type, seq, label])
    independent_test.append([seq_id, seq_type, seq, label])
    independent_test_seq_ids.add(seq_id)
    independent_test_fasta.append([seq_id, seq])
    independent_test_seq_len_stats.append(len(seq))

if len(independent_test_fasta) > 0:
    print("independent_seq_ids: %d, independent_seqs: %d" % (len(independent_test_seq_ids), len(independent_test_fasta)))
    print("total: %d, positive: %d, negative: %d, rate: %f" % (
        (len(independent_test_data[0]) if 0 in independent_test_data else 0) + (len(independent_test_data[1]) if 1 in independent_test_data else 0),
        len(independent_test_data[1]) if 1 in independent_test_data else 0,
        len(independent_test_data[0]) if 0 in independent_test_data else 0,
        len(independent_test_data[0]) if 0 in independent_test_data else 0 / (len(independent_test_data[1]) if 1 in independent_test_data else 1e-10)
    ))
    print("seq len stats:")
    print("min: %d, max: %d, mean: %f, median: %d, 25: %d, 45: %d, 60: %d, 75: %d, 80: %d, 85: %d, 90: %d, 95: %d, 99: %d" %(
        np.min(independent_test_seq_len_stats),
        np.max(independent_test_seq_len_stats),
        np.mean(independent_test_seq_len_stats),
        np.median(independent_test_seq_len_stats),
        np.percentile(independent_test_seq_len_stats, 25),
        np.percentile(independent_test_seq_len_stats, 45),
        np.percentile(independent_test_seq_len_stats, 60),
        np.percentile(independent_test_seq_len_stats, 75),
        np.percentile(independent_test_seq_len_stats, 80),
        np.percentile(independent_test_seq_len_stats, 85),
        np.percentile(independent_test_seq_len_stats, 90),
        np.percentile(independent_test_seq_len_stats, 95),
        np.percentile(independent_test_seq_len_stats, 99)
    ))
    csv_writer(
        dataset=test_set,
        handle="../../../dataset/DeepVirID/gene/binary_class/independent_test/independent_test.csv",
        header=["seq_id", "seq_type", "seq", "label"]
    )
    write_fasta("../../../dataset/DeepVirID/gene/binary_class/independent_test_nucl.fasta", independent_test_fasta)

"""
process filename: ../../../data/DeepVirID_full/gene_protein/binary_class/train/train.csv
process filename: ../../../data/DeepVirID_full/gene_protein/binary_class/dev/dev.csv
seq_ids: 4059977, seqs: 4059977
total: 4059977, positive: 130967, negative: 3929010, rate: 30.000000
seq len stats:
min: 1000, max: 3072, mean: 1983.739257, median: 1994, 25: 1342, 45: 1903, 60: 2200, 75: 2588, 80: 2679, 85: 2812, 90: 2947, 95: 3003, 99: 3018
Negative
total: 3929010, train: 3143208, dev: 392901, test: 392901
Positive
total: 130967, train: 104775, dev: 13096, test: 13096
total: 4059977, train: 3247983, dev: 405997, test: 405997
process filename: ../../../data/DeepVirID_full/gene_protein/binary_class/test/test.csv


process filename: ../../../data/DeepVirID_full/gene_protein/binary_class/train/train.csv
process filename: ../../../data/DeepVirID_full/gene_protein/binary_class/dev/dev.csv
seq_ids: 4059977, seqs: 4059977
total: 4059977, positive: 130967, negative: 3929010, rate: 30.000000
seq len stats:
min: 1000, max: 3072, mean: 1983.739257, median: 1994, 25: 1342, 45: 1903, 60: 2200, 75: 2588, 80: 2679, 85: 2812, 90: 2947, 95: 3003, 99: 3018
Negative
total: 3929010, train: 3143208, dev: 392901, test: 392901
Positive
total: 130967, train: 104775, dev: 13096, test: 13096
total: 4059977, train: 3247983, dev: 405997, test: 405997
idx: 1, size: 1014995
idx: 2, size: 1014995
idx: 3, size: 1014995
idx: 4, size: 1014992
process filename: ../../../data/DeepVirID_full/gene_protein/binary_class/test/test.csv
"""
