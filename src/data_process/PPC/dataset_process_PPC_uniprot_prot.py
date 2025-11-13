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
    "../../../data/PPC_uniprot/protein/binary_class/train/train.csv",
    "../../../data/PPC_uniprot/protein/binary_class/dev/dev.csv",
    "../../../data/PPC_uniprot/protein/binary_class/test/test.csv"
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
    if not os.path.exists("../../../dataset/PPC_uniprot/protein/binary_class/%s" % dataset_type):
        os.makedirs("../../../dataset/PPC_uniprot/protein/binary_class/%s" % dataset_type)
csv_writer(
    dataset=dataset[0],
    handle="../../../dataset/PPC_uniprot/protein/binary_class/train/train.csv",
    header=["seq_id", "seq_type", "seq", "label"]
)

csv_writer(
    dataset=dataset[1],
    handle="../../../dataset/PPC_uniprot/protein/binary_class/dev/dev.csv",
    header=["seq_id", "seq_type", "seq", "label"]
)

csv_writer(
    dataset=dataset[2],
    handle="../../../dataset/PPC_uniprot/protein/binary_class/test/test.csv",
    header=["seq_id", "seq_type", "seq", "label"]
)

csv_writer(
    dataset=[[0], [1]],
    handle="../../../dataset/PPC_uniprot/protein/binary_class/label.txt",
    header=["label"]
)

write_fasta("../../../dataset/PPC_uniprot/protein/binary_class/PPC_uniprot_all_prot.fasta", fasta)

if not os.path.exists("../../../dataset/PPC_uniprot/protein/binary_class/fasta_split"):
    os.makedirs("../../../dataset/PPC_uniprot/protein/binary_class/fasta_split")

split_num = 4
split_per_size = (len(fasta) + split_num - 1)//split_num
for idx in range(split_num):
    begin = idx * split_per_size
    end = min(len(fasta), (idx + 1) * split_per_size)
    cur_fasta = fasta[begin: end]
    print("idx: %d, size: %d" % (idx + 1, len(cur_fasta)))
    write_fasta(
        "../../../dataset/PPC_uniprot/protein/binary_class/fasta_split/PPC_uniprot_all_prot_part_%02d_of_%02d.fasta" % (idx + 1, split_num),
        cur_fasta
    )


"""
process filename: ../../../data/PPC_uniprot/protein/binary_class/train/train.csv
process filename: ../../../data/PPC_uniprot/protein/binary_class/dev/dev.csv
process filename: ../../../data/PPC_uniprot/protein/binary_class/test/test.csv
seq_ids: 206214, seqs: 206214
positive num stats:
min: 1, max: 15, mean: 1.068885, median: 1, 25: 1, 45: 1, 60: 1, 75: 1, 80: 1, 85: 1, 90: 1, 95: 1, 99: 3
positive rate stats:
min: 0, max: 0, mean: 0.002241, median: 0, 25: 0, 45: 0, 60: 0, 75: 0, 80: 0, 85: 0, 90: 0, 95: 0, 99: 0
seq len stats:
min: 41, max: 7182, mean: 612.393237, median: 566, 25: 561, 45: 566, 60: 568, 75: 850, 80: 854, 85: 858, 90: 862, 95: 869,99: 1353
idx: 1, size: 51554
idx: 2, size: 51554
idx: 3, size: 51554
idx: 4, size: 51552
"""