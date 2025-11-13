#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/12/3 13:43
@project: LucaVirusTasks
@file: split
@desc: xxxx
"""
import os, sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import csv_reader, csv_writer, write_fasta, fasta_reader
except ImportError:
    from src.file_operator import csv_reader, csv_writer, write_fasta, fasta_reader

fasta_filepath = "../../../dataset/DeepVirID_4_gpu/protein/binary_class/DeepVirID_4_gpu_all_prot.fasta"
fasta = []
for row in fasta_reader(fasta_filepath):
    seq_id = row[0]
    seq = row[1]
    if seq_id[0] == ">":
        seq_id = seq_id[1:]
    if seq != seq.strip().upper():
        print(row)
    fasta.append([seq_id, seq])


if not os.path.exists("../../../dataset/DeepVirID_4_gpu/protein/binary_class/fasta_split"):
    os.makedirs("../../../dataset/DeepVirID_4_gpu/protein/binary_class/fasta_split")

split_num = 4
split_per_size = (len(fasta) + split_num - 1)//split_num
for idx in range(split_num):
    begin = idx * split_per_size
    end = min(len(fasta), (idx + 1) * split_per_size)
    cur_fasta = fasta[begin: end]
    print("idx: %d, size: %d" % (idx + 1, len(cur_fasta)))
    write_fasta(
        "../../../dataset/DeepVirID_4_gpu/protein/binary_class/fasta_split/DeepVirID_4_gpu_all_prot_part_%02d_of_%02d.fasta" % (idx + 1, split_num),
        cur_fasta
    )

fasta_filepath = "../../../dataset/DeepVirID_4_gpu/protein/binary_class/DeepVirID_4_gpu_all_nucl.fasta"
fasta = []
for row in fasta_reader(fasta_filepath):
    seq_id = row[0]
    seq = row[1]
    if seq_id[0] == ">":
        seq_id = seq_id[1:]
    if seq != seq.strip().upper():
        print(row)
    fasta.append([seq_id, seq])


if not os.path.exists("../../../dataset/DeepVirID_4_gpu/gene/binary_class/fasta_split"):
    os.makedirs("../../../dataset/DeepVirID_4_gpu/gene/binary_class/fasta_split")

split_num = 4
split_per_size = (len(fasta) + split_num - 1)//split_num
for idx in range(split_num):
    begin = idx * split_per_size
    end = min(len(fasta), (idx + 1) * split_per_size)
    cur_fasta = fasta[begin: end]
    print("idx: %d, size: %d" % (idx + 1, len(cur_fasta)))
    write_fasta(
        "../../../dataset/DeepVirID_4_gpu/gene/binary_class/fasta_split/DeepVirID_4_gpu_all_nucl_part_%02d_of_%02d.fasta" % (idx + 1, split_num),
        cur_fasta
    )