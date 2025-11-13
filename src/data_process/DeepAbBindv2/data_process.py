#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/1/14 15:30
@project: LucaVirusTasks
@file: data_process
@desc: xxxx
"""
import random
import sys
import os.path
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../src")
try:
    from file_operator import *
    from utils import *
except ImportError as err:
    from src.file_operator import *
    from src.utils import *

dataset_name_list = [
    "DeepAbBindv2",
    "bind_dataset_lucappi_all_prots",
    "bind_dataset_lucappi_genome",
    "bind_dataset_lucappi_nucl"
]
new_dataset_name_list = [
    "DeepAbBindv2_original",
    "DeepAbBindv2_all_prots",
    "DeepAbBindv2_genome",
    "DeepAbBindv2_nucl"
]

dataset_name_2_dataset_type = {
    "DeepAbBindv2": "protein",
    "bind_dataset_lucappi_all_prots": "protein",
    "bind_dataset_lucappi_genome": "gene_protein",
    "bind_dataset_lucappi_nucl": "gene_protein"
}

new_dataset_name_2_dataset_type = {
    "DeepAbBindv2": "protein_protein_protein",
    "bind_dataset_lucappi_all_prots": "protein_protein_protein",
    "bind_dataset_lucappi_genome": "protein_protein_gene",
    "bind_dataset_lucappi_nucl": "protein_protein_gene"
}

global_prot_seq_2_prot_seq_id = {

}
global_gene_seq_2_gene_seq_id = {

}
for dataset_name_idx, dataset_name in enumerate(dataset_name_list):
    raw_dataset_dirpath = "../../../data/DeepAbBindv2/%s/%s/binary_class/" % (
        dataset_name, dataset_name_2_dataset_type[dataset_name]
    )

    save_dataset_dirpath = "../../../dataset/%s/%s/binary_class/" % (
        new_dataset_name_list[dataset_name_idx], new_dataset_name_2_dataset_type[dataset_name]
    )

    dataset_type_list = ["train", "dev", "test"]
    print("dataset_name: %s" % dataset_name)
    prot_fasta = []
    gene_fasta = []
    for dataset_type in dataset_type_list:
        if not os.path.exists(os.path.join(save_dataset_dirpath, dataset_type)):
            os.makedirs(os.path.join(save_dataset_dirpath, dataset_type))
        print("dataset_type: %s, input_type: %s" % (dataset_type, dataset_name_2_dataset_type[dataset_name]))
        new_dataset = []
        filepath = os.path.join(raw_dataset_dirpath, dataset_type, "%s.tsv" % dataset_type)
        seq_a_count_stats = []
        seq_a_len_stats = []
        seq_a_combined_len_stats = []

        seq_b_count_stats = []
        seq_b_len_stats = []
        seq_b_combined_len_stats = []

        label_stats = {}
        for row in tsv_reader(filepath, header=True, header_filter=True):
            ori_seq_id_a, ori_seq_id_b, ori_seq_type_a, ori_seq_type_b, ori_seq_a, ori_seq_b, label = row
            label = int(label)
            if dataset_name_2_dataset_type[dataset_name] == "protein":
                assert ori_seq_type_a in ["prot", "multi_prot"] and ori_seq_type_b in ["prot", "multi_prot"]
                seq_type_a = "prot"
                seq_type_b = "prot"
                seq_type_c = ori_seq_type_b
            elif dataset_name_2_dataset_type[dataset_name] == "gene_protein":
                assert ori_seq_type_a == "multi_prot" and ori_seq_type_b == "gene"
                seq_type_a = "prot"
                seq_type_b = "prot"
                seq_type_c = ori_seq_type_b
            else:
                raise Exception("filepath: %s" % filepath)

            seq_a_split_list = ori_seq_a.split(",")
            seq_a_count_stats.append(len(seq_a_split_list))
            seq_a_len_stats.extend([len(v) for v in seq_a_split_list])
            seq_a_combined_len_stats.append(sum([len(v) for v in seq_a_split_list]))
            assert len(seq_a_split_list) == 2

            seq_b_split_list = ori_seq_b.split(",")
            seq_b_count_stats.append(len(seq_b_split_list))
            seq_b_len_stats.extend([len(v) for v in seq_b_split_list])
            seq_b_combined_len_stats.append(sum([len(v) for v in seq_b_split_list]))
            if label not in label_stats:
                label_stats[label] = 0
            label_stats[label] += 1

            seq_a = seq_a_split_list[0].strip().upper()
            seq_b = seq_a_split_list[1].strip().upper()

            seq_c = ",".join(sorted(ori_seq_b.strip().upper().split(",")))
            if "gene" in seq_type_a:
                if seq_a not in global_gene_seq_2_gene_seq_id:
                    global_gene_seq_2_gene_seq_id[seq_a] = "single_gene_%d" % (len(global_gene_seq_2_gene_seq_id) + 1)
                    seq_id_a = global_gene_seq_2_gene_seq_id[seq_a]
                    gene_fasta.append([seq_id_a, seq_a])
                else:
                    seq_id_a = global_gene_seq_2_gene_seq_id[seq_a]
            else:
                if seq_a not in global_prot_seq_2_prot_seq_id:
                    global_prot_seq_2_prot_seq_id[seq_a] = "single_prot_%d" % (len(global_prot_seq_2_prot_seq_id) + 1)
                    seq_id_a = global_prot_seq_2_prot_seq_id[seq_a]
                    prot_fasta.append([seq_id_a, seq_a])
                else:
                    seq_id_a = global_prot_seq_2_prot_seq_id[seq_a]
            if "gene" in seq_type_b:
                if seq_b not in global_gene_seq_2_gene_seq_id:
                    global_gene_seq_2_gene_seq_id[seq_b] = "single_gene_%d" % (len(global_gene_seq_2_gene_seq_id) + 1)
                    seq_id_b = global_gene_seq_2_gene_seq_id[seq_b]
                    gene_fasta.append([seq_id_b, seq_b])
                else:
                    seq_id_b = global_gene_seq_2_gene_seq_id[seq_b]
            else:
                if seq_b not in global_prot_seq_2_prot_seq_id:
                    global_prot_seq_2_prot_seq_id[seq_b] = "single_prot_%d" % (len(global_prot_seq_2_prot_seq_id) + 1)
                    seq_id_b = global_prot_seq_2_prot_seq_id[seq_b]
                    prot_fasta.append([seq_id_b, seq_b])
                else:
                    seq_id_b = global_prot_seq_2_prot_seq_id[seq_b]
            if "gene" in seq_type_c:
                if seq_c not in global_gene_seq_2_gene_seq_id:
                    if seq_type_c == "gene":
                        global_gene_seq_2_gene_seq_id[seq_c] = "single_gene_%d" % (len(global_gene_seq_2_gene_seq_id) + 1)
                    else:
                        global_gene_seq_2_gene_seq_id[seq_c] = "multi_gene_%d" % (len(global_gene_seq_2_gene_seq_id) + 1)
                    seq_id_c = global_gene_seq_2_gene_seq_id[seq_c]
                    gene_fasta.append([seq_id_c, seq_c])
                else:
                    seq_id_c = global_gene_seq_2_gene_seq_id[seq_c]
            else:
                if seq_c not in global_prot_seq_2_prot_seq_id:
                    if seq_type_c == "prot":
                        global_prot_seq_2_prot_seq_id[seq_c] = "single_prot_%d" % (len(global_prot_seq_2_prot_seq_id) + 1)
                    else:
                        global_prot_seq_2_prot_seq_id[seq_c] = "multi_prot_%d" % (len(global_prot_seq_2_prot_seq_id) + 1)
                    seq_id_c = global_prot_seq_2_prot_seq_id[seq_c]
                    prot_fasta.append([seq_id_c, seq_c])
                else:
                    seq_id_c = global_prot_seq_2_prot_seq_id[seq_c]
            new_dataset.append([
                seq_id_a, seq_id_b, seq_id_c,
                seq_type_a, seq_type_b, seq_type_c,
                seq_a, seq_b, seq_c,
                label
            ])

        if dataset_type == "train":
            for _ in range(10):
                random.shuffle(new_dataset)

        print("dataset size: %d" % len(new_dataset))
        data_stats("seq_a_count_stats", seq_a_count_stats)
        data_stats("seq_a_len_stats", seq_a_len_stats)
        data_stats("seq_a_combined_len_stats", seq_a_combined_len_stats)
        data_stats("seq_b_count_stats", seq_b_count_stats)
        data_stats("seq_b_len_stats", seq_b_len_stats)
        data_stats("seq_b_combined_len_stats", seq_b_combined_len_stats)
        print("label_stats:")
        print(label_stats)
        csv_writer(
            new_dataset,
            handle=os.path.join(save_dataset_dirpath, dataset_type, "%s.csv" % dataset_type),
            header=[
                "seq_id_a", "seq_id_b", "seq_id_c",
                "seq_type_a", "seq_type_b", "seq_type_c",
                "seq_a", "seq_b", "seq_c",
                "label"
            ]
        )
        print("#" * 50)
    csv_writer(
        [[0], [1]],
        handle=os.path.join(save_dataset_dirpath, "label.txt"),
        header=[
            "label"
        ]
    )
    if len(gene_fasta) > 0:
        print("gene_fasta: %d" % len(gene_fasta))
        write_fasta(os.path.join(save_dataset_dirpath, "%s_all_gene.fasta" % dataset_name), gene_fasta)
    if len(prot_fasta) > 0:
        print("prot_fasta: %d" % len(prot_fasta))
        write_fasta(os.path.join(save_dataset_dirpath, "%s_all_prot.fasta" % dataset_name), prot_fasta)
    print("#" * 200)

if len(global_gene_seq_2_gene_seq_id) > 0:
    print("gene_fasta: %d" % len(global_gene_seq_2_gene_seq_id))
    global_single_gene_fasta = [[item[1], item[0]] for item in global_gene_seq_2_gene_seq_id.items() if item[1].startswith("single_gene")]
    global_multi_gene_fasta = [[item[1], item[0]] for item in global_gene_seq_2_gene_seq_id.items() if item[1].startswith("multi_gene")]
    print("global_single_gene_fasta: %d" % len(global_single_gene_fasta))
    print("global_multi_gene_fasta: %d" % len(global_multi_gene_fasta))
    write_fasta(os.path.join("../../../data/DeepAbBindv2/", "DeepAbBindv2_all_single_gene.fasta"), global_single_gene_fasta)
    write_fasta(os.path.join("../../../data/DeepAbBindv2/", "DeepAbBindv2_all_multi_gene.fasta"), global_multi_gene_fasta)
if len(global_prot_seq_2_prot_seq_id) > 0:
    print("prot_fasta: %d" % len(global_prot_seq_2_prot_seq_id))
    global_single_prot_fasta = [[item[1], item[0]] for item in global_prot_seq_2_prot_seq_id.items() if item[1].startswith("single_prot")]
    global_multi_prot_fasta = [[item[1], item[0]] for item in global_prot_seq_2_prot_seq_id.items() if item[1].startswith("multi_prot")]
    print("global_single_prot_fasta: %d" % len(global_single_prot_fasta))
    print("global_multi_prot_fasta: %d" % len(global_multi_prot_fasta))
    write_fasta(os.path.join("../../../data/DeepAbBindv2/", "DeepAbBindv2_all_single_prot.fasta"), global_single_prot_fasta)
    write_fasta(os.path.join("../../../data/DeepAbBindv2/", "DeepAbBindv2_all_multi_prot.fasta"), global_multi_prot_fasta)

"""
-----------
cd src/llm/lucavirus/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_single_gene.fasta \
    --save_path  /mnt/sanyuan.hy/matrices/DeepAbBindv2/lucavirus/v1.0/20240815023346/3800000 \
    --embedding_type matrix \
    --embedding_complete \
    --matrix_add_special_token \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0
    
cd src/llm/lucaone/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_single_gene.fasta \
    --save_path  /mnt2/sanyuan.hy/matrices/DeepAbBindv2/lucaone/v2.0/20231125113045/17600000 \
    --embedding_type matrix \
    --embedding_complete \
    --matrix_add_special_token \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 0

-----------
cd src/llm/lucavirus/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type multi_gene \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_multi_gene.fasta \
    --save_path  /mnt2/sanyuan.hy/matrices/DeepAbBindv2/lucavirus/v1.0/20240815023346/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 1
    
cd src/llm/lucaone/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type multi_gene \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_multi_gene.fasta \
    --save_path  /mnt2/sanyuan.hy/matrices/DeepAbBindv2/lucaone/v2.0/20231125113045/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 1

  
cd src/llm/lucavirus/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_single_prot.fasta \
    --save_path  /mnt/sanyuan.hy/matrices/DeepAbBindv2/lucavirus/v1.0/20240815023346/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2
    
cd src/llm/lucaone/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_single_prot.fasta \
    --save_path  /mnt/sanyuan.hy/matrices/DeepAbBindv2/lucaone/v2.0/20231125113045/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 1

cd src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_single_prot.fasta \
    --save_path  /mnt/sanyuan.hy/matrices/DeepAbBindv2/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 1
    
cd src/llm/esmc/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esmc \
    --llm_version 600M \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_single_prot.fasta \
    --save_path  /mnt2/sanyuan.hy/matrices/DeepAbBindv2/esm/esmc/600M \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 1
 
-----------   
cd src/llm/lucavirus/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type multi_prot \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_multi_prot.fasta \
    --save_path  /mnt/sanyuan.hy/matrices/DeepAbBindv2/lucavirus/v1.0/20240815023346/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3
    
cd src/llm/lucaone/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type multi_prot \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_multi_prot.fasta \
    --save_path  /mnt2/sanyuan.hy/matrices/DeepAbBindv2/lucaone/v2.0/20231125113045/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 3
    
cd src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type multi_prot \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_multi_prot.fasta \
    --save_path  /mnt/sanyuan.hy/matrices/DeepAbBindv2/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 3
"""


