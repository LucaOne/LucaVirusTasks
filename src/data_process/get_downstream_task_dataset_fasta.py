#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/1/15 20:21
@project: LucaOnePlusTasks
@file: get_downstream_task_dataset_fasta.py
@desc: get_downstream_task_dataset_fasta
"""
import sys
import os.path
import argparse
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from file_operator import csv_reader, write_fasta
except ImportError:
    from src.file_operator import csv_reader, write_fasta


parser = argparse.ArgumentParser("Running")
parser.add_argument(
    "--dataset_dir",
    default=None,
    type=str,
    required=True,
    help="dataset dir"
)
parser.add_argument(
    "--dataset_name",
    default=None,
    type=str,
    required=True,
    help="dataset name"
)
parser.add_argument(
    "--input_mode",
    default=None,
    type=str,
    required=True,
    choices=["single", "pair", "triple"],
    help="input mode"
)
parser.add_argument(
    "--dataset_type",
    default="protein",
    type=str,
    required=True,
    help="dataset type"
)
parser.add_argument(
    "--task_type",
    type=str,
    required=True,
    choices=[
        "multi_label",
        "multi_class",
        "binary_class",
        "regression"
    ],
    help="task type"
)

args = parser.parse_args()


if args.input_mode == "single":
    assert args.dataset_type in ["dna", "rna", "protein", "gene", "nucl"]
elif args.input_mode == "pair":
    strs = args.dataset_type.split("_")
    assert len(strs) == 2
    dataset_type_a = strs[0]
    dataset_type_b = strs[1]
    assert dataset_type_a in ["dna", "rna", "protein", "gene", "nucl"]
    assert dataset_type_b in ["dna", "rna", "protein", "gene", "nucl"]
elif args.input_mode == "triple":
    strs = args.dataset_type.split("_")
    assert len(strs) == 3
    dataset_type_a = strs[0]
    dataset_type_b = strs[1]
    dataset_type_c = strs[2]
    assert dataset_type_a in ["dna", "rna", "protein", "gene", "nucl"]
    assert dataset_type_b in ["dna", "rna", "protein", "gene", "nucl"]
    assert dataset_type_c in ["dna", "rna", "protein", "gene", "nucl"]


gene_seq_id_2_seq = {}
prot_seq_id_2_seq = {}

dataset_levels = ["train", "dev", "test"]

total_num = 0
for dataset_level in dataset_levels:
    dataset_dirpath = os.path.join(args.dataset_dir, args.dataset_name, args.dataset_type, args.task_type, dataset_level)
    if not os.path.exists(dataset_dirpath):
        print("dataset_dirpath=%s not exists.")
        continue
    for filename in os.listdir(dataset_dirpath):
        if not filename.endswith(".csv"):
            continue
        for row in csv_reader(os.path.join(dataset_dirpath, filename)):
            # pair
            if args.input_mode == "pair":
                seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b = row[0], row[1], \
                                                                           row[2].strip().lower(), row[3].strip().lower(),  \
                                                                           row[4].strip().upper(), row[5].strip().upper()
                if seq_type_a == "prot":
                    if seq_id_a not in prot_seq_id_2_seq:
                        prot_seq_id_2_seq[seq_id_a] = seq_a
                    else:
                        assert prot_seq_id_2_seq[seq_id_a] == seq_a
                else:
                    if seq_id_a not in gene_seq_id_2_seq:
                        gene_seq_id_2_seq[seq_id_a] = seq_a
                    else:
                        assert gene_seq_id_2_seq[seq_id_a] == seq_a
                if seq_type_b == "prot":
                    if seq_id_b not in prot_seq_id_2_seq:
                        prot_seq_id_2_seq[seq_id_b] = seq_b
                    else:
                        assert prot_seq_id_2_seq[seq_id_b] == seq_b
                else:
                    if seq_id_b not in gene_seq_id_2_seq:
                        gene_seq_id_2_seq[seq_id_b] = seq_b
                    else:
                        assert gene_seq_id_2_seq[seq_id_b] == seq_b

            elif args.input_mode == "triple":
                seq_id_a, seq_id_b, seq_id_c, seq_type_a, seq_type_b, seq_type_c, seq_a, seq_b, seq_c = row[0], row[1], row[2], \
                                                                                                        row[3].strip().lower(), row[4].strip().lower(), row[5].strip().lower(), \
                                                                                                        row[6].strip().upper(), row[7].strip().upper(), row[8].strip().upper()
                if seq_type_a == "prot":
                    if seq_id_a not in prot_seq_id_2_seq:
                        prot_seq_id_2_seq[seq_id_a] = seq_a
                    else:
                        assert prot_seq_id_2_seq[seq_id_a] == seq_a
                else:
                    if seq_id_a not in gene_seq_id_2_seq:
                        gene_seq_id_2_seq[seq_id_a] = seq_a
                    else:
                        assert gene_seq_id_2_seq[seq_id_a] == seq_a
                if seq_type_b == "prot":
                    if seq_id_b not in prot_seq_id_2_seq:
                        prot_seq_id_2_seq[seq_id_b] = seq_b
                    else:
                        assert prot_seq_id_2_seq[seq_id_b] == seq_b
                else:
                    if seq_id_b not in gene_seq_id_2_seq:
                        gene_seq_id_2_seq[seq_id_b] = seq_b
                    else:
                        assert gene_seq_id_2_seq[seq_id_b] == seq_b
                if seq_type_c == "prot":
                    if seq_id_c not in prot_seq_id_2_seq:
                        prot_seq_id_2_seq[seq_id_c] = seq_c
                    else:
                        assert prot_seq_id_2_seq[seq_id_c] == seq_c
                else:
                    if seq_id_c not in gene_seq_id_2_seq:
                        gene_seq_id_2_seq[seq_id_c] = seq_c
                    else:
                        assert gene_seq_id_2_seq[seq_id_c] == seq_c
            else:
                # single
                assert args.input_mode == "single"
                seq_id, seq_type, seq = row[0], row[1].strip().lower(), row[2].strip().upper()
                if seq_type == "prot":
                    if seq_id not in prot_seq_id_2_seq:
                        prot_seq_id_2_seq[seq_id] = seq
                    else:
                        if prot_seq_id_2_seq[seq_id] != seq:
                            print(seq_id, seq, prot_seq_id_2_seq[seq_id])
                        assert prot_seq_id_2_seq[seq_id] == seq
                else:
                    if seq_id not in gene_seq_id_2_seq:
                        gene_seq_id_2_seq[seq_id] = seq
                    else:
                        assert gene_seq_id_2_seq[seq_id] == seq
if len(gene_seq_id_2_seq) > 0:
    all_genes = [[item[0], item[1]] for item in gene_seq_id_2_seq.items()]
    print("dataset_name: %s, dataset_type: %s, task_type: %s, all_genes: %d" % (
        args.dataset_name,
        args.dataset_type,
        args.task_type,
        len(all_genes)
    ))
    write_fasta(
        os.path.join(args.dataset_dir, args.dataset_name, args.dataset_type, args.task_type, "all_genes.fasta"),
        all_genes
    )
if len(prot_seq_id_2_seq) > 0:
    all_prots = [[item[0], item[1]] for item in prot_seq_id_2_seq.items()]
    print("dataset_name: %s, dataset_type: %s, task_type: %s, all_prots: %d" % (
        args.dataset_name,
        args.dataset_type,
        args.task_type,
        len(all_prots)
    ))
    write_fasta(
        os.path.join(args.dataset_dir, args.dataset_name, args.dataset_type, args.task_type, "all_proteins.fasta"),
        all_prots
    )
'''

python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name CentralDogma  \
    --dataset_type gene_protein \
    --input_mode pair \
    --task_type binary_class
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name SupKTax  \
    --dataset_type gene \
    --input_mode single \
    --task_type multi_class
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name GenusTax  \
    --dataset_type gene \
    --input_mode single \
    --task_type multi_class
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name SpeciesTax  \
    --dataset_type gene \
    --input_mode single \
    --task_type multi_class
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name ncRNAFam  \
    --dataset_type gene \
    --input_mode single \
    --task_type multi_class 
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name ProtStab  \
    --dataset_type protein \
    --input_mode single \
    --task_type regression
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name ProtLoc \
    --dataset_type protein \
    --input_mode single \
    --task_type multi_class
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name InfA \
    --dataset_type gene_gene \
    --input_mode pair \
    --task_type binary_class
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name ncRPI \
    --dataset_type gene_protein \
    --input_mode pair \
    --task_type binary_class
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name PPI \
    --dataset_type protein_protein \
    --input_mode pair \
    --task_type binary_class
    
# lucavirus tasks 
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name DeepAbBindv2_genome \
    --dataset_type protein_protein_gene \
    --input_mode triple \
    --task_type binary_class
dataset_name: DeepAbBindv2_genome, dataset_type: protein_protein_gene, task_type: binary_class, all_genes: 60
dataset_name: DeepAbBindv2_genome, dataset_type: protein_protein_gene, task_type: binary_class, all_prots: 57569
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name DeepAbBindv2_nucl \
    --dataset_type protein_protein_gene \
    --input_mode triple \
    --task_type binary_class
dataset_name: DeepAbBindv2_nucl, dataset_type: protein_protein_gene, task_type: binary_class, all_genes: 58
dataset_name: DeepAbBindv2_nucl, dataset_type: protein_protein_gene, task_type: binary_class, all_prots: 57569
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name DeepAbBindv2_original \
    --dataset_type protein_protein_protein \
    --input_mode triple \
    --task_type binary_class
dataset_name: DeepAbBindv2_original, dataset_type: protein_protein_protein, task_type: binary_class, all_prots: 57627
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name DMS_Bind_Reps_Strain \
    --dataset_type protein \
    --input_mode single \
    --task_type regression
dataset_name: DMS_Bind_Reps_Strain, dataset_type: protein, task_type: regression, all_prots: 19878
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name DMS_Bind_Reps_Strain_Nucl \
    --dataset_type gene \
    --input_mode single \
    --task_type regression
dataset_name: DMS_Bind_Reps_Strain_Nucl, dataset_type: gene, task_type: regression, all_genes: 19878
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name RdRP \
    --dataset_type protein \
    --input_mode single \
    --task_type binary_class
dataset_name: RdRP, dataset_type: protein, task_type: binary_class, all_prots: 235413
    
python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name ViralCapsid \
    --dataset_type protein \
    --input_mode single \
    --task_type binary_class
dataset_name: ViralCapsid, dataset_type: protein, task_type: binary_class, all_prots: 433478

python get_downstream_task_dataset_fasta.py \
    --dataset_dir ../../dataset/\
    --dataset_name VirusEC4 \
    --dataset_type protein \
    --input_mode single \
    --task_type multi_label
dataset_name: VirusEC4, dataset_type: protein, task_type: multi_label, all_prots: 113684
'''



