# 64机器
# CentralDogma
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/CentralDogma/gene_protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/CentralDogma/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0

# 64机器
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/CentralDogma/gene_protein/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/CentralDogma/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 1

# 64机器
# SupKTax
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/SupKTax/gene/multi_class/all_genes.fasta \
    --save_path ../../../../matrices/SupKTax/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 64机器
# GenusTax
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/GenusTax/gene/multi_class/all_genes.fasta \
    --save_path ../../../../matrices/GenusTax/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# 255机器
# SpeciesTax
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/SpeciesTax/gene/multi_class/all_genes.fasta \
    --save_path ../../../../matrices/SpeciesTax/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0

# 255机器
# ncRNAFam
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/ncRNAFam/gene/multi_class/all_genes.fasta \
    --save_path ../../../../matrices/ncRNAFam/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 1

# 255机器
# ProtStab
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/ProtStab/protein/regression/all_proteins.fasta \
    --save_path ../../../../matrices/ProtStab/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 255机器
# ProtLoc
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/ProtLoc/protein/multi_class/all_proteins.fasta \
    --save_path ../../../../matrices/ProtLoc/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# 129机器
# InfA
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/InfA/gene_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/InfA/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0

# 129机器
# ncRPI
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/ncRPI/gene_protein/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/ncRPI/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 1

# 129机器
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/ncRPI/gene_protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/ncRPI/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 129机器
# PPI
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/PPI/protein_protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/PPI/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3