# 64 机器
# RdRP
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/RdRP/protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/RdRP/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0

# 129 机器
# ViralCapsid
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/ViralCapsid/protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/ViralCapsid/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 1

# 255 机器
# VirusEC4
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/VirusEC4/protein/multi_label/all_proteins.fasta \
    --save_path ../../../../matrices/VirusEC4/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 173 机器
# DeepAbBindv2_genome
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DeepAbBindv2_genome/protein_protein_gene/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 173 机器
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DeepAbBindv2_genome/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 173 机器
# DeepAbBindv2_nucl
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DeepAbBindv2_nucl/protein_protein_gene/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 173 机器
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DeepAbBindv2_nucl/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 173 机器
# DeepAbBindv2_original
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DeepAbBindv2_original/protein_protein_protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_original/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 64 机器
# DMS_Bind_Reps_Strain
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DMS_Bind_Reps_Strain/protein/regression/all_proteins.fasta \
    --save_path ../../../../matrices/DMS_Bind_Reps_Strain/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# 142 机器
# DMS_Bind_Reps_Strain_Nucl
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DMS_Bind_Reps_Strain_Nucl/gene/regression/all_genes.fasta \
    --save_path ../../../../matrices/DMS_Bind_Reps_Strain_Nucl/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 1

# 122机器
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level\
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DMS_EVA/protein/regression/all_proteins.fasta \
    --save_path ../../../../matrices/DMS_EVA/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 1

