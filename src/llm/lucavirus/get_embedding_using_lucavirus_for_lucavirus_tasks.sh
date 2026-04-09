# 57机器
# RdRP
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
    --input_file ../../../dataset/RdRP/protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/RdRP/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0

# 57机器
# ViralCapsid
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
    --input_file ../../../dataset/ViralCapsid/protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/ViralCapsid/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 1

# 57机器
# VirusEC4
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
    --input_file ../../../dataset/VirusEC4/protein/multi_label/all_proteins.fasta \
    --save_path ../../../../matrices/VirusEC4/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2


# 129机器
# DMS_Bind_Reps_Strain
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
    --input_file ../../../dataset/DMS_Bind_Reps_Strain/protein/regression/all_proteins.fasta \
    --save_path ../../../../matrices/DMS_Bind_Reps_Strain/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3


# 142机器
# DMS_Bind_Reps_Strain_Nucl
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
    --input_file ../../../dataset/DMS_Bind_Reps_Strain_Nucl/gene/regression/all_genes.fasta \
    --save_path ../../../../matrices/DMS_Bind_Reps_Strain_Nucl/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0


# 255 机器
# DeepAbBindv2_genome
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
    --input_file ../../../dataset/DeepAbBindv2_genome/protein_protein_gene/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# 255 机器
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
    --input_file ../../../dataset/DeepAbBindv2_genome/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# 255机器
# DeepAbBindv2_nucl
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
    --input_file ../../../dataset/DeepAbBindv2_nucl/protein_protein_gene/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# 255机器
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
    --input_file ../../../dataset/DeepAbBindv2_nucl/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# 255机器
# DeepAbBindv2_original
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
    --input_file ../../../dataset/DeepAbBindv2_original/protein_protein_protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_original/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3


# 122机器
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
    --input_file ../../../dataset/DMS_EVA/protein/regression/all_proteins.fasta \
    --save_path ../../../../matrices/DMS_EVA/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

