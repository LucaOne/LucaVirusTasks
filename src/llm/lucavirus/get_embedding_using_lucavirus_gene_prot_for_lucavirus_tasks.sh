# 57机器
# RdRP
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/RdRP/protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/RdRP/lucavirus-prot/v1.0/3800000 \
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
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/ViralCapsid/protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/ViralCapsid/lucavirus-prot/v1.0/3800000 \
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
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/VirusEC4/protein/multi_label/all_proteins.fasta \
    --save_path ../../../../matrices/VirusEC4/lucavirus-prot/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 57机器
# DeepAbBindv2_genome
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DeepAbBindv2_genome/protein_protein_gene/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus-prot/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# 57机器
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-gene \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250118234004 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DeepAbBindv2_genome/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus-gene/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# 173机器
# DeepAbBindv2_nucl
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DeepAbBindv2_nucl/protein_protein_gene/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus-prot/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0

# 173机器
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-gene \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250118234004 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DeepAbBindv2_nucl/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus-gene/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 1

# 173机器
# DeepAbBindv2_original
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DeepAbBindv2_original/protein_protein_protein/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_original/lucavirus-prot/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# 173机器
# DMS_Bind_Reps_Strain
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DMS_Bind_Reps_Strain/protein/regression/all_proteins.fasta \
    --save_path ../../../../matrices/DMS_Bind_Reps_Strain/lucavirus-prot/v1.0/3800000 \
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
    --llm_type lucavirus-gene \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250118234004 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DMS_Bind_Reps_Strain_Nucl/gene/regression/all_genes.fasta \
    --save_path ../../../../matrices/DMS_Bind_Reps_Strain_Nucl/lucavirus-gene/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0


# 122机器
cd ./src/llm/lucavirus
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DMS_EVA/protein/regression/all_proteins.fasta \
    --save_path ../../../../matrices/DMS_EVA/lucavirus-prot/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0