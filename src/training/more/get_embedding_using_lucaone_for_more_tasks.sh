# 122机器
# HIV_Entry_Latent_Fold_00
cd ./src/llm/lucaone
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone\
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/HIV_Entry_Latent_Fold_00/protein/regression/all_seqs.fasta \
    --save_path ../../../../matrices/HIV_Entry_Latent_Fold_00/lucaone/v2.0/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10242 \
    --gpu_id 1

# HIV_Entry_Observed_Fold_00
cd ./src/llm/lucaone
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone\
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/HIV_Entry_Observed_Fold_00/protein/regression/all_seqs.fasta \
    --save_path ../../../../matrices/HIV_Entry_Observed_Fold_00/lucaone/v2.0/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10242 \
    --gpu_id 1

# HIV_immune_escape_10_1074_Fold_00
cd ./src/llm/lucaone
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone\
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/HIV_immune_escape_10_1074_Fold_00/protein/regression/all_seqs.fasta \
    --save_path ../../../../matrices/HIV_immune_escape_10_1074_Fold_00/lucaone/v2.0/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10242 \
    --gpu_id 1

# HIV_immune_escape_3BNC117_Fold_00
cd ./src/llm/lucaone
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone\
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/HIV_immune_escape_3BNC117_Fold_00/protein/regression/all_seqs.fasta \
    --save_path ../../../../matrices/HIV_immune_escape_3BNC117_Fold_00/lucaone/v2.0/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10242 \
    --gpu_id 1

# HIV_immune_escape_sera_mean_Fold_00
cd ./src/llm/lucaone
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python get_embedding.py \
  --llm_dir ../../..  \
  --llm_type lucaone\
  --llm_version v2.0 \
  --llm_task_level token_level,span_level,seq_level,structure_level \
  --llm_time_str 20231125113045 \
  --llm_step 17600000 \
  --truncation_seq_length 10240 \
  --trunc_type right \
  --seq_type prot \
  --input_file ../../../dataset/HIV_immune_escape_sera_mean_Fold_00/protein/regression/all_seqs.fasta \
  --save_path ../../../../matrices/HIV_immune_escape_sera_mean_Fold_00/lucaone/v2.0/17600000 \
  --embedding_type matrix \
  --matrix_add_special_token \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --embedding_fixed_len_a_time 10242 \
  --gpu_id 1


# HK19_immune_escape_per_serum
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python get_embedding.py \
  --llm_dir ../../..  \
  --llm_type lucaone\
  --llm_version v2.0 \
  --llm_task_level token_level,span_level,seq_level,structure_level \
  --llm_time_str 20231125113045 \
  --llm_step 17600000 \
  --truncation_seq_length 10240 \
  --trunc_type right \
  --seq_type prot \
  --input_file ../../../dataset/HK19_immune_escape_per_serum/protein/regression/all_seqs.fasta \
  --save_path ../../../../matrices/HK19_immune_escape_per_serum/lucaone/v2.0/17600000 \
  --embedding_type matrix \
  --matrix_add_special_token \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --embedding_fixed_len_a_time 10242 \
  --gpu_id 1


# PE09_immune_escape_per_serum
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python get_embedding.py \
  --llm_dir ../../..  \
  --llm_type lucaone\
  --llm_version v2.0 \
  --llm_task_level token_level,span_level,seq_level,structure_level \
  --llm_time_str 20231125113045 \
  --llm_step 17600000 \
  --truncation_seq_length 10240 \
  --trunc_type right \
  --seq_type prot \
  --input_file ../../../dataset/PE09_immune_escape_per_serum/protein/regression/all_seqs.fasta \
  --save_path ../../../../matrices/PE09_immune_escape_per_serum/lucaone/v2.0/17600000 \
  --embedding_type matrix \
  --matrix_add_special_token \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --embedding_fixed_len_a_time 10242 \
  --gpu_id 1

# Phage_PVP
cd ./src/llm/lucaone
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python get_embedding.py \
  --llm_dir ../../..  \
  --llm_type lucaone \
  --llm_version v2.0 \
  --llm_task_level token_level,span_level,seq_level,structure_level \
  --llm_time_str 20231125113045 \
  --llm_step 17600000 \
  --truncation_seq_length 10240 \
  --trunc_type right \
  --seq_type prot \
  --input_file ../../../dataset/Phage_PVP/protein/multi_class/all_seqs.fasta \
  --save_path ../../../../matrices/Phage_PVP/lucaone/v2.0/17600000 \
  --embedding_type matrix \
  --matrix_add_special_token \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --embedding_fixed_len_a_time 4096 \
  --gpu_id 1