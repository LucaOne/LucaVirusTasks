# DeepAbBindv2_genome
cd ../llm/lucavirus/
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
    --input_file ../../../data/DeepAbBindv2/genome/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

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
    --input_file ../../../data/DeepAbBindv2/genome/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# DeepAbBindv2_nucl
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
    --input_file ../../../data/DeepAbBindv2/nucl/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3


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
    --input_file ../../../data/DeepAbBindv2/nucl/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# DeepAbBindv2_original
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
    --input_file ../../../data/DeepAbBindv2/original/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_original/lucavirus/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3


# lucavirus-mask
# DeepAbBindv2_genome
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/genome/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/genome/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# DeepAbBindv2_nucl
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/nucl/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/nucl/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# DeepAbBindv2_original
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/original/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_original/lucavirus-mask/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# lucavirus-gene/prot
# DeepAbBindv2_genome
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
    --input_file ../../../data/DeepAbBindv2/genome/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus-prot/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

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
    --input_file ../../../data/DeepAbBindv2/genome/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucavirus-gene/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# DeepAbBindv2_nucl
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
    --input_file ../../../data/DeepAbBindv2/nucl/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus-prot/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3


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
    --input_file ../../../data/DeepAbBindv2/nucl/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucavirus-gene/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

# DeepAbBindv2_original
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
    --input_file ../../../data/DeepAbBindv2/original/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_original/lucavirus-prot/v1.0/3800000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3