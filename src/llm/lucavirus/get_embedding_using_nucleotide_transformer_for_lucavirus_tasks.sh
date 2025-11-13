# 255 机器
# DMS_Bind_Reps_Strain_Nucl
cd ./src/llm/nucleotide_transformer
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_version 2.5b-multi-species \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DMS_Bind_Reps_Strain_Nucl/gene/regression/all_genes.fasta \
    --save_path ../../../../matrices/DMS_Bind_Reps_Strain_Nucl/nucleotide_transformer/multi-species/2.5B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 3


# 255 机器
# DeepAbBindv2_genome
cd ./src/llm/nucleotide_transformer
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_version 2.5b-multi-species \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DeepAbBindv2_genome/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/nucleotide_transformer/multi-species/2.5B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 3


# 255 机器
# DeepAbBindv2_nucl
cd ./src/llm/nucleotide_transformer
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_version 2.5b-multi-species \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DeepAbBindv2_nucl/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/nucleotide_transformer/multi-species/2.5B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 1
