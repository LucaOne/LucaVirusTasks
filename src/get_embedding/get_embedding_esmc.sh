
# DeepAbBindv2_original
cd ../llm/esmc/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esmc \
   --llm_version 600M \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../data/DeepAbBindv2/original/prot.fasta \
   --save_path  ../../../../matrices/DeepAbBindv2_original/esm/esmc/600M \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 10240 \
   --gpu_id 2