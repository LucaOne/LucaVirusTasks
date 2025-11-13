cd ../llm/lucavirus
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --input_file ../../../data/test_data/prot/test_prot.csv \
    --save_path ../../../data/test_data/prot/lucavirus_prot_csv_input_test \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0
