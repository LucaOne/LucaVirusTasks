# PPC_uniprot
cd LucaVirusTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id, seq_type, seq
# seq_type must be prot
# input file format also can be fasta
export CUDA_VISIBLE_DEVICES="0,1"
python predict_v2.py \
    --seq_type prot \
    --input_file ../data/PPC_uniprot_independent_test/flaviviridae_pp_refseq.faa \
    --llm_truncation_seq_length 4096 \
    --emb_dir ../predicted/LucaVirus/PPC_uniprot_independent_test/embedding/flaviviridae_pp_refseq \
    --model_path .. \
    --save_path ../predicted/LucaVirus/PPC_uniprot_independent_test/PPC_uniprot_independent_test_flaviviridae_pp_refseq_predicted_results.csv \
    --dataset_name PPC_uniprot \
    --dataset_type protein \
    --task_type binary_class \
    --task_level_type token_level \
    --model_type lucabase \
    --input_type matrix \
    --input_mode single \
    --time_str 20241203132555 \
    --step 2474580 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --gpu_id 0

export CUDA_VISIBLE_DEVICES="0,1"
python predict_v2.py \
    --seq_type prot \
    --input_file ../data/PPC_uniprot_independent_test/picorna_pp_refseq.faa \
    --llm_truncation_seq_length 4096 \
    --emb_dir ../predicted/LucaVirus/PPC_uniprot_independent_test/embedding/picorna_pp_refseq \
    --model_path .. \
    --save_path ../predicted/LucaVirus/PPC_uniprot_independent_test/PPC_uniprot_independent_test_picorna_pp_refseq_predicted_results.csv \
    --dataset_name PPC_uniprot \
    --dataset_type protein \
    --task_type binary_class \
    --task_level_type token_level \
    --model_type lucabase \
    --input_type matrix \
    --input_mode single \
    --time_str 20241203132555 \
    --step 2474580 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --gpu_id 1


