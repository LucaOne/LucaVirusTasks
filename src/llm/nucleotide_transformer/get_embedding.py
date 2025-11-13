#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/10/11 13:05
@project: LucaVirusTasks
@file: get_embedding
@desc: xxxx
"""
import os
import sys
import torch
import numpy as np
import argparse
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import fasta_reader, csv_reader, tsv_reader
    from utils import clean_seq_luca, calc_emb_filename_by_seq_id
except ImportError:
    from src.file_operator import fasta_reader, csv_reader, tsv_reader
    from src.utils import clean_seq_luca, calc_emb_filename_by_seq_id
from transformers import AutoTokenizer, AutoModelForMaskedLM


model_id_2_5b_multi_species = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
model_id_2_5b_1000g = "InstaDeepAI/nucleotide-transformer--2.5b-1000g"
model_id_500m_1000g =  "nucleotide-transformer-500m-1000g"
nucleotide_transformer_global_model, nucleotide_transformer_global_alphabet, nucleotide_transformer_global_version = None, None, None


def predict_embedding(
        sample,
        trunc_type,
        embedding_type,
        repr_layers=[-1],
        truncation_seq_length=4094,
        device=None,
        version="2.5b-multi-species",
        matrix_add_special_token=False,
        embedding_complete=True,
        fp16=False
):
    '''
    use sequence to predict seq embedding matrix or vector(bos)
    :param sample: [seq_id, seq]
    :param trunc_type:
    :param embedding_type: bos or representations
    :param repr_layers: [-1]
    :param truncation_seq_length: [4094,2046,1982,1790,1534,1278,1150,1022]
    :param device:
    :param version:
    :param matrix_add_special_token:
    :param embedding_complete:
    :param fp16:
    :return: embedding, processed_seq_len
    '''
    global nucleotide_transformer_global_model, nucleotide_transformer_global_alphabet, nucleotide_transformer_global_version
    assert "bos" in embedding_type or "representations" in embedding_type or "matrix" in embedding_type or "vector" in embedding_type
    if len(sample) > 2:
        seq_id, seq = sample[0], sample[2]
    else:
        seq_id, seq = sample[0], sample[1]
    processed_seq = clean_seq_luca(seq_id, seq)
    if len(processed_seq) > truncation_seq_length:
        if trunc_type == "left":
            processed_seq = processed_seq[-truncation_seq_length:]
        else:
            processed_seq = processed_seq[:truncation_seq_length]
    if "2.5b" in version.lower() and "multi-species" in version:
        version = "2.5b-multi-species"
    elif "2.5b" in version.lower() and "1000g" in version:
        version = "2.5b-1000g"
    elif "500m" in version.lower() and "1000g" in version:
        version = "500m-1000g"
    else:
        version = "2.5b-multi-species"
        print("Using %s" % version)
    if nucleotide_transformer_global_model is None or nucleotide_transformer_global_alphabet is None \
            or nucleotide_transformer_global_version is None or nucleotide_transformer_global_version != version:
        if version == "2.5b-multi-species":
            nucleotide_transformer_global_alphabet = AutoTokenizer.from_pretrained(model_id_2_5b_multi_species)
            nucleotide_transformer_global_model = AutoModelForMaskedLM.from_pretrained(model_id_2_5b_multi_species)
        elif version == "2.5b-1000g":
            nucleotide_transformer_global_alphabet = AutoTokenizer.from_pretrained(model_id_2_5b_1000g)
            nucleotide_transformer_global_model = AutoModelForMaskedLM.from_pretrained(model_id_2_5b_1000g)
        elif version == "500m-1000g":
            nucleotide_transformer_global_alphabet = AutoTokenizer.from_pretrained(model_id_500m_1000g)
            nucleotide_transformer_global_model = AutoModelForMaskedLM.from_pretrained(model_id_500m_1000g)
        else:
            raise Exception("not support this version=%s" % version)
        nucleotide_transformer_global_version = version
    if device is None:
        device = next(nucleotide_transformer_global_model.parameters()).device
    else:
        model_device = next(nucleotide_transformer_global_model.parameters()).device
        if device != model_device:
            nucleotide_transformer_global_model = nucleotide_transformer_global_model.to(device)
    nucleotide_transformer_global_model.eval()

    # nucleotide_transformer 应该是使用了绝对位置，最长1000
    max_length = nucleotide_transformer_global_alphabet.model_max_length
    tokens_ids = nucleotide_transformer_global_alphabet.batch_encode_plus(
        [processed_seq],
        return_tensors="pt",
        padding="max_length",
        max_length=max_length
    )["input_ids"]
    embeddings = {}
    with torch.no_grad():
        matrices = []
        vectors = []
        try:
            seg_input_ids_list = []
            if tokens_ids.shape[1] > max_length:
                if embedding_complete:
                    for idx in range(1, tokens_ids.shape[1], max_length - 1):
                        start = idx
                        end = min(idx + max_length - 1, tokens_ids.shape[1])
                        cur_tokens_ids = torch.zeros(tokens_ids.shape[0], end - start + 1, dtype=torch.long)
                        cur_tokens_ids[:, 0] = tokens_ids[:, 0]
                        cur_tokens_ids[:, 1:] = tokens_ids[:, start: end]
                        seg_input_ids_list.append(cur_tokens_ids)
                else:
                    # 截断
                    seg_input_ids_list.append(tokens_ids[:, 0:max_length])
            else:
                seg_input_ids_list.append(tokens_ids)
            for seg_tokens_ids in seg_input_ids_list:
                seg_tokens_ids = seg_tokens_ids.to(device=device, non_blocking=True)
                # Compute the embeddings
                seg_attention_mask = seg_tokens_ids != nucleotide_transformer_global_alphabet.pad_token_id
                torch_outs = nucleotide_transformer_global_model(
                    seg_tokens_ids,
                    attention_mask=seg_attention_mask,
                    encoder_attention_mask=seg_attention_mask,
                    output_hidden_states=True
                )
                # Compute sequences embeddings
                out = torch_outs['hidden_states'][-1]
                if "representations" in embedding_type or "matrix" in embedding_type:
                    if matrix_add_special_token and len(matrices) == 0:
                        embedding = out[0, :, :].to(device="cpu").clone().numpy()
                    else:
                        # only <cls>
                        embedding = out[0, 1:, :].to(device="cpu").clone().numpy()
                    matrices.append(embedding)
                if "bos" in embedding_type or "vector" in embedding_type:
                    # attention_mask = torch.unsqueeze(attention_mask, dim=-1)
                    # embedding = torch.sum(attention_mask * out, axis=-2)/torch.sum(attention_mask, axis=1)[0].to(device="cpu").clone().numpy()
                    vectors.append(out[0, 0, :].to(device="cpu").clone().numpy())
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {seq_id} of length {len(seq)}.")
                print("Please reduce the 'truncation_seq_length'")
            raise Exception(e)
        if matrices:
            if len(matrices) == 1:
                embeddings["representations"] = matrices[0]
            else:
                embeddings["representations"] = np.concatenate(matrices, axis=0)
        if vectors:
            if len(vectors) == 1:
                embeddings["bos_representations"] = vectors[0]
            else:
                embeddings["bos_representations"] = np.mean(vectors, axis=0)
        if len(embeddings) > 1:
            return embeddings, out.shape[1]
        elif len(embeddings) == 1:
            return list(embeddings.items())[0][1], out.shape[1]
        else:
            return None, None


def get_args():
    parser = argparse.ArgumentParser(description='nucleotide_transformer Embedding')
    # for one seq
    parser.add_argument(
        "--seq_id",
        type=str,
        default=None,
        help="the seq id"
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=None,
        help="when to input a seq"
    )
    parser.add_argument(
        "--seq_type",
        type=str,
        default="gene",
        choices=["gene", "multi_gene", "dna", "rna"],
        help="the input seq type"
    )

    # for many
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="the input filepath(.fasta or .csv or .tsv)"
    )

    # for input csv
    parser.add_argument(
        "--id_idx",
        type=int,
        default=None,
        help="id col idx(0 start)"
    )
    parser.add_argument(
        "--seq_idx",
        type=int,
        default=None,
        help="seq col idx(0 start)"
    )

    # for saved path
    parser.add_argument(
        "--save_path",
        type=str, default=None,
        help="embedding file save dir path"
    )
    parser.add_argument(
        "--llm_version", type=str, default="2.5b-multi-species",
        choices=["2.5b-multi-species", "2.5b-1000g"],
        help="the llm version."
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="matrix",
        choices=["matrix", "vector"],
        help="the llm embedding type."
    )
    parser.add_argument(
        "--trunc_type", type=str, default="right",
        choices=["left", "right"],
        help="llm trunc type."
    )
    parser.add_argument(
        "--truncation_seq_length", type=int, default=4094,
        help="the llm truncation seq length(not contain [CLS] and [SEP]."
    )
    parser.add_argument(
        "--matrix_add_special_token", action="store_true",
        help="whether to add special token embedding vector in seq representation matrix"
    )
    parser.add_argument(
        "--embedding_complete",
        action="store_true",
        help="when the seq len > inference_max_len, then the embedding matrix is completed by segment"
    )

    parser.add_argument(
        '--gpu_id', type=int, default=-1,
        help="the gpu id to use."
    )
    input_args = parser.parse_args()
    return input_args


def main(model_args):
    print(model_args)
    if model_args.gpu_id >= 0:
        gpu_id = model_args.gpu_id
    else:
        # gpu_id = available_gpu_id()
        gpu_id = -1
        print("gpu_id: ", gpu_id)
    model_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    assert (model_args.input_file is not None and os.path.exists(model_args.input_file)) or model_args.seq is not None
    print("input seq type: %s" % model_args.seq_type)
    print("args device: %s" % model_args.device)
    embedding_type = model_args.embedding_type
    seq_type = model_args.seq_type
    emb_save_path = model_args.save_path
    print("emb save dir: %s" % os.path.abspath(emb_save_path))
    if seq_type not in ["gene", "multi_gene", "dna", "rna"]:
        print("Error! arg: --seq_type=%s is not 'gene(dna or rna)' or 'multi_gene'" % seq_type)
        sys.exit(-1)
    input_seq_type = seq_type
    if seq_type == "multi_gene":
        seq_type = "gene"
    elif seq_type in ["dna", "rna"]:
        seq_type = "gene"
        input_seq_type = "gene"
    if not os.path.exists(emb_save_path):
        os.makedirs(emb_save_path)
    if model_args.input_file:
        done = 0
        file_reader = fasta_reader
        if model_args.input_file.endswith(".csv"):
            file_reader = csv_reader
        elif model_args.input_file.endswith(".tsv"):
            file_reader = tsv_reader

        for row in file_reader(model_args.input_file):
            if model_args.id_idx is None or model_args.seq_idx is None:
                if len(row) > 2:
                    seq_id, seq = row[0].strip(), row[2].upper()
                else:
                    seq_id, seq = row[0].strip(), row[1].upper()
            else:
                seq_id, seq = row[model_args.id_idx].strip(), row[model_args.seq_idx].upper()
            emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
            embedding_filepath = os.path.join(emb_save_path, emb_filename)
            if not os.path.exists(embedding_filepath):
                emb_list = []
                # multi seqs
                seq = seq.replace("U", "T")
                if "," in seq:
                    multi_seqs = [v.strip().upper() for v in seq.split(",")]
                else:
                    multi_seqs = [seq.strip().upper()]
                assert len(multi_seqs) == 1 or input_seq_type in ["multi_gene"]
                for seq in multi_seqs:
                    ori_seq_len = len(seq)
                    truncation_seq_length = model_args.truncation_seq_length
                    if model_args.embedding_complete:
                        truncation_seq_length = ori_seq_len
                    emb, processed_seq_len = predict_embedding(
                        [seq_id, seq_type, seq],
                        model_args.trunc_type,
                        embedding_type,
                        repr_layers=[-1],
                        truncation_seq_length=truncation_seq_length,
                        device=model_args.device,
                        version=model_args.llm_version,
                        matrix_add_special_token=model_args.matrix_add_special_token,
                        embedding_complete=model_args.embedding_complete
                    )
                    while emb is None:
                        print("%s embedding error, max_len from %d truncate to %d" % (
                            seq_id,
                            truncation_seq_length,
                            int(truncation_seq_length * 0.95)
                        ))
                        truncation_seq_length = int(truncation_seq_length * 0.95)
                        emb, processed_seq_len = predict_embedding(
                            [seq_id, seq_type, seq],
                            model_args.trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            device=model_args.device,
                            version=model_args.llm_version,
                            matrix_add_special_token=model_args.matrix_add_special_token,
                            embedding_complete=model_args.embedding_complete
                        )
                    # print("seq_len: %d" % len(seq))
                    # print("emb shape:", embedding_info.shape)
                    # torch.save(emb, embedding_filepath)
                    emb_list.append(emb)
                if input_seq_type in ["multi_gene"]:
                    torch.save(emb_list, embedding_filepath)
                else:
                    torch.save(emb_list[0], embedding_filepath)
                torch.cuda.empty_cache()
            else:
                print("%s exists." % embedding_filepath)
            done += 1
            if done % 1000 == 0:
                print("embedding done: %d" % done)
        print("embedding over, done: %d" % done)
    elif model_args.seq:
        print("input seq length: %d" % len(model_args.seq))
        emb, processed_seq_len = predict_embedding(
            ["input", model_args.seq],
            model_args.trunc_type,
            model_args.embedding_type,
            repr_layers=[-1],
            truncation_seq_length=model_args.truncation_seq_length,
            device=model_args.device,
            version=model_args.llm_version,
            matrix_add_special_token=model_args.matrix_add_special_token,
            embedding_complete=model_args.embedding_complete
        )
        print("done seq length: %d" % processed_seq_len)
        print(emb)
        if emb is not None:
            print(emb.shape)


if __name__ == "__main__":
    run_args = get_args()
    main(run_args)