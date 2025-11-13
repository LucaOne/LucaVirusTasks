#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/21 17:32
@project: LucaVirusTasks
@file: LucaTripleIntraInter
@desc: xxxx
"""

import sys
import logging
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../src")
try:
    from common.pooling import *
    from common.loss import *
    from utils import *
    from common.multi_label_metrics import *
    from common.metrics import *
    from common.luca_triple import LucaTriple
    from common.modeling_bert import BertModel, BertPreTrainedModel
except ImportError:
    from src.common.pooling import *
    from src.common.loss import *
    from src.utils import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *
    from src.common.luca_triple import LucaTriple
    from src.common.modeling_bert import BertModel, BertPreTrainedModel
logger = logging.getLogger(__name__)


class LucaTripleIntraInter(BertPreTrainedModel):
    def __init__(self, config, args):
        super(LucaTripleIntraInter, self).__init__(config)
        self.input_type = args.input_type
        self.num_labels = config.num_labels
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        if config.hidden_size != config.embedding_input_size_a:
            self.linear_a = nn.Linear(config.embedding_input_size_a, config.hidden_size, bias=True)
        else:
            self.linear_a = None
        if config.hidden_size != config.embedding_input_size_b:
            self.linear_b = nn.Linear(config.embedding_input_size_b, config.hidden_size, bias=True)
        else:
            self.linear_b = None
        if config.hidden_size != config.embedding_input_size_c:
            self.linear_c = nn.Linear(config.embedding_input_size_c, config.hidden_size, bias=True)
        else:
            self.linear_c = None
        self.encoder = LucaTriple(config)
        config.embedding_input_size = config.hidden_size
        self.pooler = nn.ModuleList([create_pooler(pooler_type="matrix", config=config, args=args) for _ in range(9)])
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=9 * config.hidden_size if self.fusion_type == "concat" else config.hidden_size,
                classifier_size=args.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )
        self.post_init()

    def forward(
            self,
            input_ids_a=None,
            input_ids_b=None,
            input_ids_c=None,
            position_ids_a=None,
            position_ids_b=None,
            position_ids_c=None,
            token_type_ids_a=None,
            token_type_ids_b=None,
            token_type_ids_c=None,
            seq_attention_masks_a=None,
            seq_attention_masks_b=None,
            seq_attention_masks_c=None,
            vectors_a=None,
            vectors_b=None,
            vectors_c=None,
            matrices_a=None,
            matrices_b=None,
            matrices_c=None,
            matrix_attention_masks_a=None,
            matrix_attention_masks_b=None,
            matrix_attention_masks_c=None,
            output_attentions=False,
            labels=None,
            **kwargs
    ):
        # 对称结构： intra-attention + inter-attention
        # matrices_a: seq_a_embedding, [B, seq_len_a, dim]
        # matrix_attention_masks_a: seq_a_mask, [B, seq_len_a]
        # matrices_b: seq_b_embedding, [B, seq_len_b, dim]
        # matrix_attention_masks_b: seq_b_mask, [B, seq_len_b]
        if self.linear_a is not None:
            # [B, seq_len_a, dim]->[B, seq_len_a, hidden_size]
            hidden_states_a = self.linear_a(matrices_a)
        else:
            hidden_states_a = matrices_a
        if self.linear_b is not None:
            # [B, seq_len_b, dim]->[B, seq_len_b, hidden_size]
            hidden_states_b = self.linear_b(matrices_b)
        else:
            hidden_states_b = matrices_b
        if self.linear_c is not None:
            # [B, seq_len_c, dim]->[B, seq_len_c, hidden_size]
            hidden_states_c = self.linear_c(matrices_c)
        else:
            hidden_states_c = matrices_c

        last_hidden_states = self.encoder(
            hidden_states_a=hidden_states_a,
            attention_mask_a=matrix_attention_masks_a,
            hidden_states_b=hidden_states_b,
            attention_mask_b=matrix_attention_masks_b,
            hidden_states_c=hidden_states_c,
            attention_mask_c=matrix_attention_masks_c,
            head_mask_a=None,
            cross_attn_head_mask_ab=None,
            cross_attn_head_mask_ac=None,
            head_mask_b=None,
            cross_attn_head_mask_ba=None,
            cross_attn_head_mask_bc=None,
            head_mask_c=None,
            cross_attn_head_mask_ca=None,
            cross_attn_head_mask_cb=None,
            past_key_values=None,
            cross_past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True
        ).last_hidden_state

        last_hidden_states = [
            # a
            self.pooler[0](last_hidden_states[0], seq_attention_masks_a),
            # b
            self.pooler[1](last_hidden_states[1], seq_attention_masks_b),
            # c
            self.pooler[2](last_hidden_states[2], seq_attention_masks_c),
            # ab
            self.pooler[3](last_hidden_states[3], seq_attention_masks_a),
            # ba
            self.pooler[4](last_hidden_states[4], seq_attention_masks_b),
            # ac
            self.pooler[5](last_hidden_states[5], seq_attention_masks_a),
            # ca
            self.pooler[6](last_hidden_states[6], seq_attention_masks_c),
            # bc
            self.pooler[7](last_hidden_states[7], seq_attention_masks_b),
            # cb
            self.pooler[8](last_hidden_states[8], seq_attention_masks_c),
        ]

        if self.dropout is not None:
            last_hidden_states = [self.dropout(hidden) for hidden in last_hidden_states]

        if self.fusion_type == "add":
            last_hidden_states = torch.add(last_hidden_states)
        else:
            last_hidden_states = torch.cat(last_hidden_states, dim=-1)

        if self.hidden_layer is not None:
            last_hidden_states = self.hidden_layer(last_hidden_states)
        if self.hidden_act is not None:
            last_hidden_states = self.hidden_act(last_hidden_states)

        logits = self.classifier(last_hidden_states)
        if self.output:
            output = self.output(logits)
        else:
            output = logits
        outputs = [logits, output]

        if labels is not None:
            if self.output_mode in ["regression"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N, seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.output_mode in ["multi_label", "multi-label"]:
                if self.loss_reduction == "meanmean":
                    # logits: N , label_size
                    # labels: N , label_size
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N , label_size
                    # labels: N , label_size
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N * seq_len * 1
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, label_size
                    # labels: N , seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len, label_size
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception("not support the output_mode=%s" % self.output_mode)
            outputs = [loss, *outputs]
        return outputs
