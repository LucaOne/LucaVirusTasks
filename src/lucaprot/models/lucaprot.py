#!/usr/bin/env python
# encoding: utf-8
'''
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/12/19 14:23
@project: LucaVirusTasks
@file: model
@desc: our model(LucaProt)
'''
import sys, math
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from common.pooling import *
    from common.modeling_bert import BertModel, BertPreTrainedModel
    from common.classification_loss import AsymmetricLossOptimized, FocalLossBinaryClass, FocalLossMultiClass, MultiLabelCCE
    from gcn import *
except ImportError:
    from src.common.pooling import *
    from src.common.modeling_bert import BertModel, BertPreTrainedModel
    from src.common.classification_loss import AsymmetricLossOptimized, FocalLossBinaryClass, FocalLossMultiClass, MultiLabelCCE
    from src.lucaprot.models.gcn import *
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss


def create_pooler(pooler_type, config, args):
    '''
    pooler building
    :param pooler_type:
    :param config:
    :param args:
    :return:
    '''
    if pooler_type == "seq":
        pooling_type = args.seq_pooling_type
        hidden_size = config.hidden_size
    elif pooler_type == "struct":
        pooling_type = args.struct_pooling_type
        hidden_size = sum(config.struct_output_size)
        if pooling_type is None:
            pooling_type = "sum"
    elif pooler_type == "embedding":
        pooling_type = args.matrix_pooling_type
        hidden_size = config.embedding_input_size
    else:
        raise Exception("Not support pooling_type=%s" % pooler_type)

    if pooling_type == "max":
        return GlobalMaskMaxPooling1D()
    elif pooling_type == "sum":
        return GlobalMaskSumPooling1D(axis=1)
    elif pooling_type == "avg":
        return GlobalMaskAvgPooling1D()
    elif pooling_type == "attention":
        return GlobalMaskContextAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "context_attention":
        return GlobalMaskContextAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "weighted_attention":
        return GlobalMaskWeightedAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "value_attention":
        return GlobalMaskValueAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "transformer":
        copy_config = copy.deepcopy(config)
        copy_config.hidden_size = hidden_size
        return GlobalMaskTransformerPooling1D(copy_config)
    else:
        return None


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


def create_activate(activate_func):
    if activate_func:
        activate_func = activate_func.lower()
    if activate_func == "tanh":
        return nn.Tanh()
    elif activate_func == "relu":
        return nn.ReLU()
    elif activate_func == "leakyrelu":
        return nn.LeakyReLU()
    elif activate_func == "gelu":
        return nn.GELU()
    elif activate_func == "gelu_new":
        return NewGELUActivation()
    else:
        return nn.Tanh()


class LucaProt(BertPreTrainedModel):
    '''
    LucaProt
    '''

    def __init__(self, config, args=None):
        super(LucaProt, self).__init__(config)
        self.num_labels = config.num_labels
        # sequence encoder, structure encoder, structural embedding encoder
        self.input_type = args.input_type
        self.has_seq_encoder = False
        if "seq" in self.input_type:
            self.has_seq_encoder = True
            self.seq_pooling_type = args.seq_pooling_type
        self.has_struct_encoder = False
        if "struct" in self.input_type:
            self.has_struct_encoder = True
            self.struct_pooling_type = args.struct_pooling_type
        self.has_embedding_encoder = False
        if "matrix" in self.input_type or "vector" in self.input_type:
            self.has_embedding_encoder = True
            self.matrix_pooling_type = args.matrix_pooling_type
        assert self.has_seq_encoder or self.has_struct_encoder or self.has_embedding_encoder
        # includes sequence encoder
        if self.has_seq_encoder:
            # sequence -> transformer(k11 layers) + pooling + dense(k12 layers)
            self.seq_encoder = BertModel(config,
                                         use_pretrained_embedding=False,
                                         add_pooling_layer=args.seq_pooling_type in ["first", "last"])
            self.seq_pooler = create_pooler(pooler_type="seq", config=config, args=args)
            assert isinstance(config.seq_fc_size, list)
            self.seq_linear = []
            input_size = config.hidden_size
            for idx in range(len(config.seq_fc_size)):
                linear = nn.Linear(input_size, config.seq_fc_size[idx])
                self.seq_linear.append(linear)
                self.seq_linear.append(create_activate(config.activate_func))
                input_size = config.seq_fc_size[idx]
            self.seq_linear = nn.ModuleList(self.seq_linear)
        # includes structure encoder
        if self.has_struct_encoder:
            # structure-> embedding + gcn(k21 layers) + pooling + dense(k22 layers)
            # k layers
            # output：[batch_size, seq_len, output_dim]
            self.struct_embedder = nn.Embedding(config.struct_vocab_size, config.struct_embed_size,
                                                padding_idx=config.pad_token_id)
            self.struct_encoder = []
            assert isinstance(config.struct_hidden_size, list) and isinstance(config.struct_output_size, list)
            input_size = config.struct_embed_size
            output_size = None
            assert len(config.struct_hidden_size) == len(config.struct_output_size)
            for idx in range(len(config.struct_output_size)):
                layer = GAT(feature_size=input_size,
                            hidden_size=config.struct_hidden_size[idx],
                            output_size=config.struct_output_size[idx],
                            dropout=config.hidden_dropout_prob,
                            nheads=config.struct_nb_heads,
                            alpha=config.struct_alpha)
                self.struct_encoder.append(layer)
                input_size = config.struct_output_size[idx]
                output_size = config.struct_output_size[idx]
            self.struct_encoder = nn.ModuleList(self.struct_encoder)
            self.struct_pooler = create_pooler(pooler_type="struct", config=config, args=args)
            assert isinstance(config.struct_fc_size, list)
            self.struct_linear = []
            input_size = output_size * len(config.struct_output_size)
            for idx in range(len(config.struct_fc_size)):
                linear = nn.Linear(input_size, config.struct_fc_size[idx])
                self.struct_linear.append(linear)
                self.struct_linear.append(create_activate(config.activate_func))
                input_size = config.struct_fc_size[idx]
            self.struct_linear = nn.ModuleList(self.struct_linear)
        # includes embedding encoder
        if self.has_embedding_encoder:
            self.embedding_pooler = create_pooler(pooler_type="embedding", config=config, args=args)
            assert isinstance(config.embedding_fc_size, list)
            self.embedding_linear = []
            input_size = config.embedding_input_size
            for idx in range(len(config.embedding_fc_size)):
                linear = nn.Linear(input_size, config.embedding_fc_size[idx])
                self.embedding_linear.append(linear)
                self.embedding_linear.append(create_activate(config.activate_func))
                input_size = config.embedding_fc_size[idx]
            self.embedding_linear = nn.ModuleList(self.embedding_linear)

        # weight assignment for addition of sequence, structure, structural embedding representation vector,
        # if none, concatenation, otherwise weighted sequence
        if self.has_seq_encoder and self.has_struct_encoder and self.has_embedding_encoder:
            if hasattr(config, "seq_weight") \
                    and hasattr(config, "struct_weight") \
                    and hasattr(config, "embedding_weight"):
                self.seq_weight = config.seq_weight
                self.struct_weight = config.struct_weight
                self.embedding_weight = config.embedding_weight
            else:
                self.seq_weight = None
                self.struct_weight = None
                self.embedding_weight = None
            assert self.seq_weight is None or self.seq_weight + self.struct_weight + self.embedding_weight == 1.0
            if self.seq_weight is None:
                # concat
                output_size = config.seq_fc_size[-1] + config.struct_fc_size[-1] + config.embedding_fc_size[-1]
            else:
                # add
                assert config.seq_fc_size[-1] == config.struct_fc_size[-1] == config.embedding_fc_size[-1]
                output_size = config.seq_fc_size[-1]
        elif self.has_seq_encoder and self.has_struct_encoder:
            if hasattr(config, "seq_weight") and hasattr(config, "struct_weight"):
                self.seq_weight = config.seq_weight
                self.struct_weight = config.struct_weight
            else:
                self.seq_weight = None
                self.struct_weight = None
            self.embedding_weight = None
            assert self.seq_weight is None or self.seq_weight + self.struct_weight == 1.0
            if self.seq_weight is None:
                # concat
                output_size = config.seq_fc_size[-1] + config.struct_fc_size[-1]
            else:
                # add
                assert config.seq_fc_size[-1] == config.struct_fc_size[-1]
                output_size = config.seq_fc_size[-1]
        elif self.has_seq_encoder and self.has_embedding_encoder:
            if hasattr(config, "seq_weight") and hasattr(config, "embedding_weight"):
                self.seq_weight = config.seq_weight
                self.embedding_weight = config.embedding_weight
            else:
                self.seq_weight = None
                self.embedding_weight = None
            self.struct_weight = None
            assert self.seq_weight is None or self.seq_weight + self.embedding_weight == 1.0
            if self.seq_weight is None:
                # concat
                output_size = config.seq_fc_size[-1] + config.embedding_fc_size[-1]
            else:
                # add
                assert config.seq_fc_size[-1] == config.embedding_fc_size[-1]
                output_size = config.seq_fc_size[-1]
        elif self.has_struct_encoder and self.has_embedding_encoder:
            if hasattr(config, "struct_weight") and hasattr(config, "embedding_weight"):
                self.struct_weight = config.struct_weight
                self.embedding_weight = config.embedding_weight
            else:
                self.struct_weight = None
                self.embedding_weight = None
            self.seq_weight = None
            assert self.struct_weight is None or self.struct_weight + self.embedding_weight == 1.0
            if self.struct_weight is None:
                # concat
                output_size = config.struct_fc_size[-1] + config.embedding_fc_size[-1]
            else:
                # add
                assert config.struct_fc_size[-1] == config.embedding_fc_size[-1]
                output_size = config.struct_fc_size[-1]
        else:  # only one encoder
            self.seq_weight = None
            self.struct_weight = None
            self.embedding_weight = None
            output_size = config.seq_fc_size[-1] if self.has_seq_encoder else (
                config.struct_fc_size[-1] if self.has_struct_encoder else config.embedding_fc_size[-1])

        # dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # output layer
        self.output_mode = args.output_mode
        if args and args.sigmoid:
            if args.output_mode in ["binary_class", "binary-class"]:
                self.classifier = nn.Linear(output_size, 1)
            else:
                self.classifier = nn.Linear(output_size, config.num_labels)
            self.output = nn.Sigmoid()
        else:
            self.classifier = nn.Linear(output_size, config.num_labels)
            if self.num_labels > 1:
                self.output = nn.Softmax(dim=-1)
            else:
                self.output = None

        # loss function type
        self.loss_type = args.loss_type

        # positive weight
        if hasattr(args, "pos_weight"):
            self.pos_weight = args.pos_weight
        elif hasattr(config, "pos_weight"):
            self.pos_weight = config.pos_weight
        else:
            self.pos_weight = None

        if hasattr(args, "weight"):
            self.weight = args.weight
        elif hasattr(config, "weight"):
            self.weight = config.weight
        else:
            self.weight = None

        if self.output_mode in ["regression"]:
            self.loss_fct = MSELoss()
        elif self.output_mode in ["multi_label", "multi-label"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [1, 1, 1, ,1, 1...] length: num_labels
                    if isinstance(self.pos_weight, str):
                        self.pos_weight = [float(self.pos_weight)] * self.num_labels
                    elif isinstance(self.pos_weight, float):
                        self.pos_weight = [self.pos_weight] * self.num_labels
                    self.pos_weight = torch.tensor(self.pos_weight, dtype=torch.float32).to(args.device)
                    print("multi_label positive weight:")
                    print(self.pos_weight)
                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == self.num_labels
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss(
                        reduction=config.loss_reduction if hasattr(config, "loss_reduction") else "sum")
            elif self.loss_type == "asl":
                self.loss_fct = AsymmetricLossOptimized(
                    gamma_neg=args.asl_gamma_neg if hasattr(args, "asl_gamma_neg") else 4,
                    gamma_pos=args.asl_gamma_pos if hasattr(args, "asl_gamma_pos") else 1,
                    clip=args.clip if hasattr(args, "clip") else 0.05,
                    eps=args.eps if hasattr(args, "eps") else 1e-8,
                    disable_torch_grad_focal_loss=args.disable_torch_grad_focal_loss if hasattr(args, "disable_torch_grad_focal_loss") else False)
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLossBinaryClass(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 0.7,
                                                     gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 2.0,
                                                     normalization=False,
                                                     reduction=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
            elif self.loss_type == "multilabel_cce":
                self.loss_fct = MultiLabelCCE(normalization=False)
        elif self.output_mode in ["binary_class", "binary-class"]:
            if self.loss_type == "bce":
                if self.pos_weight:
                    # [0.9]
                    if isinstance(self.pos_weight, str) or isinstance(self.pos_weight, int):
                        self.pos_weight = torch.tensor([float(self.pos_weight)], dtype=torch.float32).to(args.device)
                    elif isinstance(self.pos_weight, float):
                        self.pos_weight = torch.tensor([self.pos_weight], dtype=torch.float32).to(args.device)
                    print("binary_class positive weight:")
                    print(self.pos_weight)
                    assert self.pos_weight.ndim == 1 and self.pos_weight.shape[0] == 1
                    self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                else:
                    self.loss_fct = BCEWithLogitsLoss()
            elif self.loss_type == "focal_loss":
                self.loss_fct = FocalLossBinaryClass(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 0.7,
                                                     gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 2.0,
                                                     normalization=False,
                                                     reduction=args.focal_loss_reduce if hasattr(args, "focal_loss_reduce") else False)
        elif self.output_mode in ["multi_class", "multi-class"]:
            if self.weight:
                # [1, 1, 1, ,1, 1...] length: self.num_labels
                weight = torch.tensor(self.weight, dtype=torch.float32).to(args.device)
                print("multi_class loss weight:")
                print(weight)
                assert weight.ndim == 1 and weight.shape[0] == self.num_labels
                self.loss_fct = CrossEntropyLoss(weight=weight)
            else:
                self.loss_fct = CrossEntropyLoss()
        else:
            raise Exception("Not support output mode: %s." % self.output_mode)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            seq_attention_masks=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            struct_input_ids=None,
            struct_contact_map=None,
            struct_attention_masks=None,
            vectors=None,
            matrices=None,
            matrix_attention_masks=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        seq_pooled_output = None
        if self.has_seq_encoder:
            # calc for sequence
            if input_ids is not None:
                seq_outputs = self.seq_encoder(
                    input_ids,
                    attention_mask=seq_attention_masks,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                if self.seq_pooler is None:
                    # [CLS] vector
                    seq_pooled_output = seq_outputs[1]
                else:
                    # matrix to vector
                    # seq_pooled_output = self.seq_pooler(seq_outputs[0])
                    if isinstance(seq_outputs, list):
                        seq_pooled_output = self.seq_pooler(seq_outputs[0], mask=seq_attention_masks)
                    else:
                        seq_pooled_output = self.seq_pooler(seq_outputs.last_hidden_state, mask=seq_attention_masks)
                    '''
                    if isinstance(seq_outputs, list):
                        seq_pooled_output = self.seq_pooler(seq_outputs[0])
                    else:
                        seq_pooled_output = self.seq_pooler(seq_outputs.last_hidden_state)
                    '''
                seq_pooled_output = self.dropout(seq_pooled_output)
                for i, layer_module in enumerate(self.seq_linear):
                    seq_pooled_output = layer_module(seq_pooled_output)
        struct_pooled_output = None
        if self.has_struct_encoder:
            if struct_input_ids is not None:
                assert struct_contact_map is not None
                struct_encoding_list = []
                hidden_states = self.struct_embedder(struct_input_ids)
                for i, layer_module in enumerate(self.struct_encoder):
                    hidden_states = layer_module(hidden_states, struct_contact_map)
                    struct_encoding_list.append(hidden_states)
                if len(struct_encoding_list) > 1:
                    hidden_states = torch.cat(struct_encoding_list, dim=-1)
                else:
                    hidden_states = struct_encoding_list[-1]
                struct_pooled_output = self.struct_pooler(hidden_states, mask=struct_attention_masks)
                struct_pooled_output = self.dropout(struct_pooled_output)
                for i, layer_module in enumerate(self.struct_linear):
                    struct_pooled_output = layer_module(struct_pooled_output)
        embedding_pooled_output = None
        if self.has_embedding_encoder:
            if self.embedding_pooler is not None:
                embedding_pooled_output = self.embedding_pooler(matrices, mask=matrix_attention_masks)
            elif vectors is not None:
                embedding_pooled_output = vectors
            elif self.matrix_pooling_type == "last":
                embedding_pooled_output = matrices[:, -1, :]
            else:
                embedding_pooled_output = matrices[:, 0, :]

            embedding_pooled_output = self.dropout(embedding_pooled_output)
            for i, layer_module in enumerate(self.embedding_linear):
                embedding_pooled_output = layer_module(embedding_pooled_output)

        assert seq_pooled_output is not None or struct_pooled_output is not None or embedding_pooled_output is not None

        if seq_pooled_output is None and embedding_pooled_output is None:
            pooled_output = struct_pooled_output
        elif struct_pooled_output is None and embedding_pooled_output is None:
            pooled_output = seq_pooled_output
        elif seq_pooled_output is None and struct_pooled_output is None:
            pooled_output = embedding_pooled_output
        else:
            if self.seq_weight is not None and self.struct_weight is not None and self.embedding_weight is not None:
                pooled_output = torch.add(self.seq_weight * seq_pooled_output,
                                          self.struct_weight * struct_pooled_output,
                                          self.embedding_weight * embedding_pooled_output)
            elif self.seq_weight is not None and self.struct_weight is not None:
                pooled_output = torch.add(self.seq_weight * seq_pooled_output,
                                          self.struct_weight * struct_pooled_output)
            elif self.seq_weight is not None and self.embedding_weight is not None:
                pooled_output = torch.add(self.seq_weight * seq_pooled_output,
                                          self.embedding_weight * embedding_pooled_output)
            elif self.struct_weight is not None and self.embedding_weight is not None:
                pooled_output = torch.add(self.struct_weight * struct_pooled_output,
                                          self.embedding_weight * embedding_pooled_output)
            elif seq_pooled_output is not None and struct_pooled_output is not None and embedding_pooled_output is not None:
                pooled_output = torch.cat([seq_pooled_output, struct_pooled_output, embedding_pooled_output], dim=-1)
            elif seq_pooled_output is not None and struct_pooled_output is not None:
                pooled_output = torch.cat([seq_pooled_output, struct_pooled_output], dim=-1)
            elif seq_pooled_output is not None and embedding_pooled_output is not None:
                pooled_output = torch.cat([seq_pooled_output, embedding_pooled_output], dim=-1)
            elif struct_pooled_output is not None and embedding_pooled_output is not None:
                pooled_output = torch.cat([struct_pooled_output, embedding_pooled_output], dim=-1)
            else:
                raise Exception("Not support this type.")

        logits = self.classifier(pooled_output)
        if self.output:
            output = self.output(logits)
        else:
            output = logits
        outputs = [logits, output]

        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.output_mode in ["multi_label", "multi-label"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception("Not support this output_mode=%s" % self.output_mode)
            outputs = [loss, *outputs]

        return outputs
