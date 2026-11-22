import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP


class CRAFTV10(torch.nn.Module):
    """
    CRAFTV10: CRAFT with magnitude-preserving encodings of its scalar evidence.

    The problem: scalar evidence collapses to presence
    --------------------------------------------------
    CRAFT encodes its two scalar features, the elapsed time since the candidate's last interaction and the
    candidate's repeat count in the source's window, as LayerNorm(Linear(x)) = LN(w x + b). LayerNorm cancels the
    scale of w x, so once x is far above |b| / |w| every value maps to the same vector: the encoding keeps whether
    the evidence exists but not how much there is. In a CRAFT trained on uci, |b| / |w| is about 1.8 seconds, and
    elapsed times from about 100 seconds to 4 months are encoded identically (cosine 1.0000).

    This barely matters for seen edges, which are predicted from the presence of evidence (the candidate is in the
    source's history). It matters for unseen edges, which have no presence signal and can only be ranked by the
    magnitude of graded evidence: how recently the candidate was active and how often it occurs. On uci, CRAFT ranks
    unseen pairs almost at random (test MRR 7.9 on unseen pairs against 86.4 on seen ones, about 5 for random
    ranking), and seen-dominant benchmarks do not reveal it, because repeats carry the overall score.

    The mechanism: multiscale scalar encoding
    -----------------------------------------
    Before the projection, each scalar is expanded so that its magnitude changes the direction of the vector, not
    only its scale, and survives the LayerNorm:
        elapsed time dt   ->  [log(1 + dt / (s_m tau)) for each scale s_m, 1[never interacted]]
        repeat count c    ->  [log(1 + c), 1[c >= 1], 1[c >= 2], 1[c >= 5]]
    where tau is a running median of the elapsed times of the history entries, which adapts the scales to the time
    unit of each dataset. Everything else is CRAFT, unchanged: the learnable node embeddings, the cross-attention of
    the candidates over the source's history, the output MLP, the loss and the per-dataset hyperparameters. Changing
    only this encoding takes CRAFT from 72.29 to 79.92 test MRR on uci (15 epochs, seed 0), and its unseen-pair MRR
    from 7.9 to 22.2.

    With scalar_encoding='projected' the model is CRAFT, which is the control of the ablation.
    """

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=True,
                 input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=True,
                 num_output_layer=1, emb_dropout_prob=0.1, skip_connection=False, scalar_encoding='multiscale',
                 scalar_scales=(0.01, 0.1, 1.0, 10.0, 100.0), time_scale_momentum=0.99, history_pooling='attention'):
        """
        the arguments up to skip_connection are CRAFT's, with the same meaning
        :param scalar_encoding: str, 'multiscale' expands the elapsed time and the repeat count before the projection
        so that their magnitude survives the LayerNorm; 'projected' is CRAFT's LayerNorm(Linear(x))
        :param scalar_scales: tuple of float, the scales of the elapsed-time expansion, as multiples of the running
        time scale
        :param time_scale_momentum: float, momentum of the running scale of the elapsed times
        :param history_pooling: str, 'attention' is CRAFT's candidate-conditioned cross-attention over the history;
        'mean' is the ablation without target-aware matching: the history is averaged without looking at the
        candidate, and the candidate embedding is added to it, as the residual of the attention would
        """
        super(CRAFTV10, self).__init__()
        assert scalar_encoding in ('projected', 'multiscale'), f'unknown scalar_encoding {scalar_encoding}'
        assert history_pooling in ('attention', 'mean'), f'unknown history_pooling {history_pooling}'
        assert not (input_cat_time_intervals and scalar_encoding == 'multiscale'), \
            'the multiscale encoding covers the candidate time feature; input_cat_time_intervals is not supported with it'
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.n_nodes = n_nodes
        self.max_seq_length = max_seq_length
        self.use_pos = use_pos
        self.input_cat_time_intervals = input_cat_time_intervals
        self.output_cat_time_intervals = output_cat_time_intervals
        self.output_cat_repeat_times = output_cat_repeat_times
        self.emb_dropout_prob = emb_dropout_prob
        self.scalar_encoding = scalar_encoding
        self.history_pooling = history_pooling
        self.scalar_scales = tuple(scalar_scales)
        self.time_scale_momentum = time_scale_momentum
        self.eps = 1e-6
        self.device = device

        # ---- CRAFT backbone ----
        self.node_embedding = nn.Embedding(self.n_nodes + 1, self.hidden_size, padding_idx=0)
        if self.use_pos:
            self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        trm_input_dim = self.hidden_size * 2 if self.input_cat_time_intervals else self.hidden_size
        self.cross_attention = CrossAttention(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=trm_input_dim,
            inner_size=trm_input_dim * 4,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        output_dim = trm_input_dim
        # the multiscale encoding expands the elapsed time into one log feature per scale plus a never-interacted flag,
        # and the repeat count into its log and three thresholds
        multiscale = self.scalar_encoding == 'multiscale'
        time_input_dim = len(self.scalar_scales) + 1 if multiscale else 1
        repeat_input_dim = 4 if multiscale else 1
        if self.output_cat_time_intervals or self.input_cat_time_intervals:
            self.time_projection = MLP(num_layers=1, input_dim=time_input_dim, hidden_dim=self.hidden_size,
                                       output_dim=self.hidden_size, dropout=self.hidden_dropout_prob,
                                       use_act=True, skip_connection=skip_connection)
        if self.output_cat_time_intervals:
            output_dim += self.hidden_size
        if self.output_cat_repeat_times:
            self.repeat_times_projection = MLP(num_layers=1, input_dim=repeat_input_dim, hidden_dim=self.hidden_size,
                                               output_dim=self.hidden_size, dropout=self.hidden_dropout_prob,
                                               use_act=True, skip_connection=skip_connection)
            output_dim += self.hidden_size
        self.output_layer = MLP(num_layers=num_output_layer, input_dim=output_dim, hidden_dim=output_dim,
                                output_dim=1, dropout=self.hidden_dropout_prob, use_act=True,
                                skip_connection=skip_connection)
        if multiscale:
            # running scale of the elapsed times, updated during training only, so the scales adapt to the time unit
            # of each dataset
            self.register_buffer('time_scale', torch.ones(1))
            self.register_buffer('time_scale_initialized', torch.zeros(1))
            self.register_buffer('scalar_scales_tensor', torch.tensor(self.scalar_scales), persistent=False)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_time_intervals = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_repeat_times = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout(self.emb_dropout_prob)

        self.loss_type = loss_type
        if self.loss_type == 'BCE':
            self.loss_fct = nn.BCELoss()
        elif self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def set_min_idx(self, src_min_idx, dst_min_idx):
        self.src_min_idx = src_min_idx
        self.dst_min_idx = dst_min_idx

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_time_scale(self, elapsed_times: torch.Tensor, valid_mask: torch.Tensor):
        """
        running median of the observed elapsed times, updated during training only; written without host
        synchronization (no boolean indexing or .item()), so the GPU never waits on it
        """
        if self.training:
            with torch.no_grad():
                batch_scale = torch.nanmedian(elapsed_times.masked_fill(~valid_mask, float('nan'))).clamp(min=self.eps).view(1)
                updated = torch.where(self.time_scale_initialized > 0,
                                      self.time_scale_momentum * self.time_scale + (1.0 - self.time_scale_momentum) * batch_scale,
                                      batch_scale)
                # a batch with no valid history entry has a NaN median and leaves the scale unchanged
                has_valid = ~torch.isnan(batch_scale)
                self.time_scale.copy_(torch.where(has_valid, updated, self.time_scale))
                self.time_scale_initialized.copy_(torch.where(has_valid, torch.ones_like(self.time_scale_initialized),
                                                              self.time_scale_initialized))
        return self.time_scale.clamp(min=self.eps)

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst,
                dst_last_update_times):
        """
        score every candidate, node ids are raw ids and every tensor is on self.device
        :param src_neighb_seq: Tensor, shape (batch_size, max_seq_length), the source's recent history
        :param src_neighb_seq_len: Tensor, shape (batch_size, )
        :param neighbors_interact_times: Tensor, shape (batch_size, max_seq_length)
        :param cur_times: Tensor, shape (batch_size, ), prediction times
        :param test_dst: Tensor, shape (batch_size, num_candidates), column 0 is the positive candidate
        :param dst_last_update_times: Tensor, shape (batch_size, num_candidates), -100000 when unknown
        :return: Tensor, shape (batch_size, num_candidates)
        """
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates = test_dst.shape[1]
        multiscale = self.scalar_encoding == 'multiscale'

        # ---- reindex into the candidate id space, as CRAFT does ----
        src_neighb_seq = (src_neighb_seq - self.dst_min_idx + 1).masked_fill(src_neighb_seq - self.dst_min_idx + 1 < 0, 0)
        test_dst = test_dst - self.dst_min_idx + 1
        valid_mask = src_neighb_seq != 0

        # ---- elapsed time since the candidate's last interaction ----
        if self.output_cat_time_intervals:
            if multiscale:
                src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
                time_scale = self._get_time_scale(src_elapsed, valid_mask)
                never = (dst_last_update_times < -1).float()
                since_last = (cur_times.view(-1, 1) - dst_last_update_times).float().clamp(min=0.0) * (1.0 - never)
                expanded = torch.log1p(since_last.unsqueeze(-1) / (time_scale * self.scalar_scales_tensor))
                time_input = torch.cat([expanded * (1.0 - never).unsqueeze(-1), never.unsqueeze(-1)], dim=-1)
            else:
                # CRAFT: candidates with no past interaction get a sentinel elapsed time
                time_input = torch.where(dst_last_update_times < -1, torch.full_like(dst_last_update_times, -100000),
                                         cur_times.view(-1, 1) - dst_last_update_times).float().unsqueeze(-1)
            dst_node_time_intervals_feat = self.time_projection(time_input.view(-1, time_input.shape[-1])).view(
                batch_size, num_candidates, -1)
            dst_node_time_intervals_feat = self.dropout(self.LayerNorm_time_intervals(dst_node_time_intervals_feat))

        test_dst_emb = self.LayerNorm(self.node_embedding(test_dst).view(batch_size, -1, self.hidden_size))
        test_dst_emb = self.emb_dropout(test_dst_emb)

        # ---- the candidate's repeat count in the source's window ----
        if self.output_cat_repeat_times:
            repeat_times = ((test_dst.view(batch_size, num_candidates, 1) == src_neighb_seq.view(batch_size, 1, max_seq_length)) &
                            valid_mask.view(batch_size, 1, max_seq_length)).sum(dim=-1, keepdim=True).float()
            if multiscale:
                repeat_times = torch.cat([torch.log1p(repeat_times), (repeat_times >= 1).float(), (repeat_times >= 2).float(),
                                          (repeat_times >= 5).float()], dim=-1)
            repeat_times_feat = self.repeat_times_projection(repeat_times.view(-1, repeat_times.shape[-1])).view(
                batch_size, num_candidates, -1)
            repeat_times_feat = self.dropout(self.LayerNorm_repeat_times(repeat_times_feat))

        # ---- CRAFT: history representation ----
        neighb_emb = self.node_embedding(src_neighb_seq)
        if self.use_pos:
            position_ids = torch.arange(max_seq_length, dtype=torch.long, device=src_neighb_seq.device)
            neighb_emb = neighb_emb + self.position_embedding(position_ids.unsqueeze(0).expand_as(src_neighb_seq))
        input_emb = self.emb_dropout(self.LayerNorm(neighb_emb))
        if self.input_cat_time_intervals:
            src_time_intervals = (cur_times.view(-1, 1) - neighbors_interact_times).masked_fill(~valid_mask, -100000)
            src_neighb_time_embedding = self.time_projection(src_time_intervals.float().view(-1, 1)).view(
                batch_size, max_seq_length, -1)
            src_neighb_time_embedding = self.dropout(self.LayerNorm_time_intervals(src_neighb_time_embedding))
            input_emb = torch.cat([input_emb, src_neighb_time_embedding], dim=-1)

        # ---- CRAFT: candidate-conditioned attention over the source's history ----
        attention_mask = self.get_attention_mask(
            torch.ones(batch_size, num_candidates, device=src_neighb_seq.device), mask_b=valid_mask)
        if self.history_pooling == 'attention':
            context = self.cross_attention(test_dst_emb, attention_mask, input_emb, output_all_encoded_layers=False)[-1]
        else:
            # ablation: one summary of the history for all candidates, plus the candidate embedding
            pooled = (input_emb * valid_mask.unsqueeze(-1).float()).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            context = self.LayerNorm(pooled.unsqueeze(1) + test_dst_emb)

        output = context
        if self.output_cat_time_intervals:
            output = torch.cat([output, dst_node_time_intervals_feat], dim=-1).float()
        if self.output_cat_repeat_times:
            output = torch.cat([output, repeat_times_feat], dim=-1).float()
        return self.output_layer(output.view(-1, output.shape[-1])).view(batch_size, num_candidates)

    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = torch.bmm(mask_a.unsqueeze(1).transpose(1, 2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def compute_scores(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times):
        """
        move a batch to the device and score the candidates
        """
        return self.forward(src_neighb_seq=src_neighb_seq.to(self.device),
                            src_neighb_seq_len=src_neighb_seq_len.to(self.device),
                            neighbors_interact_times=src_neighb_interact_times.to(self.device),
                            cur_times=cur_pred_times.to(self.device),
                            test_dst=test_dst.to(self.device),
                            dst_last_update_times=dst_last_update_times.to(self.device))

    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                dst_last_update_times):
        """
        [0]src_neighb_seq: [B, L]
        [1]src_neighb_seq_len: [B]
        [2]src_neighb_interact_times: [B, L]
        [3]cur_pred_times: [B]
        [4]test_dst: [B, 1+num_negs], the positive candidate is at column 0
        [5]dst_last_update_times: [B, 1+num_negs]
        """
        logits = self.compute_scores(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times,
                                     test_dst, dst_last_update_times)
        if self.loss_type == 'BPR':
            return logits[:, 0].flatten(), logits[:, 1:].flatten()
        return logits[:, 0].sigmoid().flatten(), logits[:, 1:].sigmoid().flatten()

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times):
        """
        the objective CRAFT is trained with: one negative per positive, BPR or BCE
        """
        positive_probabilities, negative_probabilities = self.predict(
            src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
            dst_last_update_times)
        if self.loss_type == 'BPR':
            loss = self.loss_fct(positive_probabilities, negative_probabilities)
        elif self.loss_type == 'BCE':
            loss = self.loss_fct(torch.cat([positive_probabilities, negative_probabilities], dim=0),
                                 torch.cat([torch.ones_like(positive_probabilities),
                                            torch.zeros_like(negative_probabilities)], dim=0))
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented! Only BCE and BPR are supported!")
        predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
        labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
        return loss, predicts, labels
