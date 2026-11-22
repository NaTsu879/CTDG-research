import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP
from models.evidence_readouts import EvidenceReadouts, count_events_before  # noqa: F401, re-exported


class CRAFTV8(EvidenceReadouts, torch.nn.Module):
    """
    CRAFTV8: CRAFT with reciprocity, long-range memory and co-visit closure readouts.

    The gaps this closes
    --------------------
    0. The neighbor samplers symmetrize the graph, so CRAFT cannot tell "d contacted me and I have not
       answered" from "I contacted d". On directed non-bipartite data (uci, Flickr, YouTube) the shortest
       cycle a new edge s -> d can close is the 2-cycle d -> s -> d: a reply or a follow-back.
    1. CRAFT only sees the source's last num_neighbors interactions (120 on lastfm), and the average lastfm
       user has about 1,300. On the lastfm validation set 25.5% of the edges go to an item the user interacted
       with before, outside that window: CRAFT's repeat count is zero for them and its attention finds no
       match, so a long-time favorite looks like an item the user never touched.
    2. On unseen-dominant bipartite data the source never interacted with the candidate (GoogleLocal has
       no repeated edge at all, and its median user has only 5 interactions), so the evidence has to come
       from other users. The shortest cycle a bipartite graph allows has length 4: s - i - u - d, a user u
       who visited an item i that s visited also visited d. On the GoogleLocal validation set 58.8% of the
       positives close at least one such cycle against 1.9% of random candidates.

    The readouts (see models.evidence_readouts.EvidenceReadouts) go through a small MLP whose output is added to
    CRAFT's logit behind a gate that starts at zero, so the model starts at CRAFT. --no_reciprocity,
    --no_history_memory and --no_covisit ablate each readout, and with all three the model is CRAFT, which is the
    control run. The co-visit index is queried on the GPU with the model when it fits under --covisit_gpu_max_gb,
    and in host memory on the CPU otherwise.

    Training is untouched: one negative per positive from the same collision-checked sampler, the
    same BPR/BCE loss, validation on average precision and MRR at test only, as in the CRAFT paper.
    """

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=True,
                 input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=True,
                 num_output_layer=1, emb_dropout_prob=0.1, skip_connection=False, use_reciprocity=True,
                 use_history_memory=True, use_covisit=True, window_scales=(0.1, 1.0, 10.0, 100.0), num_decay_kernels=3,
                 time_scale_momentum=0.99, scalar_encoding='projected', scalar_scales=(0.01, 0.1, 1.0, 10.0, 100.0)):
        """
        the arguments up to skip_connection are CRAFT's, with the same meaning
        :param use_reciprocity: bool, whether the reciprocity readout feeds the correction
        :param use_history_memory: bool, whether the long-range memory readout feeds the correction
        :param use_covisit: bool, whether the co-visit closure readout feeds the correction
        :param window_scales: tuple of float, the time windows of the memory as multiples of the running time scale
        :param num_decay_kernels: int, number of learned time scales of the recency-weighted closure count
        :param time_scale_momentum: float, momentum of the running scale of the elapsed times
        :param scalar_encoding: str, how CRAFT's two scalar features (time since the candidate's last interaction, and
        its repeat count) are encoded: 'projected' is CRAFT's LayerNorm(Linear(x)), which keeps whether the evidence
        exists but collapses its magnitude once x is far from the scale of the bias; 'multiscale' first expands x into
        log(1 + x / s) at several scales s (multiples of the running time scale for the elapsed time) and indicator
        thresholds for the count, so the encoding keeps how recent and how frequent the evidence is
        :param scalar_scales: tuple of float, the scales of the multiscale time encoding, as multiples of the time scale
        """
        super(CRAFTV8, self).__init__()
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
        self.eps = 1e-6
        self.device = device
        assert scalar_encoding in ('projected', 'multiscale'), f'unknown scalar_encoding {scalar_encoding}'
        self.scalar_encoding = scalar_encoding
        self.scalar_scales = tuple(scalar_scales)

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
        # the multiscale encoding expands the elapsed time into one log feature per scale plus a never-active flag, and
        # the repeat count into its log and three indicator thresholds
        time_input_dim = len(self.scalar_scales) + 1 if self.scalar_encoding == 'multiscale' else 1
        repeat_input_dim = 4 if self.scalar_encoding == 'multiscale' else 1
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

        # ---- readouts, added to CRAFT's logit behind a gate that starts at zero ----
        self._init_readouts(use_reciprocity=use_reciprocity, use_history_memory=use_history_memory,
                            use_covisit=use_covisit, window_scales=window_scales,
                            num_decay_kernels=num_decay_kernels, time_scale_momentum=time_scale_momentum)
        if self.num_readout_features > 0:
            self.readout_head = MLP(num_layers=2, input_dim=self.num_readout_features, hidden_dim=self.hidden_size,
                                    output_dim=1, dropout=self.hidden_dropout_prob, use_act=True,
                                    skip_connection=False)
            self.readout_gate = nn.Parameter(torch.zeros(1))
        if self.num_readout_features > 0 or self.scalar_encoding == 'multiscale':
            self._init_time_scale()
        if self.scalar_encoding == 'multiscale':
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

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst,
                dst_last_update_times, src_node_ids=None, src_is_sender=None):
        """
        score every candidate, node ids are raw ids and every tensor is on self.device
        :param src_neighb_seq: Tensor, shape (batch_size, max_seq_length), the source's recent history
        :param src_neighb_seq_len: Tensor, shape (batch_size, )
        :param neighbors_interact_times: Tensor, shape (batch_size, max_seq_length)
        :param cur_times: Tensor, shape (batch_size, ), prediction times
        :param test_dst: Tensor, shape (batch_size, num_candidates), column 0 is the positive candidate
        :param dst_last_update_times: Tensor, shape (batch_size, num_candidates), -100000 when unknown
        :param src_node_ids: Tensor, shape (batch_size, ), raw ids of the sources, needed by the memory readout
        :param src_is_sender: Tensor, shape (batch_size, max_seq_length), whether the source initiated the
        interaction of that history entry, needed by the reciprocity readout
        :return: Tensor, shape (batch_size, num_candidates)
        """
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates = test_dst.shape[1]
        raw_src_neighb_seq, raw_test_dst = src_neighb_seq, test_dst

        # ---- reindex into the candidate id space, as CRAFT does ----
        src_neighb_seq = (src_neighb_seq - self.dst_min_idx + 1).masked_fill(src_neighb_seq - self.dst_min_idx + 1 < 0, 0)
        test_dst = test_dst - self.dst_min_idx + 1
        valid_mask = src_neighb_seq != 0
        if self.num_readout_features > 0 or self.scalar_encoding == 'multiscale':
            src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
            time_scale = self._get_time_scale(src_elapsed, valid_mask)

        # ---- CRAFT: elapsed time of the candidate's last interaction ----
        if self.output_cat_time_intervals and self.scalar_encoding == 'multiscale':
            never = (dst_last_update_times < -1).float()
            since_last = (cur_times.view(-1, 1) - dst_last_update_times).float().clamp(min=0.0) * (1.0 - never)
            expanded = torch.log1p(since_last.unsqueeze(-1) / (time_scale * self.scalar_scales_tensor))
            expanded = torch.cat([expanded * (1.0 - never).unsqueeze(-1), never.unsqueeze(-1)], dim=-1)
            dst_node_time_intervals_feat = self.time_projection(expanded.view(-1, expanded.shape[-1])).view(
                batch_size, num_candidates, -1)
            dst_node_time_intervals_feat = self.dropout(self.LayerNorm_time_intervals(dst_node_time_intervals_feat))
        elif self.output_cat_time_intervals:
            # candidates with no past interaction get CRAFT's sentinel, written without a host synchronization
            dst_last_update_intervals = torch.where(dst_last_update_times < -1,
                                                    torch.full_like(dst_last_update_times, -100000),
                                                    cur_times.view(-1, 1) - dst_last_update_times)
            dst_node_time_intervals_feat = self.time_projection(dst_last_update_intervals.float().view(-1, 1)).view(
                batch_size, num_candidates, -1)
            dst_node_time_intervals_feat = self.dropout(self.LayerNorm_time_intervals(dst_node_time_intervals_feat))

        test_dst_emb = self.LayerNorm(self.node_embedding(test_dst).view(batch_size, -1, self.hidden_size))
        test_dst_emb = self.emb_dropout(test_dst_emb)

        # ---- interactions of the source with each candidate, used by CRAFT's repeat count and by reciprocity ----
        candidate_matches = (test_dst.view(batch_size, num_candidates, 1) ==
                             src_neighb_seq.view(batch_size, 1, max_seq_length)) & valid_mask.view(batch_size, 1, max_seq_length)
        if self.output_cat_repeat_times:
            repeat_times = candidate_matches.sum(dim=-1, keepdim=True).float()
            if self.scalar_encoding == 'multiscale':
                repeat_times = torch.cat([torch.log1p(repeat_times), (repeat_times >= 1).float(), (repeat_times >= 2).float(),
                                          (repeat_times >= 5).float()], dim=-1)
            repeat_times_feat = self.repeat_times_projection(repeat_times.view(-1, repeat_times.shape[-1])).view(batch_size, num_candidates, -1)
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
        context = self.cross_attention(test_dst_emb, attention_mask, input_emb, output_all_encoded_layers=False)[-1]

        output = context
        if self.output_cat_time_intervals:
            output = torch.cat([output, dst_node_time_intervals_feat], dim=-1).float()
        if self.output_cat_repeat_times:
            output = torch.cat([output, repeat_times_feat], dim=-1).float()
        logits = self.output_layer(output.view(-1, output.shape[-1])).view(batch_size, num_candidates)

        # ---- additive correction from the readouts ----
        if self.num_readout_features > 0:
            readout_features = self.evidence_features(candidate_matches, raw_src_neighb_seq, valid_mask, raw_test_dst,
                                                      cur_times, src_elapsed, time_scale, src_node_ids, src_is_sender)
            readout = self.readout_head(readout_features).squeeze(dim=-1)
            logits = logits + self.readout_gate * readout
        return logits

    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = torch.bmm(mask_a.unsqueeze(1).transpose(1, 2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def compute_scores(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, src_node_ids=None, src_is_sender=None):
        """
        move a batch to the device and score the candidates
        """
        return self.forward(src_neighb_seq=src_neighb_seq.to(self.device),
                            src_neighb_seq_len=src_neighb_seq_len.to(self.device),
                            neighbors_interact_times=src_neighb_interact_times.to(self.device),
                            cur_times=cur_pred_times.to(self.device),
                            test_dst=test_dst.to(self.device),
                            dst_last_update_times=dst_last_update_times.to(self.device),
                            src_node_ids=src_node_ids.to(self.device) if src_node_ids is not None else None,
                            src_is_sender=src_is_sender.to(self.device) if src_is_sender is not None else None)

    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                dst_last_update_times, src_node_ids=None, src_is_sender=None):
        """
        [0]src_neighb_seq: [B, L]
        [1]src_neighb_seq_len: [B]
        [2]src_neighb_interact_times: [B, L]
        [3]cur_pred_times: [B]
        [4]test_dst: [B, 1+num_negs], the positive candidate is at column 0
        [5]dst_last_update_times: [B, 1+num_negs]
        [6]src_node_ids: [B]
        [7]src_is_sender: [B, L]
        """
        logits = self.compute_scores(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times,
                                     test_dst, dst_last_update_times, src_node_ids, src_is_sender)
        if self.loss_type == 'BPR':
            return logits[:, 0].flatten(), logits[:, 1:].flatten()
        return logits[:, 0].sigmoid().flatten(), logits[:, 1:].sigmoid().flatten()

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, src_node_ids=None, src_is_sender=None):
        """
        the objective CRAFT is trained with: one negative per positive, BPR or BCE
        """
        positive_probabilities, negative_probabilities = self.predict(
            src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
            dst_last_update_times, src_node_ids, src_is_sender)
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
