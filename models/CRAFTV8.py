import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP


class CRAFTV8(torch.nn.Module):
    """
    CRAFTV8: CRAFT with a long-range memory readout.

    The gap this closes
    -------------------
    CRAFT only sees the source's last num_neighbors interactions (120 on lastfm), and the average lastfm
    user has about 1,300. On the lastfm test set 68% of the edges go to an item that is inside that
    window, but another 22.5% go to an item the user has interacted with before, outside it. For those
    edges CRAFT's repeat count is zero and its attention finds no match, so a user's long-time favorite
    looks exactly like an item they have never touched. Widening the window is what the per-dataset
    budget rules out: the attention cost grows with it.

    The long-range memory readout
    -----------------------------
    Every interaction of the data is indexed once, sorted by (key, time) (see
    utils.utils.build_history_memory), so the number of events of a key in any time window before the
    prediction time t, and the time of the latest one, are binary searches on the device, whatever
    the length of the history. Two keys are indexed:
        the pair (s, d): how often, and how recently, the source interacted with the candidate over its
            whole history - log counts in windows [t - w_k, t), the log count over all time, whether the
            pair ever occurred and the log time since its latest occurrence
        the node d: how active the candidate has been with anyone - log counts in the same windows and
            over all time, its popularity momentum, which also speaks for candidates the source never met
    The windows w_k are fixed multiples of a running scale of the elapsed times, so they adapt to the
    time unit of each dataset. The features go through a small MLP whose output is added to CRAFT's
    logit behind a gate that starts at zero, so the model starts at CRAFT. With --no_history_memory the
    model is CRAFT, which is the control run. No extra neighbors are sampled.

    As with the neighbor samplers, the index used in training mode is built from the training edges only
    and the one used in evaluation mode from all edges, and only events strictly before the prediction
    time are counted, so nothing leaks from the future or from the evaluation splits.

    Training is untouched: one negative per positive from the same collision-checked sampler, the
    same BPR/BCE loss, validation on average precision and MRR at test only, as in the CRAFT paper.
    """

    MEMORY_KINDS = ('pair', 'node')
    MEMORY_TENSORS = ('keys', 'codes', 'unique_times', 'times')

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=True,
                 input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=True,
                 num_output_layer=1, emb_dropout_prob=0.1, skip_connection=False, use_history_memory=True,
                 window_scales=(0.1, 1.0, 10.0, 100.0), time_scale_momentum=0.99):
        """
        the arguments up to skip_connection are CRAFT's, with the same meaning
        :param use_history_memory: bool, whether the long-range memory readout is added to the logit
        :param window_scales: tuple of float, the time windows as multiples of the running time scale
        :param time_scale_momentum: float, momentum of the running scale of the elapsed times
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
        self.use_history_memory = use_history_memory
        self.window_scales = tuple(window_scales)
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
        if self.output_cat_time_intervals or self.input_cat_time_intervals:
            self.time_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size,
                                       output_dim=self.hidden_size, dropout=self.hidden_dropout_prob,
                                       use_act=True, skip_connection=skip_connection)
        if self.output_cat_time_intervals:
            output_dim += self.hidden_size
        if self.output_cat_repeat_times:
            self.repeat_times_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size,
                                               output_dim=self.hidden_size, dropout=self.hidden_dropout_prob,
                                               use_act=True, skip_connection=skip_connection)
            output_dim += self.hidden_size
        self.output_layer = MLP(num_layers=num_output_layer, input_dim=output_dim, hidden_dim=output_dim,
                                output_dim=1, dropout=self.hidden_dropout_prob, use_act=True,
                                skip_connection=skip_connection)

        # ---- long-range memory readout: pair windows + all time + ever + recency, node windows + all time ----
        if self.use_history_memory:
            self.num_memory_features = 2 * len(self.window_scales) + 4
            self.readout_head = MLP(num_layers=2, input_dim=self.num_memory_features, hidden_dim=self.hidden_size,
                                    output_dim=1, dropout=self.hidden_dropout_prob, use_act=True,
                                    skip_connection=False)
            self.readout_gate = nn.Parameter(torch.zeros(1))
            # running scale of the elapsed times, updated during training only, so the windows adapt to the
            # time unit of each dataset
            self.register_buffer('time_scale', torch.ones(1))
            self.register_buffer('time_scale_initialized', torch.zeros(1))
            # the memory indexes, set by set_history_memory; they are not part of the checkpoint, since they
            # are rebuilt from the data on every run
            for split in ('train', 'full'):
                for kind in self.MEMORY_KINDS:
                    for name in self.MEMORY_TENSORS:
                        self.register_buffer(f'{split}_{kind}_{name}', None, persistent=False)
            self.memory_num_ids = None

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

    def set_history_memory(self, train_memory: dict, full_memory: dict, num_ids: int):
        """
        :param train_memory: dict, as returned by utils.utils.build_history_memory on the training edges, used in
        training mode so that training never sees a validation or test edge
        :param full_memory: dict, the same on all edges, used in evaluation mode
        :param num_ids: int, the multiplier of the pair keys src * num_ids + dst used by both
        """
        self.memory_num_ids = num_ids
        for split, memory in (('train', train_memory), ('full', full_memory)):
            for kind in self.MEMORY_KINDS:
                for name, array in memory[kind].items():
                    setattr(self, f'{split}_{kind}_{name}', torch.from_numpy(array))

    def _get_time_scale(self, elapsed_times: torch.Tensor, valid_mask: torch.Tensor):
        """
        running median of the observed elapsed times, updated during training only
        """
        if self.training and valid_mask.any():
            with torch.no_grad():
                batch_scale = elapsed_times[valid_mask].median().clamp(min=self.eps).view(1)
                if self.time_scale_initialized.item() == 0:
                    self.time_scale.copy_(batch_scale)
                    self.time_scale_initialized.fill_(1.0)
                else:
                    self.time_scale.mul_(self.time_scale_momentum).add_((1.0 - self.time_scale_momentum) * batch_scale)
        return self.time_scale.clamp(min=self.eps)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def count_events(self, kind: str, query_keys: torch.Tensor, query_times: torch.Tensor):
        """
        number of events of each key strictly before each time, in the index of the current mode
        :param kind: str, 'pair' or 'node'
        :param query_keys: Tensor of int64, any shape broadcastable with query_times
        :param query_times: Tensor of float64
        :return: counts, and the position after the latest counted event in the index, both broadcast
        """
        split = 'train' if self.training else 'full'
        keys = getattr(self, f'{split}_{kind}_keys')
        if keys is None or keys.numel() == 0:
            shape = torch.broadcast_shapes(query_keys.shape, query_times.shape)
            return torch.zeros(shape, dtype=torch.long, device=query_keys.device), None
        codes = getattr(self, f'{split}_{kind}_codes')
        unique_times = getattr(self, f'{split}_{kind}_unique_times')
        num_ranks = unique_times.numel() + 1
        key_idx = torch.searchsorted(keys, query_keys.contiguous()).clamp(max=keys.numel() - 1)
        found = keys[key_idx] == query_keys
        # how many distinct event times are strictly earlier than the query time
        time_rank = torch.searchsorted(unique_times, query_times.contiguous())
        lo = torch.searchsorted(codes, (key_idx * num_ranks).contiguous())
        hi = torch.searchsorted(codes, (key_idx * num_ranks + time_rank).contiguous())
        return (hi - lo) * found.long(), hi

    def memory_features(self, src_node_ids: torch.Tensor, test_dst: torch.Tensor, cur_times: torch.Tensor,
                        time_scale: torch.Tensor):
        """
        long-range statistics of the pair (source, candidate) and of the candidate, on raw node ids
        :param src_node_ids: Tensor, shape (batch_size, )
        :param test_dst: Tensor, shape (batch_size, num_candidates)
        :param cur_times: Tensor, shape (batch_size, )
        :param time_scale: Tensor, shape (1, )
        :return: Tensor, shape (batch_size, num_candidates, num_memory_features)
        """
        split = 'train' if self.training else 'full'
        t = cur_times.double().view(-1, 1, 1)
        windows = time_scale.double() * torch.tensor(self.window_scales, dtype=torch.float64, device=t.device)
        # query times: the window starts, then the prediction time itself
        query_times = torch.cat([t - windows.view(1, 1, -1), t], dim=-1)

        def window_and_total_counts(kind, query_keys):
            counts, hi = self.count_events(kind, query_keys.unsqueeze(-1), query_times)
            total = counts[..., -1]
            in_window = (total.unsqueeze(-1) - counts[..., :-1]).float()
            return in_window, total.float(), hi

        pair_keys = src_node_ids.long().view(-1, 1) * self.memory_num_ids + test_dst.long()
        pair_windows, pair_total, pair_hi = window_and_total_counts('pair', pair_keys)
        pair_ever = (pair_total > 0).float()
        if pair_hi is None:
            pair_recency = torch.zeros_like(pair_total)
        else:
            pair_times = getattr(self, f'{split}_pair_times')
            latest = pair_times[(pair_hi[..., -1] - 1).clamp(min=0)]
            pair_recency = ((t.squeeze(-1) - latest).float().clamp(min=0.0) / time_scale) * pair_ever
        node_windows, node_total, _ = window_and_total_counts('node', test_dst.long())
        return torch.cat([
            torch.log1p(pair_windows), torch.log1p(pair_total).unsqueeze(-1), pair_ever.unsqueeze(-1),
            torch.log1p(pair_recency).unsqueeze(-1),
            torch.log1p(node_windows), torch.log1p(node_total).unsqueeze(-1),
        ], dim=-1)

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst,
                dst_last_update_times, src_node_ids=None):
        """
        score every candidate, node ids are raw ids and every tensor is on self.device
        :param src_neighb_seq: Tensor, shape (batch_size, max_seq_length), the source's recent history
        :param src_neighb_seq_len: Tensor, shape (batch_size, )
        :param neighbors_interact_times: Tensor, shape (batch_size, max_seq_length)
        :param cur_times: Tensor, shape (batch_size, ), prediction times
        :param test_dst: Tensor, shape (batch_size, num_candidates), column 0 is the positive candidate
        :param dst_last_update_times: Tensor, shape (batch_size, num_candidates), -100000 when unknown
        :param src_node_ids: Tensor, shape (batch_size, ), raw ids of the sources, needed by the memory readout
        :return: Tensor, shape (batch_size, num_candidates)
        """
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates = test_dst.shape[1]
        raw_test_dst = test_dst

        # ---- reindex into the candidate id space, as CRAFT does ----
        src_neighb_seq = (src_neighb_seq - self.dst_min_idx + 1).masked_fill(src_neighb_seq - self.dst_min_idx + 1 < 0, 0)
        test_dst = test_dst - self.dst_min_idx + 1
        valid_mask = src_neighb_seq != 0

        # ---- CRAFT: elapsed time of the candidate's last interaction ----
        if self.output_cat_time_intervals:
            dst_last_update_intervals = cur_times.view(-1, 1) - dst_last_update_times
            dst_last_update_intervals[dst_last_update_times < -1] = -100000
            dst_node_time_intervals_feat = self.time_projection(dst_last_update_intervals.float().view(-1, 1)).view(
                batch_size, num_candidates, -1)
            dst_node_time_intervals_feat = self.dropout(self.LayerNorm_time_intervals(dst_node_time_intervals_feat))

        test_dst_emb = self.LayerNorm(self.node_embedding(test_dst).view(batch_size, -1, self.hidden_size))
        test_dst_emb = self.emb_dropout(test_dst_emb)

        # ---- CRAFT: how many times the source interacted with each candidate, within the window ----
        if self.output_cat_repeat_times:
            repeat_times = ((test_dst.view(batch_size, num_candidates, 1) == src_neighb_seq.view(batch_size, 1, max_seq_length)) &
                            valid_mask.view(batch_size, 1, max_seq_length)).sum(dim=-1, keepdim=True).float()
            repeat_times_feat = self.repeat_times_projection(repeat_times.view(-1, 1)).view(batch_size, num_candidates, -1)
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

        # ---- additive correction from the long-range memory readout ----
        if self.use_history_memory:
            src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
            time_scale = self._get_time_scale(src_elapsed, valid_mask)
            if src_node_ids is None or self.memory_num_ids is None:
                readout_features = torch.zeros(batch_size, num_candidates, self.num_memory_features, device=logits.device)
            else:
                readout_features = self.memory_features(src_node_ids, raw_test_dst, cur_times, time_scale)
            logits = logits + self.readout_gate * self.readout_head(readout_features).squeeze(dim=-1)
        return logits

    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = torch.bmm(mask_a.unsqueeze(1).transpose(1, 2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def compute_scores(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, src_node_ids=None):
        """
        move a batch to the device and score the candidates
        """
        return self.forward(src_neighb_seq=src_neighb_seq.to(self.device),
                            src_neighb_seq_len=src_neighb_seq_len.to(self.device),
                            neighbors_interact_times=src_neighb_interact_times.to(self.device),
                            cur_times=cur_pred_times.to(self.device),
                            test_dst=test_dst.to(self.device),
                            dst_last_update_times=dst_last_update_times.to(self.device),
                            src_node_ids=src_node_ids.to(self.device) if src_node_ids is not None else None)

    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                dst_last_update_times, src_node_ids=None):
        """
        [0]src_neighb_seq: [B, L]
        [1]src_neighb_seq_len: [B]
        [2]src_neighb_interact_times: [B, L]
        [3]cur_pred_times: [B]
        [4]test_dst: [B, 1+num_negs], the positive candidate is at column 0
        [5]dst_last_update_times: [B, 1+num_negs]
        [6]src_node_ids: [B]
        """
        logits = self.compute_scores(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times,
                                     test_dst, dst_last_update_times, src_node_ids)
        if self.loss_type == 'BPR':
            return logits[:, 0].flatten(), logits[:, 1:].flatten()
        return logits[:, 0].sigmoid().flatten(), logits[:, 1:].sigmoid().flatten()

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, src_node_ids=None):
        """
        the objective CRAFT is trained with: one negative per positive, BPR or BCE
        """
        positive_probabilities, negative_probabilities = self.predict(
            src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
            dst_last_update_times, src_node_ids)
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
