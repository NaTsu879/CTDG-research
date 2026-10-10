import math
import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP


def count_events_before(index: dict, query_keys: torch.Tensor, query_times: torch.Tensor):
    """
    number of indexed events of each key strictly before each time
    :param index: dict of tensors keys, codes and unique_times, as built by utils.utils.build_event_index, on the
    device of the queries
    :param query_keys: Tensor of int64, any shape broadcastable with query_times
    :param query_times: Tensor of float64
    :return: counts, and the position after the latest counted event in the index, both broadcast
    """
    keys, codes, unique_times = index['keys'], index['codes'], index['unique_times']
    if keys.numel() == 0:
        shape = torch.broadcast_shapes(query_keys.shape, query_times.shape)
        return torch.zeros(shape, dtype=torch.long, device=query_keys.device), None
    num_ranks = unique_times.numel() + 1
    key_idx = torch.searchsorted(keys, query_keys.contiguous()).clamp(max=keys.numel() - 1)
    found = keys[key_idx] == query_keys
    # how many distinct event times are strictly earlier than the query time
    time_rank = torch.searchsorted(unique_times, query_times.contiguous())
    lo = torch.searchsorted(codes, (key_idx * num_ranks).contiguous())
    hi = torch.searchsorted(codes, (key_idx * num_ranks + time_rank).contiguous())
    return (hi - lo) * found.long(), hi


class CRAFTV8(torch.nn.Module):
    """
    CRAFTV8: CRAFT with reciprocity, long-range memory and co-visit closure readouts.

    The gaps this closes
    --------------------
    0. The neighbor samplers symmetrize the graph, so CRAFT cannot tell "d contacted me and I have not
       answered" from "I contacted d". On directed non-bipartite data (uci, Flickr, YouTube) the shortest
       cycle a new edge s -> d can close is the 2-cycle d -> s -> d: a reply or a follow-back.
    1. CRAFT only sees the source's last num_neighbors interactions (120 on lastfm), and the average lastfm
       user has about 1,300. On the lastfm test set 22.5% of the edges go to an item the user interacted
       with before, outside that window: CRAFT's repeat count is zero for them and its attention finds no
       match, so a long-time favorite looks like an item the user never touched.
    2. On unseen-dominant bipartite data the source never interacted with the candidate (GoogleLocal has
       no repeated edge at all, and its median user has only 5 interactions), so the evidence has to come
       from other users. The shortest cycle a bipartite graph allows has length 4: s - i - u - d, a user u
       who visited an item i that s visited also visited d. On the GoogleLocal test set 58% of the
       positives close at least one such cycle against 5% of the official negatives, and counting them
       alone ranks the positive at MRR 0.45.

    The readouts
    ------------
    Every statistic is read from indexes built once from the data and sorted by (key, time) (see
    utils.utils.build_history_memory and build_covisit_index), so each one is a few binary searches on the
    history CRAFT already samples; no extra neighbors are sampled.
        reciprocity, from the direction of each history entry (whether s sent or received it, see
            utils.utils.get_edge_directions): for each direction, the log time since the latest interaction
            with the candidate, whether one exists and the log count. On bipartite data the received half is
            zero and the readout relies on the other two.
        long-range memory, on the pair (s, d) and on the node d: log counts in windows [t - w_k, t) and over
            all time, whether the pair ever occurred and the log time since its latest occurrence. The
            windows are fixed multiples of a running scale of the elapsed times.
        co-visit closure: a co-visit (i, d) is a user interacting with both i and d within `window`
            interactions of each other, known from the later of the two. For every item i_l of the
            source's history, c_l = number of co-visits (i_l, d) before t and its cosine
            cos_l = c_l / sqrt((1 + pop(i_l)) (1 + pop(d))), which gives the sum and max of c_l, the share of
            history items with c_l > 0, the sum and max of cos_l, and the sums of c_l and cos_l weighted by
            exp(-dt_l / tau_k), how recently s visited i_l, one per learned tau_k. On the official test
            negatives these counts alone rank the positive at MRR 0.50 (GoogleLocal), 0.64 (Yelp, Taobao)
            and 0.31 (ML-20M, where only the recency-weighted cosine separates the positive).
    The features go through a small MLP whose output is added to CRAFT's logit behind a gate that starts at
    zero, so the model starts at CRAFT. --no_reciprocity, --no_history_memory and --no_covisit ablate each
    readout, and with all three the model is CRAFT, which is the control run.

    As with the neighbor samplers, the indexes used in training mode are built from the training edges only
    and the ones used in evaluation mode from all edges, and only events strictly before the prediction time
    are counted, so nothing leaks from the future or from the evaluation splits. The co-visit index can be
    large, so it is kept in host memory and queried on the CPU.

    Training is untouched: one negative per positive from the same collision-checked sampler, the
    same BPR/BCE loss, validation on average precision and MRR at test only, as in the CRAFT paper.
    """

    MEMORY_KINDS = ('pair', 'node')
    MEMORY_TENSORS = ('keys', 'codes', 'unique_times', 'times')

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=True,
                 input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=True,
                 num_output_layer=1, emb_dropout_prob=0.1, skip_connection=False, use_reciprocity=True,
                 use_history_memory=True, use_covisit=True, window_scales=(0.1, 1.0, 10.0, 100.0), num_decay_kernels=3,
                 time_scale_momentum=0.99):
        """
        the arguments up to skip_connection are CRAFT's, with the same meaning
        :param use_reciprocity: bool, whether the reciprocity readout feeds the correction
        :param use_history_memory: bool, whether the long-range memory readout feeds the correction
        :param use_covisit: bool, whether the co-visit closure readout feeds the correction
        :param window_scales: tuple of float, the time windows of the memory as multiples of the running time scale
        :param num_decay_kernels: int, number of learned time scales of the recency-weighted closure count
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
        self.use_reciprocity = use_reciprocity
        self.use_history_memory = use_history_memory
        self.use_covisit = use_covisit
        self.window_scales = tuple(window_scales)
        self.num_decay_kernels = num_decay_kernels
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

        # ---- readouts ----
        self.num_readout_features = 0
        if self.use_reciprocity:
            # elapsed time, existence and count of interactions with the candidate, once for each direction
            self.num_readout_features += 6
        if self.use_history_memory:
            # pair windows + all time + ever + recency, node windows + all time
            self.num_readout_features += 2 * len(self.window_scales) + 4
            # the memory indexes, set by set_history_memory; they are not part of the checkpoint, since they
            # are rebuilt from the data on every run
            for split in ('train', 'full'):
                for kind in self.MEMORY_KINDS:
                    for name in self.MEMORY_TENSORS:
                        self.register_buffer(f'{split}_{kind}_{name}', None, persistent=False)
            self.memory_num_ids = None
        if self.use_covisit:
            # sum, max, share, cosine sum and max, recency-weighted sum and cosine sum per kernel
            self.num_readout_features += 2 * self.num_decay_kernels + 5
            self.covisit_log_tau = nn.Parameter(torch.linspace(math.log(0.1), math.log(10.0), self.num_decay_kernels))
            # the co-visit indexes, set by set_covisit_index and kept on the CPU (not buffers, so that moving the
            # model to the GPU leaves them in host memory)
            self.covisit_index = {'train': None, 'full': None}
            self.covisit_num_ids = None
        if self.num_readout_features > 0:
            self.readout_head = MLP(num_layers=2, input_dim=self.num_readout_features, hidden_dim=self.hidden_size,
                                    output_dim=1, dropout=self.hidden_dropout_prob, use_act=True,
                                    skip_connection=False)
            self.readout_gate = nn.Parameter(torch.zeros(1))
            # running scale of the elapsed times, updated during training only, so the windows and kernels adapt
            # to the time unit of each dataset
            self.register_buffer('time_scale', torch.ones(1))
            self.register_buffer('time_scale_initialized', torch.zeros(1))

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

    def set_covisit_index(self, train_index: dict, full_index: dict, num_ids: int):
        """
        :param train_index: dict, as returned by utils.utils.build_covisit_index on the training edges, used in
        training mode so that training never sees a validation or test edge
        :param full_index: dict, the same on all edges, used in evaluation mode
        :param num_ids: int, the multiplier of the pair keys min * num_ids + max used by both
        """
        self.covisit_num_ids = num_ids
        for split, index in (('train', train_index), ('full', full_index)):
            self.covisit_index[split] = {kind: {name: torch.from_numpy(array) for name, array in index[kind].items()}
                                         for kind in index}

    def _memory_index(self, kind: str):
        split = 'train' if self.training else 'full'
        return {name: getattr(self, f'{split}_{kind}_{name}') for name in self.MEMORY_TENSORS}

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

    def reciprocity_features(self, candidate_matches: torch.Tensor, src_is_sender: torch.Tensor,
                             normalized_elapsed: torch.Tensor):
        """
        per-direction elapsed times and counts of the past interactions between the source and each candidate
        :param candidate_matches: Tensor, shape (batch_size, num_candidates, max_seq_length), whether that
        history entry is an interaction with that candidate
        :param src_is_sender: Tensor, shape (batch_size, max_seq_length), whether the source initiated it
        :param normalized_elapsed: Tensor, shape (batch_size, max_seq_length), elapsed time / time scale
        :return: Tensor, shape (batch_size, num_candidates, 6)
        """
        elapsed = normalized_elapsed.unsqueeze(1)
        large = torch.full_like(elapsed, 1e9)
        features = []
        for direction_mask in (src_is_sender.unsqueeze(1), ~src_is_sender.unsqueeze(1)):
            directed_matches = candidate_matches & direction_mask
            exists = directed_matches.any(dim=-1)
            # the smallest elapsed time is the most recent interaction in this direction
            last_elapsed = torch.where(directed_matches, elapsed, large).min(dim=-1).values * exists.float()
            features += [torch.log1p(last_elapsed).unsqueeze(-1),
                         exists.float().unsqueeze(-1),
                         torch.log1p(directed_matches.sum(dim=-1).float()).unsqueeze(-1)]
        return torch.cat(features, dim=-1)

    def memory_features(self, src_node_ids: torch.Tensor, test_dst: torch.Tensor, cur_times: torch.Tensor,
                        time_scale: torch.Tensor):
        """
        long-range statistics of the pair (source, candidate) and of the candidate, on raw node ids
        :param src_node_ids: Tensor, shape (batch_size, )
        :param test_dst: Tensor, shape (batch_size, num_candidates)
        :param cur_times: Tensor, shape (batch_size, )
        :param time_scale: Tensor, shape (1, )
        :return: Tensor, shape (batch_size, num_candidates, 2 * len(window_scales) + 4)
        """
        t = cur_times.double().view(-1, 1, 1)
        windows = time_scale.double() * torch.tensor(self.window_scales, dtype=torch.float64, device=t.device)
        # query times: the window starts, then the prediction time itself
        query_times = torch.cat([t - windows.view(1, 1, -1), t], dim=-1)

        def window_and_total_counts(kind, query_keys):
            counts, hi = count_events_before(self._memory_index(kind), query_keys.unsqueeze(-1), query_times)
            total = counts[..., -1]
            in_window = (total.unsqueeze(-1) - counts[..., :-1]).float()
            return in_window, total.float(), hi

        pair_keys = src_node_ids.long().view(-1, 1) * self.memory_num_ids + test_dst.long()
        pair_windows, pair_total, pair_hi = window_and_total_counts('pair', pair_keys)
        pair_ever = (pair_total > 0).float()
        if pair_hi is None:
            pair_recency = torch.zeros_like(pair_total)
        else:
            latest = self._memory_index('pair')['times'][(pair_hi[..., -1] - 1).clamp(min=0)]
            pair_recency = ((t.squeeze(-1) - latest).float().clamp(min=0.0) / time_scale) * pair_ever
        node_windows, node_total, _ = window_and_total_counts('node', test_dst.long())
        return torch.cat([
            torch.log1p(pair_windows), torch.log1p(pair_total).unsqueeze(-1), pair_ever.unsqueeze(-1),
            torch.log1p(pair_recency).unsqueeze(-1),
            torch.log1p(node_windows), torch.log1p(node_total).unsqueeze(-1),
        ], dim=-1)

    def covisit_features(self, items: torch.Tensor, test_dst: torch.Tensor, cur_times: torch.Tensor,
                         normalized_item_elapsed: torch.Tensor):
        """
        4-cycle closure statistics between the source's history items and each candidate, on raw node ids
        :param items: Tensor, shape (batch_size, max_seq_length), raw ids of the history items, 0 is padding
        :param test_dst: Tensor, shape (batch_size, num_candidates), raw ids
        :param cur_times: Tensor, shape (batch_size, )
        :param normalized_item_elapsed: Tensor, shape (batch_size, max_seq_length), time since the source visited
        each history item, divided by the time scale
        :return: Tensor, shape (batch_size, num_candidates, 2 * num_decay_kernels + 5)
        """
        index = self.covisit_index['train' if self.training else 'full']
        device = items.device
        items_cpu, dst_cpu = items.long().cpu(), test_dst.long().cpu()
        times_cpu = cur_times.double().cpu()
        item_valid = (items != 0).unsqueeze(1)

        # co-visits (i_l, d) before t, keyed by the unordered pair
        h, c = items_cpu.unsqueeze(1), dst_cpu.unsqueeze(2)
        pair_keys = torch.minimum(h, c) * self.covisit_num_ids + torch.maximum(h, c)
        counts, _ = count_events_before(index['pair'], pair_keys, times_cpu.view(-1, 1, 1))
        counts = counts.to(device).float() * item_valid.float()
        # popularity of the history items and of the candidates before t
        item_pop, _ = count_events_before(index['node'], items_cpu, times_cpu.view(-1, 1))
        dst_pop, _ = count_events_before(index['node'], dst_cpu, times_cpu.view(-1, 1))
        item_pop, dst_pop = item_pop.to(device).float(), dst_pop.to(device).float()

        closed = (counts > 0).float()
        num_items = item_valid.float().sum(dim=-1).clamp(min=1.0)
        # cosine: on dense data (ML-20M) almost every candidate shares raw co-visits with the history, and only the
        # count relative to the popularity of both items separates the positive
        cosine = counts / torch.sqrt((1.0 + item_pop).unsqueeze(1) * (1.0 + dst_pop).unsqueeze(2))
        tau = self.covisit_log_tau.exp().clamp(min=1e-3)
        decay = torch.exp(-normalized_item_elapsed.unsqueeze(-1) / tau.view(1, 1, -1)) * (items != 0).unsqueeze(-1).float()
        # cosine values are small, so they are scaled before the log to spread them over a useful range
        return torch.cat([
            torch.log1p(counts.sum(dim=-1)).unsqueeze(-1),
            torch.log1p(counts.amax(dim=-1)).unsqueeze(-1),
            (closed.sum(dim=-1) / num_items).unsqueeze(-1),
            torch.log1p(1000.0 * cosine.sum(dim=-1)).unsqueeze(-1),
            torch.log1p(1000.0 * cosine.amax(dim=-1)).unsqueeze(-1),
            torch.log1p(torch.einsum('bcl,blk->bck', counts, decay)),
            torch.log1p(1000.0 * torch.einsum('bcl,blk->bck', cosine, decay)),
        ], dim=-1)

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

        # ---- CRAFT: elapsed time of the candidate's last interaction ----
        if self.output_cat_time_intervals:
            dst_last_update_intervals = cur_times.view(-1, 1) - dst_last_update_times
            dst_last_update_intervals[dst_last_update_times < -1] = -100000
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

        # ---- additive correction from the readouts ----
        if self.num_readout_features > 0:
            src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
            time_scale = self._get_time_scale(src_elapsed, valid_mask)
            readout_features = []
            if self.use_reciprocity:
                if src_is_sender is None:
                    readout_features.append(torch.zeros(batch_size, num_candidates, 6, device=logits.device))
                else:
                    readout_features.append(self.reciprocity_features(candidate_matches, src_is_sender,
                                                                      src_elapsed / time_scale))
            if self.use_history_memory:
                if src_node_ids is None or self.memory_num_ids is None:
                    readout_features.append(torch.zeros(batch_size, num_candidates, 2 * len(self.window_scales) + 4,
                                                        device=logits.device))
                else:
                    readout_features.append(self.memory_features(src_node_ids, raw_test_dst, cur_times, time_scale))
            if self.use_covisit:
                if self.covisit_index['train' if self.training else 'full'] is None:
                    readout_features.append(torch.zeros(batch_size, num_candidates, 2 * self.num_decay_kernels + 5,
                                                        device=logits.device))
                else:
                    readout_features.append(self.covisit_features(raw_src_neighb_seq * valid_mask, raw_test_dst,
                                                                  cur_times, src_elapsed / time_scale))
            readout = self.readout_head(torch.cat(readout_features, dim=-1)).squeeze(dim=-1)
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
