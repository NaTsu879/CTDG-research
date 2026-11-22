import math
import torch
from torch import nn


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


class EvidenceReadouts:
    """
    The three readouts of evidence outside the source's history window, shared by CRAFTV8 and CRAFTV9.

        reciprocity, from the direction of each history entry (whether s sent or received it, see
            utils.utils.get_edge_directions): for each direction, the log time since the latest interaction
            with the candidate, whether one exists and the log count. On bipartite data the received half is
            zero.
        long-range memory, on the pair (s, d) and on the node d: log counts in windows [t - w_k, t) and over
            all time, whether the pair ever occurred and the log time since its latest occurrence. The
            windows are fixed multiples of a running scale of the elapsed times.
        co-visit closure: a co-visit (i, d) is a user interacting with both i and d within `window`
            interactions of each other, known from the later of the two. For every item i_l of the
            source's history, c_l = number of co-visits (i_l, d) before t and its cosine
            cos_l = c_l / sqrt((1 + pop(i_l)) (1 + pop(d))), which gives the sum and max of c_l, the share of
            history items with c_l > 0, the sum and max of cos_l, and the sums of c_l and cos_l weighted by
            exp(-dt_l / tau_k), how recently s visited i_l, one per learned tau_k.

    Every statistic is read from indexes built once from the data and sorted by (key, time) (see
    utils.utils.build_history_memory and build_covisit_index), so each one is a few binary searches; no extra
    neighbors are sampled. As with the neighbor samplers, the indexes used in training mode are built from the
    training edges only and the ones used in evaluation mode from all edges, and only events strictly before
    the prediction time are counted, so nothing leaks from the future or from the evaluation splits.

    A model using this mixin calls _init_readouts in its constructor, then evidence_features in its forward.
    """

    MEMORY_KINDS = ('pair', 'node')
    MEMORY_TENSORS = ('keys', 'codes', 'unique_times', 'times')
    COVISIT_TENSORS = ('keys', 'codes', 'unique_times')

    def _init_readouts(self, use_reciprocity: bool, use_history_memory: bool, use_covisit: bool,
                       window_scales: tuple, num_decay_kernels: int, time_scale_momentum: float):
        """
        set up the readouts and count their features in self.num_readout_features
        :param use_reciprocity: bool, whether the reciprocity readout is computed
        :param use_history_memory: bool, whether the long-range memory readout is computed
        :param use_covisit: bool, whether the co-visit closure readout is computed
        :param window_scales: tuple of float, the time windows of the memory as multiples of the running time scale
        :param num_decay_kernels: int, number of learned time scales of the recency-weighted closure count
        :param time_scale_momentum: float, momentum of the running scale of the elapsed times
        """
        self.use_reciprocity = use_reciprocity
        self.use_history_memory = use_history_memory
        self.use_covisit = use_covisit
        self.window_scales = tuple(window_scales)
        self.num_decay_kernels = num_decay_kernels
        self.time_scale_momentum = time_scale_momentum
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
            # the window multiples as a buffer, so they are not copied to the device on every batch
            self.register_buffer('window_scales_tensor', torch.tensor(self.window_scales, dtype=torch.float64), persistent=False)
        if self.use_covisit:
            # sum, max, share, cosine sum and max, recency-weighted sum and cosine sum per kernel
            self.num_readout_features += 2 * self.num_decay_kernels + 5
            self.covisit_log_tau = nn.Parameter(torch.linspace(math.log(0.1), math.log(10.0), self.num_decay_kernels))
            # the co-visit indexes, set by set_covisit_index: as buffers, so that they move to the GPU with the model
            # and are queried there, or, when they are too large for the GPU, in host memory and queried on the CPU
            for split in ('train', 'full'):
                for kind in self.MEMORY_KINDS:
                    for name in self.COVISIT_TENSORS:
                        self.register_buffer(f'covisit_{split}_{kind}_{name}', None, persistent=False)
            self.covisit_host_index = {'train': None, 'full': None}
            self.covisit_num_ids = None

    def _init_time_scale(self):
        """
        running scale of the elapsed times, updated during training only, so the windows and kernels adapt to the
        time unit of each dataset
        """
        self.register_buffer('time_scale', torch.ones(1))
        self.register_buffer('time_scale_initialized', torch.zeros(1))

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

    def set_covisit_index(self, train_index: dict, full_index: dict, num_ids: int, max_device_gb: float = None):
        """
        :param train_index: dict, as returned by utils.utils.build_covisit_index on the training edges, used in
        training mode so that training never sees a validation or test edge
        :param full_index: dict, the same on all edges, used in evaluation mode
        :param num_ids: int, the multiplier of the pair keys min * num_ids + max used by both
        :param max_device_gb: float, the largest size of both indexes together that is moved to the model's device;
        larger indexes stay in host memory and are queried on the CPU. None always moves them
        """
        self.covisit_num_ids = num_ids
        size_gb = sum(array.nbytes for index in (train_index, full_index) for kind in index.values()
                      for array in kind.values()) / 1e9
        on_device = max_device_gb is None or size_gb <= max_device_gb
        print(f'co-visit indexes: {size_gb:.2f} GB, queried on the {"model device" if on_device else "CPU"}')
        for split, index in (('train', train_index), ('full', full_index)):
            tensors = {kind: {name: torch.from_numpy(array) for name, array in index[kind].items()} for kind in index}
            if on_device:
                for kind in tensors:
                    for name, tensor in tensors[kind].items():
                        setattr(self, f'covisit_{split}_{kind}_{name}', tensor)
            else:
                self.covisit_host_index[split] = tensors

    def _covisit_index(self):
        """
        the co-visit index of the current mode, on the model's device or in host memory, or None when it is not set
        """
        split = 'train' if self.training else 'full'
        if getattr(self, f'covisit_{split}_pair_keys') is not None:
            return {kind: {name: getattr(self, f'covisit_{split}_{kind}_{name}') for name in self.COVISIT_TENSORS}
                    for kind in self.MEMORY_KINDS}
        return self.covisit_host_index[split]

    def _memory_index(self, kind: str):
        split = 'train' if self.training else 'full'
        return {name: getattr(self, f'{split}_{kind}_{name}') for name in self.MEMORY_TENSORS}

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
        windows = time_scale.double() * self.window_scales_tensor
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
        index = self._covisit_index()
        device = items.device
        # the queries run where the index lives: on the model's device when it fits there, otherwise on the CPU
        index_device = index['pair']['keys'].device
        query_items, query_dst = items.long().to(index_device), test_dst.long().to(index_device)
        query_times = cur_times.double().to(index_device)
        item_valid = (items != 0).unsqueeze(1)

        # co-visits (i_l, d) before t, keyed by the unordered pair
        h, c = query_items.unsqueeze(1), query_dst.unsqueeze(2)
        pair_keys = torch.minimum(h, c) * self.covisit_num_ids + torch.maximum(h, c)
        counts, _ = count_events_before(index['pair'], pair_keys, query_times.view(-1, 1, 1))
        counts = counts.to(device).float() * item_valid.float()
        # popularity of the history items and of the candidates before t
        item_pop, _ = count_events_before(index['node'], query_items, query_times.view(-1, 1))
        dst_pop, _ = count_events_before(index['node'], query_dst, query_times.view(-1, 1))
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

    def evidence_features(self, candidate_matches: torch.Tensor, raw_src_neighb_seq: torch.Tensor,
                          valid_mask: torch.Tensor, raw_test_dst: torch.Tensor, cur_times: torch.Tensor,
                          src_elapsed: torch.Tensor, time_scale: torch.Tensor, src_node_ids: torch.Tensor = None,
                          src_is_sender: torch.Tensor = None):
        """
        the enabled readouts, concatenated; a readout whose inputs or index are missing contributes zeros
        :param candidate_matches: Tensor, shape (batch_size, num_candidates, max_seq_length), whether that history
        entry is an interaction with that candidate
        :param raw_src_neighb_seq: Tensor, shape (batch_size, max_seq_length), raw ids of the source's history
        :param valid_mask: Tensor, shape (batch_size, max_seq_length), whether the history entry is not padding
        :param raw_test_dst: Tensor, shape (batch_size, num_candidates), raw ids of the candidates
        :param cur_times: Tensor, shape (batch_size, ), prediction times
        :param src_elapsed: Tensor, shape (batch_size, max_seq_length), time since each history entry
        :param time_scale: Tensor, shape (1, ), the running time scale
        :param src_node_ids: Tensor, shape (batch_size, ), raw ids of the sources, needed by the memory readout
        :param src_is_sender: Tensor, shape (batch_size, max_seq_length), needed by the reciprocity readout
        :return: Tensor, shape (batch_size, num_candidates, num_readout_features)
        """
        batch_size, num_candidates = raw_test_dst.shape
        device = src_elapsed.device
        readout_features = []
        if self.use_reciprocity:
            if src_is_sender is None:
                readout_features.append(torch.zeros(batch_size, num_candidates, 6, device=device))
            else:
                readout_features.append(self.reciprocity_features(candidate_matches, src_is_sender,
                                                                  src_elapsed / time_scale))
        if self.use_history_memory:
            if src_node_ids is None or self.memory_num_ids is None:
                readout_features.append(torch.zeros(batch_size, num_candidates, 2 * len(self.window_scales) + 4,
                                                    device=device))
            else:
                readout_features.append(self.memory_features(src_node_ids, raw_test_dst, cur_times, time_scale))
        if self.use_covisit:
            if self._covisit_index() is None:
                readout_features.append(torch.zeros(batch_size, num_candidates, 2 * self.num_decay_kernels + 5,
                                                    device=device))
            else:
                readout_features.append(self.covisit_features(raw_src_neighb_seq * valid_mask, raw_test_dst,
                                                              cur_times, src_elapsed / time_scale))
        return torch.cat(readout_features, dim=-1)
