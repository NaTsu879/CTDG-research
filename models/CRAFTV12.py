import numpy as np
import torch
from torch import nn
from models.CRAFTV9 import CRAFTV9



class CRAFTV12(CRAFTV9):
    """
    CRAFTV12: CRAFTV9 with test-time identity fold-in.

    Problem (frozen identity). Transductive CTDG models represent a candidate d by an ID embedding learned on the
    training split and frozen afterwards. Seen edges are recognised from the source's history window, which updates
    with every event; unseen edges depend on the candidate's identity, which does not. On unseen-dominant data many test
    destinations do most of their linking after the training cutoff, so their identity describes who they used to be
    (on Flickr, positives with more than 60% of their links after training lose 10 to 13 MRR points at every degree
    level). A model cannot learn to distrust a stale identity, because no identity is stale in training: every training
    node's embedding has been fitted to all of its training links.

    Solution (fold-in, from matrix factorisation, where a user who arrives after training gets a vector solved from
    their new interactions with the item vectors held fixed). At prediction time, without gradients, a candidate's
    identity is moved toward where the trained model's own scoring geometry says the sources that recently linked to it
    are looking:

        q(u, t_u)   = mean_i P(LN(e(i)))  over the L most recent history items i of u before t_u
                      (the vectors a candidate is matched against, as in CRAFTV9's affinity)
        f(d, t)     = mean of q(u_j, t_j) over the m most recent neighbours u_j of d before t
        kappa(d, t) = links of d before the training cutoff / links of d before t      (1 if d has no links yet)
        v_d(t)      = (1 - rho) LN(e(d)) + rho ||LN(e(d))|| f / ||f||,     rho = 1 - kappa

    rho is the share of the candidate's links that its embedding never saw, so it is not tuned: a node the embedding has
    fully seen keeps its learned identity, a node that rose after training is placed by its new linkers. Training is
    exactly CRAFTV9's (kappa = 1 for every training query, so fold-in is applied in evaluation only), there are no new
    parameters, and a trained CRAFTV9 checkpoint loads as is. Lookups use a sorted event index on the device instead of
    the neighbour sampler, so the extra cost is a few gathers per candidate.
    """

    def __init__(self, *args, foldin_neighbors=10, foldin_history=30, **kwargs):
        """
        :param foldin_neighbors: int, m, recent neighbours of a candidate that place its identity
        :param foldin_history: int, L, history items of each such neighbour
        (the other parameters are CRAFTV9's)
        """
        super(CRAFTV12, self).__init__(*args, **kwargs)
        self.foldin_neighbors = foldin_neighbors
        self.foldin_history = foldin_history
        self.foldin_index = None

    def set_foldin_index(self, train_data, full_data):
        """
        index every node's links by (node, time), with the neighbour and time of each, and record the training cutoff
        :param train_data: Data, the training edges
        :param full_data: Data, all edges (only links strictly before a prediction time are ever read)
        """
        nodes = np.concatenate([full_data.src_node_ids, full_data.dst_node_ids]).astype(np.int64)
        neighbours = np.concatenate([full_data.dst_node_ids, full_data.src_node_ids]).astype(np.int64)
        times = np.concatenate([full_data.node_interact_times, full_data.node_interact_times]).astype(np.float64)
        edge_order = np.concatenate([np.arange(len(full_data.src_node_ids))] * 2)
        unique_keys, key_index = np.unique(nodes, return_inverse=True)
        unique_times, time_rank = np.unique(times, return_inverse=True)
        codes = key_index.astype(np.int64) * (len(unique_times) + 1) + time_rank.astype(np.int64)
        # links of a node in time order, ties in edge order, as the neighbour sampler keeps them
        order = np.lexsort((edge_order, codes))
        to_device = lambda array: torch.from_numpy(array).to(self.device)
        self.foldin_index = {'keys': to_device(unique_keys), 'codes': to_device(codes[order]),
                             'unique_times': to_device(unique_times),
                             'neighbours': to_device(neighbours[order]), 'times': to_device(times[order])}
        self.training_cutoff = float(np.nextafter(float(np.max(train_data.node_interact_times)), np.inf))

    def _range_before(self, nodes, query_times):
        """
        positions [lo, hi) in the index of the links of each node strictly before each time
        :param nodes: Tensor of int64, any shape
        :param query_times: Tensor of float64, same shape
        """
        index = self.foldin_index
        keys, codes, unique_times = index['keys'], index['codes'], index['unique_times']
        num_ranks = unique_times.numel() + 1
        key_idx = torch.searchsorted(keys, nodes.contiguous()).clamp(max=keys.numel() - 1)
        found = keys[key_idx] == nodes
        time_rank = torch.searchsorted(unique_times, query_times.contiguous())
        lo = torch.searchsorted(codes, (key_idx * num_ranks).contiguous())
        hi = torch.searchsorted(codes, (key_idx * num_ranks + time_rank).contiguous())
        hi = torch.where(found, hi, lo)
        return lo, hi

    def _recent(self, lo, hi, k):
        """
        the k most recent indexed links in [lo, hi): positions (..., k) and their validity
        """
        positions = hi.unsqueeze(-1) - 1 - torch.arange(k, device=hi.device)
        valid = positions >= lo.unsqueeze(-1)
        return positions.clamp(min=0), valid

    def folded_identity(self, raw_test_dst, cur_times, static_emb):
        """
        the fold-in identity of each candidate, and how far to move toward it
        :param raw_test_dst: Tensor, shape (batch_size, num_candidates), raw ids
        :param cur_times: Tensor, shape (batch_size, )
        :param static_emb: Tensor, shape (batch_size, num_candidates, F), LN(e(d))
        :return: folded Tensor (batch_size, num_candidates, F) and rho Tensor (batch_size, num_candidates)
        """
        index = self.foldin_index
        nodes = raw_test_dst.long()
        query_times = cur_times.double().view(-1, 1).expand_as(nodes)
        lo, hi = self._range_before(nodes, query_times)
        _, hi_cut = self._range_before(nodes, torch.full_like(query_times, self.training_cutoff).minimum(query_times))
        links_now, links_cut = hi - lo, hi_cut - lo
        kappa = torch.where(links_now > 0, links_cut.float() / links_now.clamp(min=1).float(), torch.ones_like(links_now, dtype=torch.float))
        # the m most recent neighbours of each candidate, and when they linked
        positions, linker_valid = self._recent(lo, hi, self.foldin_neighbors)
        linkers, linker_times = index['neighbours'][positions], index['times'][positions]
        # the L most recent history items of each neighbour before it linked
        lo_u, hi_u = self._range_before(linkers, linker_times)
        item_positions, item_valid = self._recent(lo_u, hi_u, self.foldin_history)
        items = index['neighbours'][item_positions]
        items = (items - self.dst_min_idx + 1).masked_fill(items - self.dst_min_idx + 1 < 0, 0)
        item_valid = item_valid & (items != 0) & linker_valid.unsqueeze(-1)
        projected = self.history_projection(self.LayerNorm(self.node_embedding(items * item_valid.long())))
        weights = item_valid.float()
        query = (projected * weights.unsqueeze(-1)).sum(-2) / weights.sum(-1, keepdim=True).clamp(min=1.0)
        has_query = item_valid.any(-1)
        folded = (query * has_query.unsqueeze(-1).float()).sum(-2) / has_query.sum(-1, keepdim=True).clamp(min=1).float()
        usable = has_query.any(-1)
        folded = folded * (static_emb.norm(dim=-1, keepdim=True) / folded.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        rho = (1.0 - kappa) * usable.float()
        return folded, rho

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst,
                dst_last_update_times):
        """
        CRAFTV9's scorer; in evaluation the candidate identity is folded in (see the class docstring)
        """
        if self.training or self.foldin_index is None:
            return super(CRAFTV12, self).forward(src_neighb_seq, src_neighb_seq_len, neighbors_interact_times,
                                                  cur_times, test_dst, dst_last_update_times)
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates = test_dst.shape[1]
        raw_test_dst = test_dst

        # ---- reindex into the candidate id space, as CRAFT does ----
        src_neighb_seq = (src_neighb_seq - self.dst_min_idx + 1).masked_fill(src_neighb_seq - self.dst_min_idx + 1 < 0, 0)
        test_dst = test_dst - self.dst_min_idx + 1
        valid_mask = src_neighb_seq != 0
        src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0) * valid_mask.float()
        time_scale = self._get_time_scale(src_elapsed, valid_mask)

        # ---- candidate identity, folded in when the embedding has not seen most of the candidate's links ----
        candidate_matches = (test_dst.view(batch_size, num_candidates, 1) ==
                             src_neighb_seq.view(batch_size, 1, max_seq_length)) & valid_mask.view(batch_size, 1, max_seq_length)
        static_emb = self.LayerNorm(self.node_embedding(test_dst))
        folded, rho = self.folded_identity(raw_test_dst, cur_times, static_emb)
        # a candidate already in the source's history is recognised from the history, which is never stale; only
        # unseen candidates depend on their identity
        rho = rho * (~candidate_matches.any(dim=-1)).float()
        candidate_emb = (1.0 - rho).unsqueeze(-1) * static_emb + rho.unsqueeze(-1) * folded

        # ---- CRAFTV9's affinity and candidate features ----
        history_emb = self.history_projection(self.LayerNorm(self.node_embedding(src_neighb_seq)))
        features = [self.latent_affinity(history_emb, candidate_emb, valid_mask, src_elapsed / time_scale)]
        if self.output_cat_time_intervals:
            never = (dst_last_update_times < -1).float()
            since_last = ((cur_times.view(-1, 1) - dst_last_update_times).float().clamp(min=0.0) / time_scale) * (1.0 - never)
            features += [torch.log1p(since_last).unsqueeze(-1), never.unsqueeze(-1)]
        if self.output_cat_repeat_times:
            features.append(torch.log1p(candidate_matches.sum(dim=-1, keepdim=True).float()))
        scorer_input = torch.cat(features, dim=-1)
        logits = self.output_layer(scorer_input.view(-1, scorer_input.shape[-1])).view(batch_size, num_candidates)
        return logits + self.node_bias(test_dst).squeeze(-1)
