import math
import numpy as np
import torch
from torch import nn
from models.CRAFTV9 import CRAFTV9

from models.evidence_readouts import count_events_before
from utils.utils import build_event_index


class CRAFTV11(CRAFTV9):
    """
    CRAFTV11: CRAFTV9 with a stream-refreshed candidate identity.

    Problem (frozen identity). Transductive CTDG models represent a candidate d by an ID embedding e(d) and bias b(d),
    learned on the training split and frozen afterwards. Seen edges are recognised from the source's history window,
    which updates with every event; unseen edges depend on the candidate's identity, which does not. On unseen-dominant
    data many test destinations do most of their linking after the training cutoff, so their identity describes who they
    used to be (on Flickr, positives with more than 60% of their links after training lose 10 to 13 MRR points at every
    degree level).

    Solution. The candidate's identity at time t is its learned (static) part plus a part recomputed from the stream:
    the time-decayed sum of the embeddings of its m most recent neighbours before t,

        r(d, t) = (1 / m) sum_j exp(-(t - t_j) / tau) LN(e(u_j))
        v_d(t)  = LN(e(d)) + W_r r(d, t)                 (matching identity, used against the source's history)
        b_d(t)  = b(d)     + w_b . r(d, t)               (attractiveness)

    A node that rose after training gets a current identity from the nodes that recently linked to it, whose own
    embeddings are not stale; a node that did not change keeps its learned embedding. The sum (not a mean) keeps how
    much recent activity there is. W_r and w_b start at zero, so training starts from CRAFTV9 exactly.

    Ablations (identity_refresh): 'sum' (default), 'mean' (weights normalised to sum to one), 'refresh_only' (no static
    embedding or bias for the candidate), 'none' (CRAFTV9). Training is CRAFT's: same negatives, loss and
    hyperparameters; the only extra input is each candidate's m most recent neighbours, read from the same neighbour
    sampler CRAFT already uses for the candidate's last update time.

    Staleness simulation (staleness_simulation=True). A learned identity is never stale in training: the embedding of
    every training node has been fitted to all of its training links, later ones included. So a model that combines it
    with the refreshed part learns to trust it fully, and keeps trusting it for nodes that rose after training (on
    Flickr the refreshed identity alone ranks those best, but the combination does not use it). The learned identity
    (embedding and bias) is therefore scaled by the share of the candidate's links that existed at a cutoff,

        kappa(d, t) = links of d before t_cut / links of d before t      (1 for a node with no links yet)

    evaluation: t_cut is the real training cutoff, so kappa is how much of the candidate's current activity its learned
                identity was fitted on;
    training:   the cutoff is simulated: it is placed delta events before t, delta ~ Uniform(0, H) with H the number of
                events after the real training cutoff (the evaluation horizon), so kappa follows the same law as in
                evaluation. The horizon is counted in events rather than time because activity is not uniform in time
                (on uci the evaluation period lasts three times longer than the training period).
    Simulating the cutoff, rather than drawing kappa at random, matters: at test time the most-grown candidates are
    the likely positives, and a random kappa would teach the scorer that a weak identity means a worse candidate.
    """

    REFRESH_MODES = ('sum', 'mean', 'refresh_only', 'none')

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, initializer_range,
                 n_nodes, max_seq_length, device, loss_type, use_pos=True, output_cat_time_intervals=True,
                 output_cat_repeat_times=True, num_output_layer=2, emb_dropout_prob=0.1, num_decay_kernels=3,
                 time_scale_momentum=0.99, identity_refresh='sum', num_refresh_neighbors=10, n_src_nodes=None,
                 staleness_simulation=False):
        """
        :param identity_refresh: str, one of 'sum', 'mean', 'refresh_only', 'none'
        :param staleness_simulation: bool, whether the learned identity is scaled by its staleness (see above)
        :param num_refresh_neighbors: int, m, the number of recent neighbours of each candidate
        :param n_src_nodes: int or None, number of source-side nodes when the graph is bipartite (the neighbours of a
            candidate are then sources, which get their own embedding table); None when sources and candidates share
            one id space
        (the other parameters are CRAFTV9's)
        """
        super(CRAFTV11, self).__init__(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
                                       initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=use_pos,
                                       output_cat_time_intervals=output_cat_time_intervals,
                                       output_cat_repeat_times=output_cat_repeat_times,
                                       num_output_layer=num_output_layer, emb_dropout_prob=emb_dropout_prob,
                                       num_decay_kernels=num_decay_kernels, time_scale_momentum=time_scale_momentum)
        assert identity_refresh in self.REFRESH_MODES, f'identity_refresh must be one of {self.REFRESH_MODES}'
        self.identity_refresh = identity_refresh
        self.bipartite = n_src_nodes is not None
        # read by the training and evaluation loops: how many recent neighbours of each candidate to fetch
        self.num_refresh_neighbors = 1 if identity_refresh == 'none' else num_refresh_neighbors
        if identity_refresh != 'none':
            if self.bipartite:
                self.linker_embedding = nn.Embedding(n_src_nodes + 1, hidden_size, padding_idx=0)
                self.linker_embedding.weight.data.normal_(mean=0.0, std=initializer_range)
                self.linker_LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.refresh_log_tau = nn.Parameter(torch.zeros(1))
            self.refresh_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.refresh_bias = nn.Linear(hidden_size, 1, bias=False)
            with torch.no_grad():
                # start from CRAFTV9: the refreshed part contributes nothing until it is learned
                self.refresh_projection.weight.zero_()
                self.refresh_bias.weight.zero_()
        self.staleness_simulation = staleness_simulation
        self.staleness_index = None
        if staleness_simulation:
            assert identity_refresh in ('sum', 'mean'), 'staleness simulation needs both a learned and a refreshed identity'

    def set_staleness_index(self, train_data, full_data):
        """
        index when every node's links happened, to count those before any time, and record the training cutoff and the
        evaluation horizon
        :param train_data: Data, the training edges
        :param full_data: Data, all edges
        """
        index = build_event_index(keys=np.concatenate([full_data.src_node_ids, full_data.dst_node_ids]),
                                  times=np.concatenate([full_data.node_interact_times, full_data.node_interact_times]))
        self.staleness_index = {key: torch.from_numpy(value).to(self.device) for key, value in index.items()}
        train_end = float(np.max(train_data.node_interact_times))
        # counting events strictly before the next representable time counts every training event
        self.training_cutoff = float(np.nextafter(train_end, np.inf))
        # the time of every event in order, and how many events come after the training cutoff
        event_times = np.sort(full_data.node_interact_times.astype(np.float64))
        self.event_times = torch.from_numpy(event_times).to(self.device)
        self.staleness_horizon = int(len(event_times) - np.searchsorted(event_times, self.training_cutoff))

    def staleness(self, raw_test_dst, cur_times):
        """
        kappa per candidate: links before the cutoff over links before t; the cutoff is simulated in training
        :param raw_test_dst: Tensor, shape (batch_size, num_candidates), raw node ids
        :param cur_times: Tensor, shape (batch_size, )
        :return: Tensor, shape (batch_size, num_candidates)
        """
        assert self.staleness_index is not None, 'call set_staleness_index first'
        nodes = raw_test_dst.long()
        query_times = cur_times.double().view(-1, 1).expand_as(nodes)
        if self.training:
            # place the simulated cutoff delta events before the prediction time
            position = torch.searchsorted(self.event_times, query_times.contiguous())
            delta = (torch.rand(nodes.shape, device=nodes.device) * (self.staleness_horizon + 1)).long()
            cutoff_times = self.event_times[(position - delta).clamp(min=0)]
        else:
            cutoff_times = torch.full_like(query_times, self.training_cutoff).minimum(query_times)
        links_now, _ = count_events_before(self.staleness_index, nodes, query_times)
        links_at_cutoff, _ = count_events_before(self.staleness_index, nodes, cutoff_times)
        kappa = links_at_cutoff.float() / links_now.clamp(min=1).float()
        return torch.where(links_now > 0, kappa, torch.ones_like(kappa))

    def refreshed_identity(self, dst_neighb_seq, dst_neighb_interact_times, cur_times, time_scale, batch_size,
                           num_candidates):
        """
        time-decayed sum of the embeddings of each candidate's recent neighbours
        :param dst_neighb_seq: Tensor, shape (batch_size * num_candidates, m), raw ids, 0 is padding
        :param dst_neighb_interact_times: Tensor, shape (batch_size * num_candidates, m)
        :param cur_times: Tensor, shape (batch_size, )
        :param time_scale: Tensor, shape (1, )
        :return: Tensor, shape (batch_size, num_candidates, F)
        """
        valid = dst_neighb_seq != 0
        if self.bipartite:
            ids = (dst_neighb_seq - self.src_min_idx + 1).clamp(min=0) * valid.long()
            linkers = self.linker_LayerNorm(self.linker_embedding(ids))
        else:
            ids = (dst_neighb_seq - self.dst_min_idx + 1).clamp(min=0) * valid.long()
            linkers = self.LayerNorm(self.node_embedding(ids))
        linkers = self.emb_dropout(linkers)
        query_times = cur_times.float().repeat_interleave(num_candidates).view(-1, 1)
        elapsed = (query_times - dst_neighb_interact_times.float()).clamp(min=0.0) / time_scale
        tau = self.refresh_log_tau.exp().clamp(min=1e-3)
        weights = torch.exp(-elapsed / tau) * valid.float()
        if self.identity_refresh == 'mean':
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        else:
            weights = weights / dst_neighb_seq.shape[1]
        refreshed = torch.einsum('nm,nmf->nf', weights, linkers)
        return refreshed.view(batch_size, num_candidates, -1)

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst,
                dst_last_update_times, dst_neighb_seq=None, dst_neighb_interact_times=None):
        """
        score every candidate, node ids are raw ids and every tensor is on self.device
        :param dst_neighb_seq: Tensor, shape (batch_size * num_candidates, m), each candidate's recent neighbours
        :param dst_neighb_interact_times: Tensor, shape (batch_size * num_candidates, m)
        (the other parameters are CRAFTV9's)
        :return: Tensor, shape (batch_size, num_candidates)
        """
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates = test_dst.shape[1]
        kappa = self.staleness(test_dst, cur_times) if self.staleness_simulation else None

        # ---- reindex into the candidate id space, as CRAFT does ----
        src_neighb_seq = (src_neighb_seq - self.dst_min_idx + 1).masked_fill(src_neighb_seq - self.dst_min_idx + 1 < 0, 0)
        test_dst = test_dst - self.dst_min_idx + 1
        valid_mask = src_neighb_seq != 0
        src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0) * valid_mask.float()
        time_scale = self._get_time_scale(src_elapsed, valid_mask)

        # ---- candidate identity: static (learned) part plus the part refreshed from the stream ----
        static_emb = self.LayerNorm(self.node_embedding(test_dst))
        static_bias = self.node_bias(test_dst).squeeze(-1)
        if kappa is not None:
            # a stale learned identity counts for less
            static_emb, static_bias = static_emb * kappa.unsqueeze(-1), static_bias * kappa
        if self.identity_refresh == 'none':
            candidate_emb, candidate_bias = static_emb, static_bias
        else:
            refreshed = self.refreshed_identity(dst_neighb_seq, dst_neighb_interact_times, cur_times, time_scale,
                                                batch_size, num_candidates)
            refreshed_emb = self.refresh_projection(refreshed)
            refreshed_bias = self.refresh_bias(refreshed).squeeze(-1)
            if self.identity_refresh == 'refresh_only':
                candidate_emb, candidate_bias = refreshed_emb, refreshed_bias
            else:
                candidate_emb, candidate_bias = static_emb + refreshed_emb, static_bias + refreshed_bias
        candidate_emb = self.emb_dropout(candidate_emb)

        # ---- light pairwise affinity (CRAFTV9) ----
        history_emb = self.history_projection(self.emb_dropout(self.LayerNorm(self.node_embedding(src_neighb_seq))))
        features = [self.latent_affinity(history_emb, candidate_emb, valid_mask, src_elapsed / time_scale)]

        # ---- CRAFT's candidate features, as log-scaled scalars (CRAFTV9) ----
        if self.output_cat_time_intervals:
            never = (dst_last_update_times < -1).float()
            since_last = ((cur_times.view(-1, 1) - dst_last_update_times).float().clamp(min=0.0) / time_scale) * (1.0 - never)
            features += [torch.log1p(since_last).unsqueeze(-1), never.unsqueeze(-1)]
        if self.output_cat_repeat_times:
            candidate_matches = (test_dst.view(batch_size, num_candidates, 1) ==
                                 src_neighb_seq.view(batch_size, 1, max_seq_length)) & valid_mask.view(batch_size, 1, max_seq_length)
            features.append(torch.log1p(candidate_matches.sum(dim=-1, keepdim=True).float()))

        scorer_input = torch.cat(features, dim=-1)
        logits = self.output_layer(scorer_input.view(-1, scorer_input.shape[-1])).view(batch_size, num_candidates)
        return logits + candidate_bias

    def compute_scores(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, dst_neighb_seq=None, dst_neighb_interact_times=None):
        """
        move a batch to the device and score the candidates
        """
        extra = {}
        if dst_neighb_seq is not None:
            extra = dict(dst_neighb_seq=dst_neighb_seq.to(self.device),
                         dst_neighb_interact_times=dst_neighb_interact_times.to(self.device))
        return self.forward(src_neighb_seq=src_neighb_seq.to(self.device),
                            src_neighb_seq_len=src_neighb_seq_len.to(self.device),
                            neighbors_interact_times=src_neighb_interact_times.to(self.device),
                            cur_times=cur_pred_times.to(self.device),
                            test_dst=test_dst.to(self.device),
                            dst_last_update_times=dst_last_update_times.to(self.device), **extra)

    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                dst_last_update_times, dst_neighb_seq=None, dst_neighb_interact_times=None):
        """
        the positive candidate is at column 0 of test_dst
        """
        logits = self.compute_scores(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times,
                                     test_dst, dst_last_update_times, dst_neighb_seq, dst_neighb_interact_times)
        if self.loss_type == 'BPR':
            return logits[:, 0].flatten(), logits[:, 1:].flatten()
        return logits[:, 0].sigmoid().flatten(), logits[:, 1:].sigmoid().flatten()

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, dst_neighb_seq=None, dst_neighb_interact_times=None):
        """
        the objective CRAFT is trained with: one negative per positive, BPR or BCE
        """
        positive_probabilities, negative_probabilities = self.predict(
            src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
            dst_last_update_times, dst_neighb_seq, dst_neighb_interact_times)
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
