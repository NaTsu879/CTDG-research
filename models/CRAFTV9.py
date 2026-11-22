import math
import torch
from torch import nn
from models.modules import BPRLoss, MLP


class CRAFTV9(torch.nn.Module):
    """
    CRAFTV9: a light, attention-free link predictor.

    Each of the source's k most recent history items excites the candidate d with a learned pairwise affinity
        a(i, d) = softplus(<u_i, v_d> / sqrt(F) + c),   u_i = LN(e(i)) W_u,   v_d = LN(e(d)),
    summed over the history with learned time-decay kernels exp(-dt_i / tau_k), plus the strongest single affinity.
    The softplus is applied to every (history item, candidate) pair before the sum, so the score cannot collapse
    into one source vector dotted with the candidate. This is close to the Hawkes-process neighborhood formation of
    HTNE (Zuo et al., KDD 2018).

    CRAFT's candidate features are kept as log-scaled scalars rather than LayerNorm(Linear(x)): the log time since the
    candidate's last interaction, whether it never interacted, and (on seen-dominant datasets) its log repeat count
    in the window. A two-layer MLP of hidden width F scores the features, and a learned bias per candidate is added:

        s_j = MLP([affinity features, candidate time features, repeat count]) + b_{d_j}

    There is no cross-attention, no softmax over the history, no query, key, value or output projection, no
    feed-forward block, no stacked layers and no position embedding; time enters through the decay kernels. Cost per
    batch: B q k F for the affinities and B k F^2 for the history projection. Training is CRAFT's: one negative per
    positive from the same collision-checked sampler, the same BPR/BCE loss and per-dataset hyperparameters
    (num_layers, num_heads and use_pos are not used).
    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, initializer_range,
                 n_nodes, max_seq_length, device, loss_type, use_pos=True, output_cat_time_intervals=True,
                 output_cat_repeat_times=True, num_output_layer=2, emb_dropout_prob=0.1, num_decay_kernels=3,
                 time_scale_momentum=0.99):
        """
        :param n_heads: int, not used (no attention); kept so the constructor matches the other CRAFT variants
        :param hidden_size: int, F, the embedding size
        :param hidden_dropout_prob: float, dropout of the scorer
        :param attn_dropout_prob: float, not used (no attention)
        :param layer_norm_eps: float
        :param initializer_range: float, std of the initial embeddings and linear weights
        :param n_nodes: int, number of candidate (destination-side) nodes
        :param max_seq_length: int, k, the number of history entries
        :param device: str
        :param loss_type: str, 'BPR' or 'BCE'
        :param use_pos: bool, not used (time enters through decay kernels)
        :param output_cat_time_intervals: bool, whether the time since the candidate's last interaction is used
        :param output_cat_repeat_times: bool, whether the candidate's repeat count in the window is used
        :param num_output_layer: int, number of layers of the scorer MLP
        :param emb_dropout_prob: float, dropout of the embeddings
        :param num_decay_kernels: int, K, number of learned time scales of the affinity decay
        :param time_scale_momentum: float, momentum of the running scale of the elapsed times
        """
        super(CRAFTV9, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.n_nodes = n_nodes
        self.max_seq_length = max_seq_length
        self.output_cat_time_intervals = output_cat_time_intervals
        self.output_cat_repeat_times = output_cat_repeat_times
        self.num_latent_kernels = num_decay_kernels
        self.time_scale_momentum = time_scale_momentum
        self.eps = 1e-6
        self.device = device

        # ---- light pairwise affinity ----
        self.node_embedding = nn.Embedding(self.n_nodes + 1, self.hidden_size, padding_idx=0)
        self.node_bias = nn.Embedding(self.n_nodes + 1, 1, padding_idx=0)
        self.history_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.excitation_bias = nn.Parameter(torch.zeros(1))
        self.latent_log_tau = nn.Parameter(torch.linspace(math.log(0.1), math.log(10.0), self.num_latent_kernels))
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.emb_dropout = nn.Dropout(emb_dropout_prob)
        # decayed affinities per kernel, the strongest single affinity, whether there is any history, and how many
        # history entries there are
        scorer_dim = self.num_latent_kernels + 3

        # ---- CRAFT's candidate features, as log-scaled scalars ----
        if self.output_cat_time_intervals:
            # log time since the candidate's last interaction, and whether it never interacted
            scorer_dim += 2
        if self.output_cat_repeat_times:
            scorer_dim += 1

        # running scale of the elapsed times, updated during training only, so the kernels and the log features adapt
        # to the time unit of each dataset
        self.register_buffer('time_scale', torch.ones(1))
        self.register_buffer('time_scale_initialized', torch.zeros(1))

        self.output_layer = MLP(num_layers=num_output_layer, input_dim=scorer_dim, hidden_dim=self.hidden_size,
                                output_dim=1, dropout=self.hidden_dropout_prob, use_act=True, skip_connection=False)

        self.loss_type = loss_type
        if self.loss_type == 'BCE':
            self.loss_fct = nn.BCELoss()
        elif self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)
        with torch.no_grad():
            # start from plain embedding similarity, and from no preference between candidates
            self.history_projection.weight.copy_(torch.eye(self.hidden_size))
            self.node_bias.weight.zero_()

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

    def latent_affinity(self, history_emb: torch.Tensor, candidate_emb: torch.Tensor, valid_mask: torch.Tensor,
                        normalized_elapsed: torch.Tensor):
        """
        learned affinity of each candidate to the source's history items, decayed with time
        :param history_emb: Tensor, shape (batch_size, max_seq_length, F)
        :param candidate_emb: Tensor, shape (batch_size, num_candidates, F)
        :param valid_mask: Tensor, shape (batch_size, max_seq_length)
        :param normalized_elapsed: Tensor, shape (batch_size, max_seq_length), elapsed time / time scale
        :return: Tensor, shape (batch_size, num_candidates, num_latent_kernels + 3)
        """
        # (batch_size, num_candidates, max_seq_length): one affinity per (candidate, history item) pair
        similarity = torch.einsum('bcf,blf->bcl', candidate_emb, history_emb) / math.sqrt(self.hidden_size)
        valid = valid_mask.unsqueeze(1)
        excitation = nn.functional.softplus(similarity + self.excitation_bias) * valid.float()
        tau = self.latent_log_tau.exp().clamp(min=1e-3)
        decay = torch.exp(-normalized_elapsed.unsqueeze(-1) / tau.view(1, 1, -1)) * valid_mask.unsqueeze(-1).float()
        decayed = torch.log1p(torch.einsum('bcl,blk->bck', excitation, decay))
        has_history = valid_mask.any(dim=1).view(-1, 1, 1).float()
        strongest = similarity.masked_fill(~valid, -1e4).amax(dim=-1, keepdim=True) * has_history
        history_size = torch.log1p(valid_mask.sum(dim=1).float()).view(-1, 1, 1).expand_as(strongest)
        return torch.cat([decayed, strongest, has_history.expand_as(strongest), history_size], dim=-1)

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

        # ---- reindex into the candidate id space, as CRAFT does ----
        src_neighb_seq = (src_neighb_seq - self.dst_min_idx + 1).masked_fill(src_neighb_seq - self.dst_min_idx + 1 < 0, 0)
        test_dst = test_dst - self.dst_min_idx + 1
        valid_mask = src_neighb_seq != 0
        src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0) * valid_mask.float()
        time_scale = self._get_time_scale(src_elapsed, valid_mask)

        # ---- light pairwise affinity ----
        history_emb = self.history_projection(self.emb_dropout(self.LayerNorm(self.node_embedding(src_neighb_seq))))
        candidate_emb = self.emb_dropout(self.LayerNorm(self.node_embedding(test_dst)))
        features = [self.latent_affinity(history_emb, candidate_emb, valid_mask, src_elapsed / time_scale)]

        # ---- CRAFT's candidate features, as log-scaled scalars ----
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
        return logits + self.node_bias(test_dst).squeeze(-1)

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
