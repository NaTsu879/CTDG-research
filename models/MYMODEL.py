import math
import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP


class MYMODEL(torch.nn.Module):
    """
    MYMODEL: direction-aware and structure-aware candidate ranking, built on CRAFT's backbone.

    The gap this closes
    -------------------
    Every neighbor sampler in this repository symmetrizes the temporal graph: for a node s it returns
    the set of nodes s interacted with, with no record of who initiated each interaction (see
    get_neighbor_sampler in utils/utils.py, which appends each edge to the adjacency list of both
    endpoints). TGAT, TGN, CAWN, GraphMixer, DyGFormer, SGNN-HN and CRAFT therefore all consume an
    undirected history. But the link prediction task here is directed - rank the destinations that
    the source will interact with next - and on a communication network such as uci the direction of
    a past interaction carries most of the signal: "d messaged me five minutes ago and I have not
    replied yet" predicts s -> d far better than "d and I interacted five minutes ago".

    Nothing extra has to be sampled to recover this. The neighbor sampler already returns the edge id
    of every history entry, which the CRAFT code path discards; the id of the sender of that edge is
    in the data the pipeline already holds, so a single boolean per history entry says whether the
    source was the sender or the receiver.

    The three additions to CRAFT
    ----------------------------
    1. Direction-aware history (the main mechanism). A learned direction encoding is added to the
       history entries before the candidate attends over them, so the attention can separate
       "the people I contacted" from "the people who contacted me". Two vectors of parameters, and
       CRAFT cannot express the distinction at any width, because the distinction is absent from
       its input.

    2. Reciprocity and overlap readout. Per candidate, the elapsed time and count of interactions in
       each direction with that candidate (was it me who wrote last, or them?), plus the co-occurrence
       between the source's history and the candidate's own recent history - the vectorized
       counterpart of DyGFormer's neighbor co-occurrence encoding, whose per-row np.unique loop is a
       large part of why that model is too heavy to run on the bigger datasets. Overlap is also
       aggregated with K learned exponential kernels, since a shared partner matters more when the
       interaction was recent. The readout is an additive correction to the logit behind a gate that
       starts at zero, so the backbone keeps CRAFT's exact width, initialization and gradient path.

    3. Optional reverse view. The source attends over the candidate's own recent history through the
       same cross-attention weights (no new parameters), giving the model both endpoints' viewpoints
       the way DyGFormer has them, and it is added to the logit behind its own zero-initialized gate.
       Only meaningful when both endpoints live in one id space, i.e. on non-bipartite datasets.

    Why it cannot regress
    ---------------------
    All additions are zero-initialized: the direction encoding is zero, the structural attention bias
    is zero, and both logit corrections are switched off by their gates. With --no_structural_bias
    --no_structural_features --no_reciprocity --no_direction --no_inner_product the model loads
    CRAFT's state_dict and reproduces CRAFT's logits exactly, which is how the control run is done.

    Training is untouched: one negative per positive from the same collision-checked sampler, the
    same BPR/BCE loss, validation on average precision and MRR at test only, as in the CRAFT paper.
    """

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=True,
                 input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=True,
                 num_output_layer=1, emb_dropout_prob=0.1, skip_connection=False, num_dst_neighbors=20,
                 num_decay_kernels=4, use_direction=True, use_reciprocity=True, use_structural_bias=True,
                 use_structural_features=True, use_reverse_view=False, use_inner_product=True,
                 shares_node_space=False, time_scale_momentum=0.99):
        """
        the arguments up to skip_connection are CRAFT's, with the same meaning
        :param num_dst_neighbors: int, number of recent neighbors of each candidate used for co-occurrence
        :param num_decay_kernels: int, number of exponential kernels of the recency-weighted overlap
        :param use_direction: bool, whether the direction encoding is added to the history entries
        :param use_reciprocity: bool, whether the per-direction elapsed times and counts feed the readout
        :param use_structural_bias: bool, whether the co-occurrence counts bias the attention logits
        :param use_structural_features: bool, whether the overlap statistics feed the readout
        :param use_reverse_view: bool, whether the source also attends over the candidate's own history,
        requires shares_node_space
        :param use_inner_product: bool, whether the scaled inner product term is added to the logit
        :param shares_node_space: bool, whether sources and destinations are indexed in one id space
        (true for non-bipartite datasets, where dst_min_idx == src_min_idx == 1)
        """
        super(MYMODEL, self).__init__()
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
        self.num_dst_neighbors = num_dst_neighbors
        self.num_decay_kernels = num_decay_kernels
        self.use_direction = use_direction
        self.use_reciprocity = use_reciprocity
        self.use_structural_bias = use_structural_bias
        self.use_structural_features = use_structural_features
        self.use_reverse_view = use_reverse_view and shares_node_space
        self.use_inner_product = use_inner_product
        self.shares_node_space = shares_node_space
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

        # ---- additions ----
        # index 0: the source received this interaction, index 1: the source initiated it
        if self.use_direction:
            self.direction_embedding = nn.Embedding(2, self.hidden_size)
        if self.use_structural_bias:
            self.structural_bias = nn.Linear(2, self.n_heads)
        # the readout features: reciprocity first, then overlap
        num_readout_features = 0
        if self.use_reciprocity:
            # elapsed time and count of interactions with the candidate, per direction, plus two flags
            num_readout_features += 6
        if self.use_structural_features:
            # overlap count, recency-weighted overlap per kernel, normalized overlap, candidate history
            # length, and whether the candidate has any history
            num_readout_features += self.num_decay_kernels + 4
            self.log_tau = nn.Parameter(torch.linspace(math.log(0.05), math.log(5.0), self.num_decay_kernels))
        self.num_readout_features = num_readout_features
        if num_readout_features > 0:
            self.readout_head = MLP(num_layers=2, input_dim=num_readout_features, hidden_dim=self.hidden_size,
                                    output_dim=1, dropout=self.hidden_dropout_prob, use_act=True,
                                    skip_connection=False)
            self.readout_gate = nn.Parameter(torch.zeros(1))
        if self.use_reverse_view:
            self.reverse_head = nn.Linear(self.hidden_size, 1)
            self.reverse_gate = nn.Parameter(torch.zeros(1))
        if self.use_inner_product:
            self.dot_weight = nn.Parameter(torch.zeros(1))

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_time_intervals = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_repeat_times = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout(self.emb_dropout_prob)

        # running scale of the elapsed times, updated during training only, so the kernels and the log
        # features are comparable across datasets whose timestamps differ by orders of magnitude
        self.register_buffer('time_scale', torch.ones(1))
        self.register_buffer('time_scale_initialized', torch.zeros(1))

        self.loss_type = loss_type
        if self.loss_type == 'BCE':
            self.loss_fct = nn.BCELoss()
        elif self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)
        self._init_at_craft()

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

    def _init_at_craft(self):
        """
        zero every added term, so that the model starts exactly at CRAFT: the direction encoding and the
        attention bias vanish, and the two logit corrections are switched off by their gates
        """
        with torch.no_grad():
            if self.use_direction:
                self.direction_embedding.weight.zero_()
            if self.use_structural_bias:
                self.structural_bias.weight.zero_()
                self.structural_bias.bias.zero_()

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

    def count_co_occurrences(self, src_neighb_seq: torch.Tensor, dst_neighb_seq: torch.Tensor):
        """
        vectorized neighbor co-occurrence, the counterpart of DyGFormer's per-row np.unique loop
        :param src_neighb_seq: Tensor, shape (batch_size, max_seq_length), raw ids of the source's history
        :param dst_neighb_seq: Tensor, shape (batch_size, num_candidates, num_dst_neighbors), raw ids of the
        candidates' own histories
        :return: shared_counts (batch_size, num_candidates, max_seq_length), own_counts (batch_size,
        max_seq_length), dst_seq_len (batch_size, num_candidates)
        """
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates, num_dst_neighbors = dst_neighb_seq.shape[1], dst_neighb_seq.shape[2]
        src_valid = src_neighb_seq != 0
        dst_valid = dst_neighb_seq != 0

        matches = (src_neighb_seq.view(batch_size, 1, max_seq_length, 1) ==
                   dst_neighb_seq.view(batch_size, num_candidates, 1, num_dst_neighbors))
        matches = matches & src_valid.view(batch_size, 1, max_seq_length, 1) & \
            dst_valid.view(batch_size, num_candidates, 1, num_dst_neighbors)
        shared_counts = matches.sum(dim=-1).float()

        own_matches = (src_neighb_seq.unsqueeze(2) == src_neighb_seq.unsqueeze(1)) & src_valid.unsqueeze(1)
        own_counts = own_matches.sum(dim=-1).float() * src_valid.float()
        return shared_counts, own_counts, dst_valid.sum(dim=-1).float()

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

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst,
                dst_last_update_times, dst_neighb_seq, src_is_sender=None, src_node_ids=None):
        """
        score every candidate, node ids are raw ids and every tensor is on self.device
        :param src_neighb_seq: Tensor, shape (batch_size, max_seq_length), the source's recent history
        :param src_neighb_seq_len: Tensor, shape (batch_size, )
        :param neighbors_interact_times: Tensor, shape (batch_size, max_seq_length)
        :param cur_times: Tensor, shape (batch_size, ), prediction times
        :param test_dst: Tensor, shape (batch_size, num_candidates), column 0 is the positive candidate
        :param dst_last_update_times: Tensor, shape (batch_size, num_candidates), -100000 when unknown
        :param dst_neighb_seq: Tensor, shape (batch_size, num_candidates, num_dst_neighbors)
        :param src_is_sender: Tensor, shape (batch_size, max_seq_length), whether the source initiated the
        interaction of that history entry
        :param src_node_ids: Tensor, shape (batch_size, ), needed by the reverse view
        :return: Tensor, shape (batch_size, num_candidates)
        """
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates = test_dst.shape[1]
        needs_time_scale = self.use_reciprocity or self.use_structural_features

        # ---- structure of the pair, computed on the raw ids before they are reindexed ----
        if self.use_structural_bias or self.use_structural_features:
            shared_counts, own_counts, dst_seq_len = self.count_co_occurrences(src_neighb_seq, dst_neighb_seq)

        # ---- reindex into the candidate id space, as CRAFT does ----
        raw_src_neighb_seq = src_neighb_seq
        src_neighb_seq = (src_neighb_seq - self.dst_min_idx + 1).masked_fill(src_neighb_seq - self.dst_min_idx + 1 < 0, 0)
        test_dst = test_dst - self.dst_min_idx + 1
        valid_mask = src_neighb_seq != 0

        src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
        time_scale = self._get_time_scale(src_elapsed, valid_mask) if needs_time_scale else None

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

        # ---- CRAFT: history representation, with the direction of each entry ----
        neighb_emb = self.node_embedding(src_neighb_seq)
        if self.use_pos:
            position_ids = torch.arange(max_seq_length, dtype=torch.long, device=src_neighb_seq.device)
            neighb_emb = neighb_emb + self.position_embedding(position_ids.unsqueeze(0).expand_as(src_neighb_seq))
        if self.use_direction and src_is_sender is not None:
            neighb_emb = neighb_emb + self.direction_embedding(src_is_sender.long())
        input_emb = self.emb_dropout(self.LayerNorm(neighb_emb))
        if self.input_cat_time_intervals:
            src_time_intervals = (cur_times.view(-1, 1) - neighbors_interact_times).masked_fill(~valid_mask, -100000)
            src_neighb_time_embedding = self.time_projection(src_time_intervals.float().view(-1, 1)).view(
                batch_size, max_seq_length, -1)
            src_neighb_time_embedding = self.dropout(self.LayerNorm_time_intervals(src_neighb_time_embedding))
            input_emb = torch.cat([input_emb, src_neighb_time_embedding], dim=-1)

        # ---- candidate-conditioned attention, optionally biased by the co-occurrence counts ----
        attention_mask = self.get_attention_mask(
            torch.ones(batch_size, num_candidates, device=src_neighb_seq.device), mask_b=valid_mask)
        if self.use_structural_bias:
            bias_inputs = torch.stack([
                torch.log1p(shared_counts),
                torch.log1p(own_counts).unsqueeze(1).expand(batch_size, num_candidates, max_seq_length),
            ], dim=-1)
            attention_mask = attention_mask + self.structural_bias(bias_inputs).permute(0, 3, 1, 2)
        context = self.cross_attention(test_dst_emb, attention_mask, input_emb, output_all_encoded_layers=False)[-1]

        output = context
        if self.output_cat_time_intervals:
            output = torch.cat([output, dst_node_time_intervals_feat], dim=-1).float()
        if self.output_cat_repeat_times:
            output = torch.cat([output, repeat_times_feat], dim=-1).float()
        logits = self.output_layer(output.view(-1, output.shape[-1])).view(batch_size, num_candidates)

        # ---- additive correction from the reciprocity and overlap readout ----
        readout_features = []
        if self.use_reciprocity:
            if src_is_sender is None:
                readout_features.append(torch.zeros(batch_size, num_candidates, 6, device=logits.device))
            else:
                readout_features.append(self.reciprocity_features(candidate_matches, src_is_sender,
                                                                  src_elapsed / time_scale))
        if self.use_structural_features:
            is_shared = (shared_counts > 0).float()
            tau = self.log_tau.exp().clamp(min=1e-3)
            decay = torch.exp(-(src_elapsed / time_scale).unsqueeze(-1) / tau.view(1, 1, -1))
            decay = decay * valid_mask.unsqueeze(-1).float()
            decayed_overlap = torch.einsum('bcl,blk->bck', is_shared, decay)
            overlap = is_shared.sum(dim=-1)
            src_seq_len = valid_mask.sum(dim=1).float().view(-1, 1)
            readout_features.append(torch.cat([
                torch.log1p(overlap).unsqueeze(-1),
                torch.log1p(decayed_overlap),
                (overlap / (src_seq_len + dst_seq_len - overlap + self.eps)).unsqueeze(-1),
                torch.log1p(dst_seq_len).unsqueeze(-1),
                (dst_seq_len > 0).float().unsqueeze(-1),
            ], dim=-1))
        if len(readout_features) > 0:
            logits = logits + self.readout_gate * self.readout_head(torch.cat(readout_features, dim=-1)).squeeze(dim=-1)

        # ---- optional reverse view: the source attends over each candidate's own history ----
        if self.use_reverse_view and src_node_ids is not None:
            num_dst_neighbors = dst_neighb_seq.shape[-1]
            dst_hist = (dst_neighb_seq - self.dst_min_idx + 1)
            dst_hist = dst_hist.masked_fill(dst_neighb_seq == 0, 0).masked_fill(dst_hist < 0, 0)
            dst_valid = dst_hist != 0
            dst_hist_emb = self.emb_dropout(self.LayerNorm(self.node_embedding(dst_hist))).view(
                batch_size * num_candidates, num_dst_neighbors, -1)
            src_query = self.emb_dropout(self.LayerNorm(self.node_embedding(
                (src_node_ids - self.dst_min_idx + 1).clamp(min=0)))).view(batch_size, 1, 1, -1)
            src_query = src_query.expand(batch_size, num_candidates, 1, self.hidden_size).reshape(
                batch_size * num_candidates, 1, self.hidden_size)
            reverse_mask = torch.where(dst_valid.view(batch_size * num_candidates, 1, 1, num_dst_neighbors),
                                       0.0, -10000.0)
            reverse_context = self.cross_attention(src_query, reverse_mask, dst_hist_emb,
                                                   output_all_encoded_layers=False)[-1]
            reverse_context = reverse_context.view(batch_size, num_candidates, -1)[..., :self.hidden_size]
            # a candidate with no history at all contributes nothing
            reverse_context = reverse_context * dst_valid.any(dim=-1, keepdim=True).float()
            logits = logits + self.reverse_gate * self.reverse_head(reverse_context).squeeze(dim=-1)

        # SASRec / SGNN-HN style scaled inner product between the candidate and its context
        if self.use_inner_product:
            logits = logits + self.dot_weight * (test_dst_emb * context[..., :self.hidden_size]).sum(dim=-1) / \
                math.sqrt(self.hidden_size)
        return logits

    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = torch.bmm(mask_a.unsqueeze(1).transpose(1, 2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def compute_scores(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, dst_neighb_seq, src_is_sender=None, src_node_ids=None):
        """
        move a batch to the device and score the candidates
        :param dst_neighb_seq: Tensor, shape (batch_size * num_candidates, num_dst_neighbors), as returned by
        utils.utils.get_dst_neighbors
        :param src_is_sender: Tensor, shape (batch_size, max_seq_length), whether the source initiated each
        history entry, recovered from the edge ids by the caller
        """
        test_dst = test_dst.to(self.device)
        dst_neighb_seq = dst_neighb_seq.to(self.device).view(test_dst.shape[0], test_dst.shape[1], -1)
        return self.forward(src_neighb_seq=src_neighb_seq.to(self.device),
                            src_neighb_seq_len=src_neighb_seq_len.to(self.device),
                            neighbors_interact_times=src_neighb_interact_times.to(self.device),
                            cur_times=cur_pred_times.to(self.device),
                            test_dst=test_dst,
                            dst_last_update_times=dst_last_update_times.to(self.device),
                            dst_neighb_seq=dst_neighb_seq,
                            src_is_sender=src_is_sender.to(self.device) if src_is_sender is not None else None,
                            src_node_ids=src_node_ids.to(self.device) if src_node_ids is not None else None)

    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                dst_last_update_times, dst_neighb_seq, src_is_sender=None, src_node_ids=None):
        """
        [0]src_neighb_seq: [B, L]
        [1]src_neighb_seq_len: [B]
        [2]src_neighb_interact_times: [B, L]
        [3]cur_pred_times: [B]
        [4]test_dst: [B, 1+num_negs], the positive candidate is at column 0
        [5]dst_last_update_times: [B, 1+num_negs]
        [6]dst_neighb_seq: [B * (1+num_negs), num_dst_neighbors]
        [7]src_is_sender: [B, L]
        [8]src_node_ids: [B]
        """
        logits = self.compute_scores(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times,
                                     test_dst, dst_last_update_times, dst_neighb_seq, src_is_sender, src_node_ids)
        if self.loss_type == 'BPR':
            return logits[:, 0].flatten(), logits[:, 1:].flatten()
        return logits[:, 0].sigmoid().flatten(), logits[:, 1:].sigmoid().flatten()

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, dst_neighb_seq, src_is_sender=None, src_node_ids=None):
        """
        the objective CRAFT is trained with: one negative per positive, BPR or BCE
        """
        positive_probabilities, negative_probabilities = self.predict(
            src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
            dst_last_update_times, dst_neighb_seq, src_is_sender, src_node_ids)
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
