import math
import torch
from torch import nn
from models.modules import BPRLoss, MLP
from models.evidence_readouts import EvidenceReadouts


class CRAFTV9(EvidenceReadouts, torch.nn.Module):
    """
    CRAFTV9: a light target-aware matcher with explicit evidence, without CRAFT's transformer.

    What is kept, and why
    ---------------------
    Each candidate still attends over the source's recent history with its own softmax. A model that replaced this
    per-candidate attention with a few history summaries shared by all candidates lost about 15 MRR points on uci,
    so this is the part of CRAFT that does the work.

    What is removed
    ---------------
    CRAFT runs the candidates through stacked transformer layers: query, key, value and output projections,
    residual blocks and a feed-forward network of width 4F, at every layer. CRAFTV9 keeps one attention pass:
        q_j = LN(e(d_j)) W_q,   k_l = LN(e(n_l) + p(l)),
        a_jl = softmax_l( <q_j, k_l> / sqrt(F/h) - beta_h * log(1 + dt_l / tau) ),   c_j = sum_l a_jl k_l,
    per head h, with a single query projection W_q (initialized to the identity) and no key, value or output
    projection, no feed-forward block and no stacked layers. The learned per-head slope beta_h lets attention favor
    recent history entries directly from their elapsed times dt_l.

    What is added
    -------------
    The three readouts of CRAFTV8 (reciprocity, long-range memory, co-visit closure; see
    models.evidence_readouts.EvidenceReadouts), read by binary searches on time-sorted indexes. In CRAFTV8 they
    corrected CRAFT's logit from outside; here they are an input of the scorer, projected to F dimensions, so
    the scorer can weigh the learned match against the explicit evidence for each candidate.

    Scoring
    -------
        s_j = MLP([c_j * q_j ; c_j ; time(d_j) ; repeat(d_j) ; evidence(d_j)])
    where time(d_j) and repeat(d_j) are CRAFT's projections of the elapsed time since the candidate's last
    interaction and of its repeat count in the window, and the MLP has hidden width F (CRAFT's output MLP is as
    wide as its input).

    Cost per batch: B q k F for the attention, B q F^2 for the query projection and the scorer, and B q k log P for
    the readouts, against CRAFT's (B q F^2 + B k F^2 + B q k F) per layer, with a 4F-wide feed-forward network.
    Training is CRAFT's: one negative per positive from the same collision-checked sampler, the same BPR/BCE
    loss and per-dataset hyperparameters (num_layers is not used, since there are no stacked layers).
    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, initializer_range,
                 n_nodes, max_seq_length, device, loss_type, use_pos=True, output_cat_time_intervals=True,
                 output_cat_repeat_times=True, num_output_layer=2, emb_dropout_prob=0.1, use_reciprocity=True,
                 use_history_memory=True, use_covisit=True, window_scales=(0.1, 1.0, 10.0, 100.0), num_decay_kernels=3,
                 time_scale_momentum=0.99):
        """
        :param n_heads: int, number of attention heads; hidden_size must be divisible by it
        :param hidden_size: int, F, the embedding size
        :param hidden_dropout_prob: float, dropout of the projections and of the scorer
        :param attn_dropout_prob: float, dropout of the attention weights
        :param layer_norm_eps: float
        :param initializer_range: float, std of the initial embeddings and linear weights
        :param n_nodes: int, number of candidate (destination-side) nodes
        :param max_seq_length: int, k, the number of history entries
        :param device: str
        :param loss_type: str, 'BPR' or 'BCE'
        :param use_pos: bool, whether a learnable position embedding is added to the history entries
        :param output_cat_time_intervals: bool, whether the elapsed time since the candidate's last interaction is used
        :param output_cat_repeat_times: bool, whether the candidate's repeat count in the window is used
        :param num_output_layer: int, number of layers of the scorer MLP
        :param emb_dropout_prob: float, dropout of the embeddings
        :param use_reciprocity: bool, whether the reciprocity readout feeds the scorer
        :param use_history_memory: bool, whether the long-range memory readout feeds the scorer
        :param use_covisit: bool, whether the co-visit closure readout feeds the scorer
        :param window_scales: tuple of float, the time windows of the memory as multiples of the running time scale
        :param num_decay_kernels: int, number of learned time scales of the recency-weighted closure count
        :param time_scale_momentum: float, momentum of the running scale of the elapsed times
        """
        super(CRAFTV9, self).__init__()
        assert hidden_size % n_heads == 0, f'hidden_size {hidden_size} must be divisible by n_heads {n_heads}'
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // n_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.n_nodes = n_nodes
        self.max_seq_length = max_seq_length
        self.use_pos = use_pos
        self.output_cat_time_intervals = output_cat_time_intervals
        self.output_cat_repeat_times = output_cat_repeat_times
        self.eps = 1e-6
        self.device = device

        # ---- light target-aware matching ----
        self.node_embedding = nn.Embedding(self.n_nodes + 1, self.hidden_size, padding_idx=0)
        if self.use_pos:
            self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.query_projection = nn.Linear(self.hidden_size, self.hidden_size)
        # per-head slope of the recency bias on the attention logits, starting from no bias
        self.recency_slope = nn.Parameter(torch.zeros(self.n_heads))
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        scorer_dim = 2 * self.hidden_size

        # ---- CRAFT's candidate features ----
        if self.output_cat_time_intervals:
            self.time_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size,
                                       output_dim=self.hidden_size, dropout=self.hidden_dropout_prob,
                                       use_act=True, skip_connection=False)
            self.LayerNorm_time_intervals = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            scorer_dim += self.hidden_size
        if self.output_cat_repeat_times:
            self.repeat_times_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size,
                                               output_dim=self.hidden_size, dropout=self.hidden_dropout_prob,
                                               use_act=True, skip_connection=False)
            self.LayerNorm_repeat_times = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            scorer_dim += self.hidden_size

        # ---- explicit evidence, projected to F dimensions as an input of the scorer ----
        self._init_readouts(use_reciprocity=use_reciprocity, use_history_memory=use_history_memory,
                            use_covisit=use_covisit, window_scales=window_scales,
                            num_decay_kernels=num_decay_kernels, time_scale_momentum=time_scale_momentum)
        if self.num_readout_features > 0:
            self.evidence_projection = nn.Linear(self.num_readout_features, self.hidden_size)
            self.LayerNorm_evidence = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            scorer_dim += self.hidden_size
        # the recency bias of the attention needs the time scale whether or not a readout is on
        self._init_time_scale()

        self.output_layer = MLP(num_layers=num_output_layer, input_dim=scorer_dim, hidden_dim=self.hidden_size,
                                output_dim=1, dropout=self.hidden_dropout_prob, use_act=True, skip_connection=False)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout(emb_dropout_prob)

        self.loss_type = loss_type
        if self.loss_type == 'BCE':
            self.loss_fct = nn.BCELoss()
        elif self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)
        # start attention from plain embedding similarity: the query projection is the identity
        with torch.no_grad():
            self.query_projection.weight.copy_(torch.eye(self.hidden_size))

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
        src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
        time_scale = self._get_time_scale(src_elapsed, valid_mask)

        # ---- light target-aware matching: one attention pass, per candidate and per head ----
        query = self.emb_dropout(self.LayerNorm(self.node_embedding(test_dst)))
        history = self.node_embedding(src_neighb_seq)
        if self.use_pos:
            position_ids = torch.arange(max_seq_length, dtype=torch.long, device=src_neighb_seq.device)
            history = history + self.position_embedding(position_ids).unsqueeze(0)
        history = self.emb_dropout(self.LayerNorm(history))

        heads_query = self.query_projection(query).view(batch_size, num_candidates, self.n_heads, self.head_size)
        heads_history = history.view(batch_size, max_seq_length, self.n_heads, self.head_size)
        scores = torch.einsum('bchd,blhd->bhcl', heads_query, heads_history) / math.sqrt(self.head_size)
        recency = torch.log1p(src_elapsed / time_scale).view(batch_size, 1, 1, max_seq_length)
        scores = scores - self.recency_slope.view(1, -1, 1, 1) * recency
        scores = scores.masked_fill(~valid_mask.view(batch_size, 1, 1, max_seq_length), -10000.0)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.einsum('bhcl,blhd->bchd', weights, heads_history).reshape(batch_size, num_candidates, -1)
        # a source with an empty history has nothing to attend to
        context = context * valid_mask.any(dim=1).view(batch_size, 1, 1).float()
        scorer_inputs = [context * query, context]

        # ---- CRAFT's candidate features ----
        if self.output_cat_time_intervals:
            # candidates with no past interaction get CRAFT's sentinel
            dst_last_update_intervals = torch.where(dst_last_update_times < -1,
                                                    torch.full_like(dst_last_update_times, -100000),
                                                    cur_times.view(-1, 1) - dst_last_update_times)
            time_feat = self.time_projection(dst_last_update_intervals.float().view(-1, 1)).view(
                batch_size, num_candidates, -1)
            scorer_inputs.append(self.dropout(self.LayerNorm_time_intervals(time_feat)))
        candidate_matches = (test_dst.view(batch_size, num_candidates, 1) ==
                             src_neighb_seq.view(batch_size, 1, max_seq_length)) & valid_mask.view(batch_size, 1, max_seq_length)
        if self.output_cat_repeat_times:
            repeat_times = candidate_matches.sum(dim=-1, keepdim=True).float()
            repeat_feat = self.repeat_times_projection(repeat_times.view(-1, 1)).view(batch_size, num_candidates, -1)
            scorer_inputs.append(self.dropout(self.LayerNorm_repeat_times(repeat_feat)))

        # ---- explicit evidence ----
        if self.num_readout_features > 0:
            evidence = self.evidence_features(candidate_matches, raw_src_neighb_seq, valid_mask, raw_test_dst,
                                              cur_times, src_elapsed, time_scale, src_node_ids, src_is_sender)
            scorer_inputs.append(self.dropout(self.LayerNorm_evidence(self.evidence_projection(evidence))))

        scorer_input = torch.cat(scorer_inputs, dim=-1)
        return self.output_layer(scorer_input.view(-1, scorer_input.shape[-1])).view(batch_size, num_candidates)

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
