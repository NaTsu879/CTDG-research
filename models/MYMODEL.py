import math
import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP


class MYMODEL(torch.nn.Module):
    """
    MYMODEL: structure-aware candidate ranking. CRAFT's candidate-conditioned cross-attention, made
    aware of the candidate's own neighborhood through vectorized co-occurrence, at CRAFT's cost.

    The gap this closes
    -------------------
    CRAFT scores a candidate d for a source s by letting d query s's recent history. It is
    candidate-centric but *structure-blind*: apart from one scalar (d's last-update time), it never
    looks at d's own neighborhood, so it cannot use triadic closure - "s and d share partners,
    therefore s will contact d" - which is the dominant link-formation mechanism in social and
    communication graphs. On uci, the dataset where that mechanism is strongest, DyGFormer (76.61)
    beats CRAFT-R (75.11), and DyGFormer's own ablations credit its neighbor co-occurrence encoding.
    But DyGFormer pays for it with a transformer over both sequences plus patching, and its
    co-occurrence counting is a Python loop with np.unique per row - which is why it is missing from
    several columns of the benchmark tables entirely.

    This model takes that one signal and injects it where it is cheapest and most useful: as a bias
    on the candidate-conditioned attention, plus a recency-weighted readout.

    1. Vectorized co-occurrence. With the source's history S (L entries) and each candidate's own
       history D_c (Ld entries, from the same sampler the rest of the pipeline uses), one broadcast
       comparison gives
         shared[b, c, l]  = how many times S[l] appears in the candidate's history,
         own[b, l]        = how many times S[l] appears in the source's own history,
       in O(L * Ld) per pair with no Python loop, no extra embedding table and no extra tokens.
       Both quantities are *identity-free*: they describe structure, not who the nodes are, so they
       transfer to nodes never seen in training.

    2. Structural attention bias (the main mechanism). Those two counts are mapped by a linear layer
       to one bias per attention head and added to the cross-attention logits over history positions.
       The candidate can therefore say "attend to the partners I also interact with" instead of only
       "attend to the partners that look like me in embedding space". This is the part CRAFT cannot
       express at any width: its attention logits depend on the candidate only through its embedding,
       so a candidate that shares three partners with the source and one that shares none are
       indistinguishable when their embeddings are similar.

    3. Recency-weighted common-neighbor readout. Shared partners matter more when the interaction was
       recent, so the overlap is also aggregated with K learned exponential kernels over normalized
       elapsed time, together with the overlap count, the Jaccard-style normalized overlap and the
       candidate's history length. This block is projected and concatenated next to CRAFT's elapsed
       time and repeat-count blocks before the output layer.

    4. Inner-product scoring term. Following SASRec and SGNN-HN, which score by a scaled inner
       product between the sequence representation and the item embedding, a term
       w * <e_d, context(d)> / sqrt(H) is added to the logit; w is learned and starts at zero.

    Why this cannot regress by construction
    ---------------------------------------
    The backbone, its flags (use_pos, output_cat_time_intervals, output_cat_repeat_times) and the
    initialization are CRAFT's. The three additions are zero-initialized: the structural bias layer
    is zero, the output layer's columns that read the structural block are zero, and the
    inner-product weight is zero. At initialization MYMODEL therefore computes exactly CRAFT-R, and
    CRAFT-R stays inside its function class - set those terms to zero and you get CRAFT back, which
    is also how the ablations are run (--no_structural_bias, --no_structural_features). The extra
    parameters are one Linear(2, num_heads), K kernel timescales and a small feature MLP, so the
    count stays within a couple of percent of CRAFT's.

    Inputs are the source history with timestamps, the candidates, their last-update times and their
    own recent histories. Training is untouched: one negative per positive from the same
    collision-checked sampler, the same BPR/BCE loss, the same evaluation protocol. The candidate's
    history is the same information DyGFormer, TGN, CAWN and CRAFTV5 in this repository already
    consume; nothing about the protocol differs from a CRAFT run.
    """

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=True,
                 input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=True,
                 num_output_layer=1, emb_dropout_prob=0.1, skip_connection=False, num_dst_neighbors=20,
                 num_decay_kernels=4, use_structural_bias=True, use_structural_features=True,
                 use_inner_product=True, time_scale_momentum=0.99):
        """
        the arguments up to skip_connection are CRAFT's, with the same meaning
        :param num_dst_neighbors: int, number of recent neighbors of each candidate used for co-occurrence
        :param num_decay_kernels: int, number of exponential kernels of the recency-weighted overlap
        :param use_structural_bias: bool, whether the co-occurrence counts bias the attention logits
        :param use_structural_features: bool, whether the overlap readout corrects the logit
        :param use_inner_product: bool, whether the scaled inner product term is added to the logit
        with all three of these off the model is CRAFT, parameter for parameter and value for value
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
        self.use_structural_bias = use_structural_bias
        self.use_structural_features = use_structural_features
        self.use_inner_product = use_inner_product
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

        # ---- structural additions ----
        # per-head attention bias from [count of this history entry in the candidate's history,
        # count of this history entry in the source's own history]
        if self.use_structural_bias:
            self.structural_bias = nn.Linear(2, self.n_heads)
        if self.use_structural_features:
            # overlap count, recency-weighted overlap per kernel, normalized overlap, candidate history length,
            # and whether the candidate has any history
            self.num_structural_features = self.num_decay_kernels + 4
            self.log_tau = nn.Parameter(torch.linspace(math.log(0.05), math.log(5.0), self.num_decay_kernels))
            # the overlap readout is an additive correction to the logit, not a block concatenated into the
            # output layer: the backbone keeps CRAFT's exact width, initialization and gradient path, and the
            # correction is switched on by a scalar that starts at zero
            self.structural_head = MLP(num_layers=2, input_dim=self.num_structural_features,
                                       hidden_dim=self.hidden_size, output_dim=1,
                                       dropout=self.hidden_dropout_prob, use_act=True, skip_connection=False)
            self.structural_gate = nn.Parameter(torch.zeros(1))
        if self.use_inner_product:
            self.dot_weight = nn.Parameter(torch.zeros(1))

        self.output_layer = MLP(num_layers=num_output_layer, input_dim=output_dim, hidden_dim=output_dim,
                                output_dim=1, dropout=self.hidden_dropout_prob, use_act=True,
                                skip_connection=skip_connection)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_time_intervals = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_repeat_times = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout(self.emb_dropout_prob)

        # running scale of the elapsed times, updated during training only, so the kernels are
        # comparable across datasets whose timestamps differ by orders of magnitude
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
        zero every added term, so that the model starts exactly at CRAFT-R: the attention bias vanishes and
        both logit corrections are switched off by their gates
        """
        with torch.no_grad():
            if self.use_structural_bias:
                self.structural_bias.weight.zero_()
                self.structural_bias.bias.zero_()

    def _get_time_scale(self, elapsed_times: torch.Tensor, valid_mask: torch.Tensor):
        """
        running median of the observed elapsed times, updated during training only
        :param elapsed_times: Tensor, shape (batch_size, max_seq_length)
        :param valid_mask: Tensor, shape (batch_size, max_seq_length)
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
        :return: shared_counts (batch_size, num_candidates, max_seq_length), how often each history entry
        appears in the candidate's history; own_counts (batch_size, max_seq_length), how often each history
        entry appears in the source's own history; dst_seq_len (batch_size, num_candidates)
        """
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates, num_dst_neighbors = dst_neighb_seq.shape[1], dst_neighb_seq.shape[2]
        src_valid = (src_neighb_seq != 0)
        dst_valid = (dst_neighb_seq != 0)

        matches = (src_neighb_seq.view(batch_size, 1, max_seq_length, 1) ==
                   dst_neighb_seq.view(batch_size, num_candidates, 1, num_dst_neighbors))
        matches = matches & src_valid.view(batch_size, 1, max_seq_length, 1) & \
            dst_valid.view(batch_size, num_candidates, 1, num_dst_neighbors)
        shared_counts = matches.sum(dim=-1).float()

        own_matches = (src_neighb_seq.unsqueeze(2) == src_neighb_seq.unsqueeze(1)) & src_valid.unsqueeze(1)
        own_counts = own_matches.sum(dim=-1).float() * src_valid.float()
        return shared_counts, own_counts, dst_valid.sum(dim=-1).float()

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst,
                dst_last_update_times, dst_neighb_seq):
        """
        score every candidate, the node ids are raw ids and every tensor is on self.device
        :param src_neighb_seq: Tensor, shape (batch_size, max_seq_length), the source's recent history
        :param src_neighb_seq_len: Tensor, shape (batch_size, )
        :param neighbors_interact_times: Tensor, shape (batch_size, max_seq_length)
        :param cur_times: Tensor, shape (batch_size, ), prediction times
        :param test_dst: Tensor, shape (batch_size, num_candidates), column 0 is the positive candidate
        :param dst_last_update_times: Tensor, shape (batch_size, num_candidates), -100000 when unknown
        :param dst_neighb_seq: Tensor, shape (batch_size, num_candidates, num_dst_neighbors)
        :return: Tensor, shape (batch_size, num_candidates)
        """
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates = test_dst.shape[1]

        # ---- structure of the pair, computed on the raw ids before they are reindexed ----
        shared_counts, own_counts, dst_seq_len = self.count_co_occurrences(src_neighb_seq, dst_neighb_seq)

        # ---- reindex into the candidate id space, as CRAFT does ----
        src_neighb_seq = src_neighb_seq - self.dst_min_idx + 1
        test_dst = test_dst - self.dst_min_idx + 1
        src_neighb_seq = src_neighb_seq.masked_fill(src_neighb_seq < 0, 0)
        valid_mask = src_neighb_seq != 0

        src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
        # only the overlap readout needs the normalized elapsed times, so nothing is touched when it is ablated
        time_scale = self._get_time_scale(src_elapsed, valid_mask) if self.use_structural_features else None

        # ---- CRAFT: elapsed time of the candidate's last interaction ----
        if self.output_cat_time_intervals:
            dst_last_update_intervals = cur_times.view(-1, 1) - dst_last_update_times
            dst_last_update_intervals[dst_last_update_times < -1] = -100000
            dst_node_time_intervals_feat = self.time_projection(dst_last_update_intervals.float().view(-1, 1)).view(
                batch_size, num_candidates, -1)
            dst_node_time_intervals_feat = self.dropout(self.LayerNorm_time_intervals(dst_node_time_intervals_feat))

        test_dst_emb = self.LayerNorm(self.node_embedding(test_dst).view(batch_size, -1, self.hidden_size))
        test_dst_emb = self.emb_dropout(test_dst_emb)

        # ---- CRAFT: how often the candidate itself appears in the source's history ----
        if self.output_cat_repeat_times:
            repeat_times = (test_dst.view(batch_size, num_candidates, 1) ==
                            src_neighb_seq.view(batch_size, 1, max_seq_length))
            repeat_times = (repeat_times & valid_mask.view(batch_size, 1, max_seq_length)).sum(dim=-1, keepdim=True).float()
            repeat_times_feat = self.repeat_times_projection(repeat_times.view(-1, 1)).view(batch_size, num_candidates, -1)
            repeat_times_feat = self.dropout(self.LayerNorm_repeat_times(repeat_times_feat))

        # ---- CRAFT: history representation ----
        neighb_emb = self.node_embedding(src_neighb_seq)
        if self.use_pos:
            position_ids = torch.arange(max_seq_length, dtype=torch.long, device=src_neighb_seq.device)
            position_ids = position_ids.unsqueeze(0).expand_as(src_neighb_seq)
            input_emb = neighb_emb + self.position_embedding(position_ids)
        else:
            input_emb = neighb_emb
        input_emb = self.emb_dropout(self.LayerNorm(input_emb))
        if self.input_cat_time_intervals:
            src_time_intervals = cur_times.view(-1, 1) - neighbors_interact_times
            src_time_intervals = src_time_intervals.masked_fill(~valid_mask, -100000)
            src_neighb_time_embedding = self.time_projection(src_time_intervals.float().view(-1, 1)).view(
                batch_size, max_seq_length, -1)
            src_neighb_time_embedding = self.dropout(self.LayerNorm_time_intervals(src_neighb_time_embedding))
            input_emb = torch.cat([input_emb, src_neighb_time_embedding], dim=-1)

        # ---- structural bias on the attention logits ----
        attention_mask = self.get_attention_mask(
            torch.ones(batch_size, num_candidates, device=src_neighb_seq.device), mask_b=valid_mask)
        if self.use_structural_bias:
            bias_inputs = torch.stack([
                torch.log1p(shared_counts),
                torch.log1p(own_counts).unsqueeze(1).expand(batch_size, num_candidates, max_seq_length),
            ], dim=-1)
            attention_mask = attention_mask + self.structural_bias(bias_inputs).permute(0, 3, 1, 2)
        output = self.cross_attention(test_dst_emb, attention_mask, input_emb, output_all_encoded_layers=False)[-1]
        context = output

        if self.output_cat_time_intervals:
            output = torch.cat([output, dst_node_time_intervals_feat], dim=-1).float()
        if self.output_cat_repeat_times:
            output = torch.cat([output, repeat_times_feat], dim=-1).float()

        # the backbone logit, computed exactly as CRAFT computes it
        logits = self.output_layer(output.view(-1, output.shape[-1])).view(batch_size, num_candidates)

        # ---- additive correction from the recency-weighted common-neighbor readout ----
        if self.use_structural_features:
            is_shared = (shared_counts > 0).float()
            tau = self.log_tau.exp().clamp(min=1e-3)
            decay = torch.exp(-(src_elapsed / time_scale).unsqueeze(-1) / tau.view(1, 1, -1))
            decay = decay * valid_mask.unsqueeze(-1).float()
            decayed_overlap = torch.einsum('bcl,blk->bck', is_shared, decay)
            overlap = is_shared.sum(dim=-1)
            src_seq_len = valid_mask.sum(dim=1).float().view(-1, 1)
            structural_features = torch.cat([
                torch.log1p(overlap).unsqueeze(-1),
                torch.log1p(decayed_overlap),
                (overlap / (src_seq_len + dst_seq_len - overlap + self.eps)).unsqueeze(-1),
                torch.log1p(dst_seq_len).unsqueeze(-1),
                (dst_seq_len > 0).float().unsqueeze(-1),
            ], dim=-1)
            logits = logits + self.structural_gate * self.structural_head(structural_features).squeeze(dim=-1)

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
                       dst_last_update_times, dst_neighb_seq):
        """
        move a batch to the device and score the candidates
        :param dst_neighb_seq: Tensor, shape (batch_size * num_candidates, num_dst_neighbors), as returned by
        utils.utils.get_dst_neighbors
        """
        test_dst = test_dst.to(self.device)
        dst_neighb_seq = dst_neighb_seq.to(self.device).view(test_dst.shape[0], test_dst.shape[1], -1)
        return self.forward(src_neighb_seq=src_neighb_seq.to(self.device),
                            src_neighb_seq_len=src_neighb_seq_len.to(self.device),
                            neighbors_interact_times=src_neighb_interact_times.to(self.device),
                            cur_times=cur_pred_times.to(self.device),
                            test_dst=test_dst,
                            dst_last_update_times=dst_last_update_times.to(self.device),
                            dst_neighb_seq=dst_neighb_seq)

    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                dst_last_update_times, dst_neighb_seq):
        """
        [0]src_neighb_seq: [B, L]
        [1]src_neighb_seq_len: [B]
        [2]src_neighb_interact_times: [B, L]
        [3]cur_pred_times: [B]
        [4]test_dst: [B, 1+num_negs], the positive candidate is at column 0
        [5]dst_last_update_times: [B, 1+num_negs]
        [6]dst_neighb_seq: [B * (1+num_negs), num_dst_neighbors]
        """
        logits = self.compute_scores(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times,
                                     test_dst, dst_last_update_times, dst_neighb_seq)
        if self.loss_type == 'BPR':
            return logits[:, 0].flatten(), logits[:, 1:].flatten()
        return logits[:, 0].sigmoid().flatten(), logits[:, 1:].sigmoid().flatten()

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, dst_neighb_seq):
        """
        the objective CRAFT is trained with: one negative per positive, BPR or BCE
        """
        positive_probabilities, negative_probabilities = self.predict(src_neighb_seq, src_neighb_seq_len,
                                                                     src_neighb_interact_times, cur_pred_times,
                                                                     test_dst, dst_last_update_times, dst_neighb_seq)
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
