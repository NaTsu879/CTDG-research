import math
import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP


class CRAFTV8(torch.nn.Module):
    """
    CRAFTV8: CRAFT with a collaborative closure readout.

    The gap this closes
    -------------------
    On a bipartite graph (users -> items) a candidate edge s -> d can never close a 2-cycle (there is no
    reverse edge d -> s) or a triangle (s and d have no neighbor in common), so reciprocity and
    common-neighbor signals are identically zero. The shortest cycle a bipartite graph allows has length
    4: s - i - u - d, where i is an item s interacted with and u is another user who interacted with both
    i and d. Closing such cycles is temporal item-based collaborative filtering: "people who engaged with
    what I engaged with went on to engage with d". It is built from other users' interactions, so it
    carries signal even when s has never interacted with d, which is the regime of the unseen-dominant
    datasets (GoogleLocal and ML-20M have no repeated edges at all).

    The collaborative closure readout
    ---------------------------------
    A co-transition i -> d is a user interacting with d within `window` interactions after interacting
    with i. All co-transitions of the dataset are indexed once, sorted by (pair, time) (see
    utils.utils.build_transition_index), so the number of co-transitions i -> d that happened before
    the prediction time t, and the time of the latest one, are two binary searches on the device. For
    every item i_l in the source's history (the history CRAFT already samples) and every candidate d:
        n_l = number of co-transitions i_l -> d before t
    which gives the features
        log(1 + sum_l n_l)                                   how many 4-cycles s -> d would close
        log(1 + sum_l 1[n_l > 0] exp(-dt_l / tau_k))         the same, weighted by how recently s engaged
                                                             with i_l, one per learned tau_k
        1[any n_l > 0]                                       whether any closes
        log(1 + time since the latest co-transition into d)  how fresh the collaborative evidence is
    which go through a small MLP whose output is added to CRAFT's logit behind a gate that starts at
    zero, so the model starts at CRAFT. With --no_closure the model is CRAFT, which is the control run.
    No extra neighbors are sampled, so training costs about as much as CRAFT.

    Training is untouched: one negative per positive from the same collision-checked sampler, the
    same BPR/BCE loss, validation on average precision and MRR at test only, as in the CRAFT paper.
    """

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=True,
                 input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=True,
                 num_output_layer=1, emb_dropout_prob=0.1, skip_connection=False, use_closure=True,
                 num_decay_kernels=3, time_scale_momentum=0.99):
        """
        the arguments up to skip_connection are CRAFT's, with the same meaning
        :param use_closure: bool, whether the collaborative closure readout is added to the logit
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
        self.use_closure = use_closure
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

        # ---- collaborative closure readout ----
        if self.use_closure:
            self.log_tau = nn.Parameter(torch.linspace(math.log(0.1), math.log(10.0), self.num_decay_kernels))
            self.readout_head = MLP(num_layers=2, input_dim=self.num_decay_kernels + 3, hidden_dim=self.hidden_size,
                                    output_dim=1, dropout=self.hidden_dropout_prob, use_act=True,
                                    skip_connection=False)
            self.readout_gate = nn.Parameter(torch.zeros(1))
            # running scale of the elapsed times, updated during training only, so the kernels are comparable
            # across datasets whose timestamps differ by orders of magnitude
            self.register_buffer('time_scale', torch.ones(1))
            self.register_buffer('time_scale_initialized', torch.zeros(1))
            # the co-transition indexes, built from the data by set_transition_index: one from the training
            # edges, used in training mode, and one from all edges, used in evaluation mode, mirroring the
            # train and full neighbor samplers. They are not part of the checkpoint, since they are rebuilt
            # from the data on every run
            for split in ('train', 'full'):
                for name in ('codes', 'times', 'pairs', 'unique_times'):
                    self.register_buffer(f'{split}_transition_{name}', None, persistent=False)
            self.transition_num_ids = None

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

    def set_transition_index(self, train_transition_index: dict, full_transition_index: dict):
        """
        :param train_transition_index: dict, as returned by utils.utils.build_transition_index on the training
        edges, used in training mode so that training never sees a validation or test edge
        :param full_transition_index: dict, the same on all edges, used in evaluation mode
        """
        # both indexes encode pairs with the same multiplier, the largest node id of the full data plus one
        self.transition_num_ids = full_transition_index['num_ids']
        for split, transition_index in (('train', train_transition_index), ('full', full_transition_index)):
            # with no co-transition at all no 4-cycle can close, and the readout stays at zero
            if len(transition_index['pairs']) > 0:
                for name in ('codes', 'times', 'pairs', 'unique_times'):
                    setattr(self, f'{split}_transition_{name}', torch.from_numpy(transition_index[name]))

    def _transition_index(self):
        """
        the index of the current mode, as (codes, times, pairs, unique_times), or None when it is empty
        """
        split = 'train' if self.training else 'full'
        codes = getattr(self, f'{split}_transition_codes')
        if codes is None:
            return None
        return (codes, getattr(self, f'{split}_transition_times'), getattr(self, f'{split}_transition_pairs'),
                getattr(self, f'{split}_transition_unique_times'))

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

    def count_transitions(self, items: torch.Tensor, candidates: torch.Tensor, cur_times: torch.Tensor):
        """
        number of co-transitions items -> candidates before cur_times, and the time of the latest one
        :param items: Tensor, shape (batch_size, max_seq_length), raw ids, 0 is padding
        :param candidates: Tensor, shape (batch_size, num_candidates), raw ids
        :param cur_times: Tensor, shape (batch_size, )
        :return: counts and latest times, both with shape (batch_size, num_candidates, max_seq_length)
        """
        transition_index = self._transition_index()
        if transition_index is None:
            shape = (items.shape[0], candidates.shape[1], items.shape[1])
            return torch.zeros(shape, device=items.device), torch.zeros(shape, dtype=torch.float64, device=items.device)
        codes, times, pairs, unique_times = transition_index
        num_ranks = unique_times.numel() + 1
        pair_codes = items.unsqueeze(1).long() * self.transition_num_ids + candidates.unsqueeze(2).long()
        pair_idx = torch.searchsorted(pairs, pair_codes).clamp(max=pairs.numel() - 1)
        pair_found = (pairs[pair_idx] == pair_codes) & (items.unsqueeze(1) != 0)
        # rank of the prediction time among the transition times: how many distinct times are strictly earlier
        time_rank = torch.searchsorted(unique_times, cur_times.double().contiguous()).view(-1, 1, 1)
        lo = torch.searchsorted(codes, pair_idx * num_ranks)
        hi = torch.searchsorted(codes, pair_idx * num_ranks + time_rank)
        counts = (hi - lo) * pair_found.long()
        latest_times = times[(hi - 1).clamp(min=0)]
        return counts.float(), latest_times

    def closure_features(self, src_neighb_seq: torch.Tensor, test_dst: torch.Tensor, cur_times: torch.Tensor,
                         normalized_item_elapsed: torch.Tensor, time_scale: torch.Tensor):
        """
        4-cycle closure statistics between the source's history items and each candidate, on raw node ids
        :param src_neighb_seq: Tensor, shape (batch_size, max_seq_length), raw ids, 0 is padding
        :param test_dst: Tensor, shape (batch_size, num_candidates), raw ids
        :param cur_times: Tensor, shape (batch_size, )
        :param normalized_item_elapsed: Tensor, shape (batch_size, max_seq_length), time since the source
        engaged with each history item, divided by the time scale
        :param time_scale: Tensor, shape (1, )
        :return: Tensor, shape (batch_size, num_candidates, num_decay_kernels + 3)
        """
        counts, latest_times = self.count_transitions(src_neighb_seq, test_dst, cur_times)
        closed = (counts > 0).float()
        item_valid = (src_neighb_seq != 0).float()

        tau = self.log_tau.exp().clamp(min=1e-3)
        decay = torch.exp(-normalized_item_elapsed.unsqueeze(-1) / tau.view(1, 1, -1)) * item_valid.unsqueeze(-1)
        decayed_closures = torch.einsum('bcl,blk->bck', closed, decay)

        any_closed = closed.amax(dim=-1)
        # the latest co-transition into the candidate, over all of the source's items
        latest_elapsed = (cur_times.double().view(-1, 1, 1) - latest_times).float().clamp(min=0.0) / time_scale
        freshest = torch.where(counts > 0, latest_elapsed, torch.full_like(latest_elapsed, 1e9)).amin(dim=-1) * any_closed
        return torch.cat([
            torch.log1p(counts.sum(dim=-1)).unsqueeze(-1),
            torch.log1p(decayed_closures),
            any_closed.unsqueeze(-1),
            torch.log1p(freshest).unsqueeze(-1),
        ], dim=-1)

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

        # ---- CRAFT: how many times the source interacted with each candidate ----
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

        # ---- additive correction from the collaborative closure readout ----
        if self.use_closure:
            src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
            time_scale = self._get_time_scale(src_elapsed, valid_mask)
            if self._transition_index() is None:
                readout_features = torch.zeros(batch_size, num_candidates, self.num_decay_kernels + 3, device=logits.device)
            else:
                readout_features = self.closure_features(raw_src_neighb_seq * valid_mask, raw_test_dst, cur_times,
                                                         src_elapsed / time_scale, time_scale)
            logits = logits + self.readout_gate * self.readout_head(readout_features).squeeze(dim=-1)
        return logits

    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = torch.bmm(mask_a.unsqueeze(1).transpose(1, 2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

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
