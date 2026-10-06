import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP


class CRAFTV7(torch.nn.Module):
    """
    CRAFTV7: CRAFT with CRAFTV4's behavioral-intent gated fusion and MYMODEL's reciprocity readout.

    1. CRAFT cross-attention: each candidate destination attends over the source's recent history,
       giving a candidate-specific context c_d.
    2. Behavioral intent (from CRAFTV4): a source state z_t is pooled from the history by self-attentive
       pooling, independent of the candidate. A candidate-specific gate decides how much of it to mix in:
       g_d = sigma(W_g [c_d ; z_t]),  h_d = c_d + g_d * z_t
       ('projected' fusion first projects c_d and z_t, as in CRAFTV4).
    3. CRAFT head: h_d is concatenated with the candidate's elapsed-time feature (and the repeat-count
       feature on seen-dominant datasets) and scored by an MLP.
    4. Reciprocity (from MYMODEL): the source's past interactions with each candidate are split by
       direction (source sent / source received), and each direction gives the log elapsed time of the
       most recent one, whether any exists, and the log count. The six features go through a small MLP
       whose output is added to the logit behind a gate that starts at zero.

    With --no_reciprocity the model is CRAFTV4, with --no_behavior_gate it is MYMODEL, and with both it is
    CRAFT, which is how the ablation is done. Training follows the CRAFT recipe unchanged.
    """

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, initializer_range, n_nodes, max_seq_length, device, loss_type, use_pos=True,
                 input_cat_time_intervals=False, output_cat_time_intervals=True, output_cat_repeat_times=True,
                 num_output_layer=1, emb_dropout_prob=0.1, skip_connection=False, fusion_mode='projected',
                 use_behavior_gate=True, use_reciprocity=True, time_scale_momentum=0.99):
        """
        the arguments up to skip_connection are CRAFT's, with the same meaning
        :param fusion_mode: str, 'projected' or 'simple', the gated fusion of CRAFTV4
        :param use_behavior_gate: bool, whether the behavioral state is fused into the candidate context
        :param use_reciprocity: bool, whether the reciprocity readout is added to the logit
        :param time_scale_momentum: float, momentum of the running scale of the elapsed times
        """
        super(CRAFTV7, self).__init__()
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
        self.fusion_mode = fusion_mode
        self.use_behavior_gate = use_behavior_gate
        self.use_reciprocity = use_reciprocity
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

        # ---- behavioral intent: source state z_t and the candidate-specific gate, as in CRAFTV4 ----
        if self.use_behavior_gate:
            self.behavior_attn = nn.Sequential(nn.Linear(trm_input_dim, self.hidden_size), nn.Tanh(),
                                               nn.Linear(self.hidden_size, 1))
            self.behavior_proj = nn.Sequential(nn.Linear(trm_input_dim, self.hidden_size),
                                               nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
                                               nn.Dropout(self.hidden_dropout_prob))
            if self.fusion_mode == 'projected':
                self.proj_c = nn.Sequential(nn.Linear(trm_input_dim, self.hidden_size),
                                            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
                                            nn.Dropout(self.hidden_dropout_prob))
                self.proj_z = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
                                            nn.Dropout(self.hidden_dropout_prob))
                self.gate_layer = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid())
                output_dim = self.hidden_size
            else:
                # the simple fusion adds z_t to c_d directly, so the two must have the same width
                assert trm_input_dim == self.hidden_size, "simple fusion requires input_cat_time_intervals=False"
                self.gate_layer = nn.Sequential(nn.Linear(trm_input_dim + self.hidden_size, trm_input_dim),
                                                nn.Sigmoid())

        # ---- CRAFT head ----
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

        # ---- reciprocity readout: elapsed time, existence and count of interactions with the candidate,
        # once for each direction ----
        if self.use_reciprocity:
            self.readout_head = MLP(num_layers=2, input_dim=6, hidden_dim=self.hidden_size, output_dim=1,
                                    dropout=self.hidden_dropout_prob, use_act=True, skip_connection=False)
            self.readout_gate = nn.Parameter(torch.zeros(1))
            # running scale of the elapsed times, updated during training only, so the features are
            # comparable across datasets whose timestamps differ by orders of magnitude
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

    def extract_source_behavioral_state(self, input_emb: torch.Tensor, valid_mask: torch.Tensor):
        """
        self-attentive pooling of the source's history into one behavioral state, as in CRAFTV4
        :param input_emb: Tensor, shape (batch_size, max_seq_length, trm_input_dim)
        :param valid_mask: Tensor, shape (batch_size, max_seq_length)
        :return: Tensor, shape (batch_size, hidden_size)
        """
        attn_logits = self.behavior_attn(input_emb).masked_fill(~valid_mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        # a source with an empty history pools to zero
        attn_weights = attn_weights * valid_mask.any(dim=1).view(-1, 1, 1).float()
        return self.behavior_proj((attn_weights * input_emb).sum(dim=1))

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
                dst_last_update_times, src_is_sender=None):
        """
        score every candidate, node ids are raw ids and every tensor is on self.device
        :param src_neighb_seq: Tensor, shape (batch_size, max_seq_length), the source's recent history
        :param src_neighb_seq_len: Tensor, shape (batch_size, )
        :param neighbors_interact_times: Tensor, shape (batch_size, max_seq_length)
        :param cur_times: Tensor, shape (batch_size, ), prediction times
        :param test_dst: Tensor, shape (batch_size, num_candidates), column 0 is the positive candidate
        :param dst_last_update_times: Tensor, shape (batch_size, num_candidates), -100000 when unknown
        :param src_is_sender: Tensor, shape (batch_size, max_seq_length), whether the source initiated the
        interaction of that history entry
        :return: Tensor, shape (batch_size, num_candidates)
        """
        batch_size, max_seq_length = src_neighb_seq.shape
        num_candidates = test_dst.shape[1]

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

        # ---- behavioral intent: gate the source state into each candidate's context ----
        if self.use_behavior_gate:
            z_t = self.extract_source_behavioral_state(input_emb, valid_mask)
            z_t = z_t.unsqueeze(1).expand(-1, num_candidates, -1)
            if self.fusion_mode == 'projected':
                context = self.proj_c(context)
                z_t = self.proj_z(z_t)
            gate = self.gate_layer(torch.cat([context, z_t], dim=-1))
            context = context + gate * z_t

        output = context
        if self.output_cat_time_intervals:
            output = torch.cat([output, dst_node_time_intervals_feat], dim=-1).float()
        if self.output_cat_repeat_times:
            output = torch.cat([output, repeat_times_feat], dim=-1).float()
        logits = self.output_layer(output.view(-1, output.shape[-1])).view(batch_size, num_candidates)

        # ---- additive correction from the reciprocity readout ----
        if self.use_reciprocity:
            src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
            time_scale = self._get_time_scale(src_elapsed, valid_mask)
            if src_is_sender is None:
                readout_features = torch.zeros(batch_size, num_candidates, 6, device=logits.device)
            else:
                readout_features = self.reciprocity_features(candidate_matches, src_is_sender, src_elapsed / time_scale)
            logits = logits + self.readout_gate * self.readout_head(readout_features).squeeze(dim=-1)
        return logits

    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = torch.bmm(mask_a.unsqueeze(1).transpose(1, 2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def compute_scores(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, src_is_sender=None):
        """
        move a batch to the device and score the candidates
        :param src_is_sender: Tensor, shape (batch_size, max_seq_length), whether the source initiated each
        history entry, recovered from the edge ids by the caller
        """
        return self.forward(src_neighb_seq=src_neighb_seq.to(self.device),
                            src_neighb_seq_len=src_neighb_seq_len.to(self.device),
                            neighbors_interact_times=src_neighb_interact_times.to(self.device),
                            cur_times=cur_pred_times.to(self.device),
                            test_dst=test_dst.to(self.device),
                            dst_last_update_times=dst_last_update_times.to(self.device),
                            src_is_sender=src_is_sender.to(self.device) if src_is_sender is not None else None)

    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                dst_last_update_times, src_is_sender=None):
        """
        [0]src_neighb_seq: [B, L]
        [1]src_neighb_seq_len: [B]
        [2]src_neighb_interact_times: [B, L]
        [3]cur_pred_times: [B]
        [4]test_dst: [B, 1+num_negs], the positive candidate is at column 0
        [5]dst_last_update_times: [B, 1+num_negs]
        [6]src_is_sender: [B, L]
        """
        logits = self.compute_scores(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times,
                                     test_dst, dst_last_update_times, src_is_sender)
        if self.loss_type == 'BPR':
            return logits[:, 0].flatten(), logits[:, 1:].flatten()
        return logits[:, 0].sigmoid().flatten(), logits[:, 1:].sigmoid().flatten()

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, src_is_sender=None):
        """
        the objective CRAFT is trained with: one negative per positive, BPR or BCE
        """
        positive_probabilities, negative_probabilities = self.predict(
            src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
            dst_last_update_times, src_is_sender)
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
