import math
import torch
import torch.nn.functional as F
from torch import nn
from models.modules import BPRLoss, MLP


class MYMODEL(torch.nn.Module):
    """
    MYMODEL: ranking candidate destinations by a delay-resolved excitation field.

    Motivation
    ----------
    Every model in this repository turns the source's history into one pooled vector and then
    compares it with the candidate: TGAT/TCL/DyGFormer pool with self-attention, GraphMixer with an
    MLP, SGNN-HN with a session graph, CRAFT pools with cross-attention in which the candidate is
    the query. Time enters those models as a feature that is encoded and then mixed into the pooling
    (a time encoding on the keys, or, in CRAFT, two scalars - the elapsed time of the candidate's
    last event and a repeat count - concatenated to the pooled vector just before the output MLP).
    Identity and time therefore interact only through a final MLP, which means such a model can
    learn "this candidate matches the context" and "recent things matter", but it cannot represent
    *at which delay* one interaction makes another one likely.

    That delay structure is the dominant signal in this task. A user re-listens to an artist within
    minutes but re-watches a lecture after a day; a Wikipedia editor rarely edits the same page twice
    in a row (inhibition), while a subreddit visit repeats in bursts; an item that was consumed
    milliseconds ago is often *less* likely than one consumed an hour ago. Exponential-decay memory
    (JODIE/DyRep/TGN), time-interval sampling (CAWN), and a plain repeat count (EdgeBank, CRAFT-R)
    can only express monotone decay or delay-blind counting.

    The operator
    ------------
    MYMODEL replaces candidate-conditioned attention with an explicit excitation field. The score of
    candidate d for source s at time t is the log-intensity of a marked point process,

        log lambda(d | s, t) = base_d + f( E(d) , S(d) , I(d) )

    whose terms come from one shared construction: a learned basis over *delays*, and a factorized
    tensor that says how strongly a past interaction with item a excites item d at each delay.

    1. Delay basis. Elapsed times are normalized by a running median (a buffer updated during
       training only, so the model is independent of the time unit of the dataset) and mapped to
       z = log1p(dt). The basis is one constant function plus M-1 Gaussian bumps in z with learned
       centers mu_m and widths sigma_m, i.e. phi_m(dt) picks out "about this long ago". Bumps, unlike
       exponentials, are non-monotone, so a characteristic delay or a refractory period is
       representable; the constant basis recovers delay-blind counting.

    2. Event excitation E(d), a CP-factorized (trigger item x response item x delay) tensor.
       With trigger vectors g_a = W_g e_a and response vectors r_d = W_r e_d, the excitation of the
       pair (a -> d) at delay dt is sum_m phi_m(dt) * <g_a * D_m, r_d>, where D_m are R diagonal
       metrics per basis. The pairwise, delay-resolved tensor T[a, d, m] is thus factorized as
       sum_h g_a[h] D_m[h] r_d[h], which is what keeps it affordable: because the sum over history
       entries j moves inside, the whole field collapses to M*R delay summaries of the source,
           G_m = sum_j phi_m(dt_j) g_{h_j},   E(d)_{m,r} = <G_m * D_{m,r}, r_d> / sqrt(H),
       so no batch x candidate x history tensor is ever formed. Cost is O(L*M*H + C*M*R*H) against
       CRAFT's O(C*L*H) attention, which matters at evaluation time with 100 negatives per positive.
       E(d) is the candidate's excitation profile across delays; the readout f is a small network on
       that profile, so the effect of the profile can saturate, be non-monotone, or be negative
       (inhibition), none of which decay-based or count-based models can express.

    3. Self-dynamics S(d). The candidate's own last interaction with anybody (the one extra input
       CRAFT also receives) enters through the same basis: S(d)_m = phi_m(dt_d) * <P_m, r_d>, an
       item-specific recurrence profile - how long after its own last event an item tends to recur.

    4. Novelty channel I(d). For candidates that never appear in the source's history the excitation
       field is empty, so the field is complemented by P interest prototypes that pool the history
       with delay-aware attention weights (the delay basis also enters the pooling logits), giving
       I(d)_p = <q_p(s), r_d>. Multiple prototypes keep multi-modal tastes separable instead of
       averaging them into one vector. This is additive with the excitation, as intensities are in a
       point process - there is no gating heuristic.

    What it subsumes (useful as an ablation table)
    ---------------------------------------------
      * EdgeBank / CRAFT's repeat count: constant basis only (M = 1), identity metric, linear readout.
      * TGN / JODIE style exponential decay: one monotone basis, linear readout.
      * SLRC-style Hawkes recommendation: self-dynamics term alone.
      * A pure factorization model (SASRec/SGNN-HN-like scoring): novelty channel alone.
    Each is reachable with a flag, so the contribution of the delay-resolved pairwise tensor is
    measurable rather than asserted.

    Fairness and budget
    -------------------
    The model consumes exactly the inputs CRAFT consumes (the source's recent history with its
    timestamps, the candidate ids, and each candidate's last-update time), is trained with the same
    single negative per positive and the same BPR/BCE loss, and is evaluated by the same protocol.
    It has no attention over candidates and no transformer block, so at equal hidden size it is
    *smaller* than CRAFT; raise --embedding_size, --num_delay_bases or --num_response_channels to
    spend the remaining budget if strict parameter parity is wanted.
    """

    def __init__(self, hidden_size, n_nodes, max_seq_length, device, loss_type, num_delay_bases=8,
                 num_response_channels=4, num_interests=4, num_layers=1, hidden_dropout_prob=0.1,
                 emb_dropout_prob=0.1, layer_norm_eps=1e-12, initializer_range=0.02, readout='mlp',
                 use_self_dynamics=True, use_novelty=True, delay_basis='gaussian',
                 time_scale_momentum=0.99):
        """
        :param hidden_size: int, dimension of the item embeddings and of the trigger/response spaces
        :param n_nodes: int, number of candidate (destination role) nodes
        :param max_seq_length: int, number of history entries per source
        :param num_delay_bases: int, number of delay basis functions, the first one is the constant
        :param num_response_channels: int, number of diagonal metrics per delay basis
        :param num_interests: int, number of interest prototypes of the novelty channel
        :param num_layers: int, depth of the trigger and response projections (1 is linear)
        :param readout: str, 'mlp' or 'linear', the readout on the excitation profile
        :param use_self_dynamics: bool, whether to use the candidate's own recurrence profile
        :param use_novelty: bool, whether to use the interest prototypes
        :param delay_basis: str, 'gaussian' (non-monotone bumps) or 'exponential' (monotone decay,
        for the ablation that reduces the field to decaying memory)
        """
        super(MYMODEL, self).__init__()
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.max_seq_length = max_seq_length
        self.device = device
        self.num_delay_bases = num_delay_bases
        self.num_response_channels = num_response_channels
        self.num_interests = num_interests if use_novelty else 0
        self.readout_type = readout
        self.use_self_dynamics = use_self_dynamics
        self.use_novelty = use_novelty
        self.delay_basis = delay_basis
        self.initializer_range = initializer_range
        self.time_scale_momentum = time_scale_momentum
        self.eps = 1e-6

        # items live in one embedding table, as in CRAFT, plus a scalar base intensity per item
        self.node_embedding = nn.Embedding(self.n_nodes + 1, self.hidden_size, padding_idx=0)
        self.base_rate = nn.Embedding(self.n_nodes + 1, 1, padding_idx=0)
        # an item plays two different roles: it triggers later interactions, and it responds to earlier ones
        self.trigger_projection = MLP(num_layers=num_layers, input_dim=self.hidden_size, hidden_dim=self.hidden_size,
                                      output_dim=self.hidden_size, dropout=hidden_dropout_prob, use_act=True,
                                      skip_connection=False)
        self.response_projection = MLP(num_layers=num_layers, input_dim=self.hidden_size, hidden_dim=self.hidden_size,
                                       output_dim=self.hidden_size, dropout=hidden_dropout_prob, use_act=True,
                                       skip_connection=False)

        # delay basis: basis 0 is the constant, the remaining ones are bumps (or decays) in log-delay space
        num_shaped_bases = max(self.num_delay_bases - 1, 0)
        if num_shaped_bases > 0:
            self.delay_centers = nn.Parameter(torch.linspace(0.0, 4.0, num_shaped_bases))
            self.delay_widths = nn.Parameter(torch.full((num_shaped_bases,), 0.8))
        # diagonal metrics of the factorized (trigger, response, delay) excitation tensor
        self.response_metrics = nn.Parameter(torch.empty(self.num_delay_bases, self.num_response_channels,
                                                         self.hidden_size))
        if self.use_self_dynamics:
            # per-delay direction of the candidate's own recurrence profile
            self.self_dynamics = nn.Parameter(torch.empty(self.num_delay_bases, self.hidden_size))
        if self.use_novelty:
            self.interest_queries = nn.Parameter(torch.empty(self.num_interests, self.hidden_size))
            # the interest pooling is delay-aware: each prototype has a preference over the delay basis
            self.interest_delay_weights = nn.Parameter(torch.zeros(self.num_interests, self.num_delay_bases))

        # the profile of a candidate: excitation (M x R), self-dynamics (M), interests (P), history flag
        self.profile_dim = self.num_delay_bases * self.num_response_channels + self.num_interests + 1
        if self.use_self_dynamics:
            self.profile_dim += self.num_delay_bases
        self.readout = MLP(num_layers=2 if readout == 'mlp' else 1, input_dim=self.profile_dim,
                           hidden_dim=self.hidden_size, output_dim=1, dropout=hidden_dropout_prob,
                           use_act=True, skip_connection=False)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.emb_dropout = nn.Dropout(emb_dropout_prob)

        # running scale of the elapsed times, so the delay basis is comparable across datasets
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
        self._init_field_parameters()

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

    def _init_field_parameters(self):
        with torch.no_grad():
            # the first response channel starts as the plain inner product, the others as random metrics,
            # so the channels are not symmetric at initialization
            self.response_metrics.normal_(mean=0.0, std=1.0 / math.sqrt(self.num_response_channels))
            self.response_metrics[:, 0, :] = 1.0
            if self.use_self_dynamics:
                self.self_dynamics.normal_(mean=0.0, std=self.initializer_range)
            if self.use_novelty:
                self.interest_queries.normal_(mean=0.0, std=1.0)
            # nothing is predicted for the padding item
            self.node_embedding.weight[0].zero_()
            self.base_rate.weight[0].zero_()

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

    def _get_delay_basis(self, elapsed_times: torch.Tensor, valid_mask: torch.Tensor, time_scale: torch.Tensor):
        """
        evaluate the delay basis on elapsed times, the first basis is the constant one
        :param elapsed_times: Tensor, shape (..., ), raw elapsed times
        :param valid_mask: Tensor, shape (..., ), whether the event exists
        :param time_scale: Tensor, shape (1, ), running scale of the elapsed times
        :return: Tensor, shape (..., num_delay_bases), zero where the event does not exist
        """
        log_delays = torch.log1p((elapsed_times / time_scale).clamp(min=0.0)).unsqueeze(-1)
        basis = [torch.ones_like(log_delays)]
        if self.num_delay_bases > 1:
            if self.delay_basis == 'exponential':
                # ablation: monotone decay with learned rates, i.e. classical Hawkes triggering
                rates = F.softplus(self.delay_widths) + self.eps
                basis.append(torch.exp(-log_delays / rates.view(1, -1)))
            else:
                widths = F.softplus(self.delay_widths) + 1e-2
                basis.append(torch.exp(-0.5 * ((log_delays - self.delay_centers.view(1, -1)) / widths.view(1, -1)) ** 2))
        return torch.cat(basis, dim=-1) * valid_mask.unsqueeze(-1).to(log_delays.dtype)

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst,
                dst_last_update_times):
        """
        log-intensity of every candidate, all inputs are reindexed into the candidate id space and on self.device
        :param src_neighb_seq: Tensor, shape (batch_size, max_seq_length), the source's recent history, 0 is padding
        :param src_neighb_seq_len: Tensor, shape (batch_size, ), number of valid history entries
        :param neighbors_interact_times: Tensor, shape (batch_size, max_seq_length)
        :param cur_times: Tensor, shape (batch_size, ), prediction times
        :param test_dst: Tensor, shape (batch_size, num_candidates), candidates, column 0 is the positive one
        :param dst_last_update_times: Tensor, shape (batch_size, num_candidates), the candidate's own last
        interaction time, -100000 when it has none
        :return: Tensor, shape (batch_size, num_candidates)
        """
        batch_size = src_neighb_seq.shape[0]
        valid_mask = src_neighb_seq != 0
        has_history = valid_mask.any(dim=1)

        # delays of the history entries and of the candidates' own last events
        src_elapsed = (cur_times.float().view(-1, 1) - neighbors_interact_times.float()).clamp(min=0.0)
        time_scale = self._get_time_scale(src_elapsed, valid_mask)
        src_basis = self._get_delay_basis(src_elapsed, valid_mask, time_scale)
        dst_has_history = dst_last_update_times > -1
        dst_elapsed = (cur_times.float().view(-1, 1) - dst_last_update_times.float()).clamp(min=0.0)

        # the two roles of an item: trigger of later interactions, and response to earlier ones
        hist_emb = self.emb_dropout(self.LayerNorm(self.node_embedding(src_neighb_seq)))
        cand_emb = self.emb_dropout(self.LayerNorm(self.node_embedding(test_dst)))
        triggers = self.trigger_projection(hist_emb)
        responses = self.response_projection(cand_emb)

        # event excitation: delay summaries of the source, read out per candidate through the diagonal metrics
        delay_summaries = torch.einsum('blm,blh->bmh', src_basis, triggers)
        metric_summaries = delay_summaries.unsqueeze(2) * self.response_metrics.unsqueeze(0)
        excitation = torch.einsum('bmrh,bnh->bnmr', metric_summaries, responses) / math.sqrt(self.hidden_size)
        profile = [excitation.flatten(start_dim=2)]

        # self-dynamics: the candidate's own recurrence profile over delays
        if self.use_self_dynamics:
            dst_basis = self._get_delay_basis(dst_elapsed, dst_has_history, time_scale)
            self_affinity = torch.einsum('mh,bnh->bnm', self.self_dynamics, responses) / math.sqrt(self.hidden_size)
            profile.append(dst_basis * self_affinity)

        # novelty channel: delay-aware interest prototypes, for candidates the history says nothing about
        if self.use_novelty:
            interest_logits = torch.einsum('ph,blh->bpl', self.interest_queries, triggers) / math.sqrt(self.hidden_size)
            interest_logits = interest_logits + torch.einsum('pm,blm->bpl', self.interest_delay_weights, src_basis)
            interest_logits = interest_logits.masked_fill(~valid_mask.unsqueeze(1), -1e10)
            interests = torch.einsum('bpl,blh->bph', torch.softmax(interest_logits, dim=-1), triggers)
            interests = interests * has_history.view(-1, 1, 1).to(interests.dtype)
            profile.append(torch.einsum('bph,bnh->bnp', interests, responses) / math.sqrt(self.hidden_size))

        profile.append(dst_has_history.to(excitation.dtype).unsqueeze(-1))
        profile = torch.cat(profile, dim=-1)
        return self.readout(profile).squeeze(dim=-1) + self.base_rate(test_dst).squeeze(dim=-1)

    def compute_scores(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times):
        """
        reindex the raw node ids of a batch into the candidate id space and score the candidates
        """
        src_neighb_seq = src_neighb_seq.to(self.device) - self.dst_min_idx + 1
        test_dst = test_dst.to(self.device) - self.dst_min_idx + 1
        # padding entries and source-role nodes fall outside the candidate id space
        src_neighb_seq[src_neighb_seq < 0] = 0
        return self.forward(src_neighb_seq=src_neighb_seq,
                            src_neighb_seq_len=src_neighb_seq_len.to(self.device),
                            neighbors_interact_times=src_neighb_interact_times.to(self.device),
                            cur_times=cur_pred_times.to(self.device),
                            test_dst=test_dst,
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
        the same objective CRAFT is trained with: one negative per positive, BPR or BCE
        """
        positive_probabilities, negative_probabilities = self.predict(src_neighb_seq, src_neighb_seq_len,
                                                                     src_neighb_interact_times, cur_pred_times,
                                                                     test_dst, dst_last_update_times)
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
