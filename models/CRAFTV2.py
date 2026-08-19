import torch
from torch import nn
from models.modules import CrossAttention
from models.modules import BPRLoss, MLP

class CRAFTV2(torch.nn.Module):
    """
    CRAFTV2: Behavioral Intent Guided CRAFT.
    
    1. Extracts a 1D behavioral intent vector from the M recent neighbor interactions
       using self-attentive intent pooling.
    2. Calculates the similarity score between the behavioral intent vector and
       the node embeddings of the M recent neighbors.
    3. Keeps only the Top-K most similar/relevant neighbor nodes.
    4. Passes the Top-K intent-filtered neighbor sequence to the CRAFT Cross-Attention module.
    """

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob,
                 hidden_act, layer_norm_eps, initializer_range, n_nodes, max_seq_length,
                 top_k_intent=20, device='cpu', loss_type='BCE', use_pos=True,
                 input_cat_time_intervals=False, output_cat_time_intervals=True,
                 output_cat_repeat_times=False, num_output_layer=1, emb_dropout_prob=0.1,
                 skip_connection=False):
        super(CRAFTV2, self).__init__()
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
        self.top_k_intent = top_k_intent
        self.output_cat_time_intervals = output_cat_time_intervals
        self.output_cat_repeat_times = output_cat_repeat_times
        self.emb_dropout_prob = emb_dropout_prob
        self.device = device
        
        self.node_embedding = nn.Embedding(
            self.n_nodes + 1, self.hidden_size, padding_idx=0
        )
        self.use_pos = use_pos
        self.input_cat_time_intervals = input_cat_time_intervals
        
        # Position embedding for recent neighbors and filtered top-k sequence
        if use_pos:
            self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
            self.topk_position_embedding = nn.Embedding(self.top_k_intent, self.hidden_size)

        # ── Behavioral Intent Network ──────────────────────────────────────────
        # Computes attention weight for each historical interaction to form 1D intent vector
        self.intent_attn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        self.intent_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
            nn.Dropout(self.hidden_dropout_prob)
        )

        output_dim = 0 
        if self.input_cat_time_intervals:
            trm_input_dim = self.hidden_size * 2
        else:
            trm_input_dim = self.hidden_size

        # CrossAttention module for Top-K intent filtered neighbors
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
        output_dim += trm_input_dim

        if self.output_cat_time_intervals or self.input_cat_time_intervals:
            self.time_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size,
                                       output_dim=self.hidden_size, dropout=self.hidden_dropout_prob,
                                       use_act=True, skip_connection=skip_connection)
        if self.output_cat_repeat_times:
            self.repeat_times_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size,
                                               output_dim=self.hidden_size, dropout=self.hidden_dropout_prob,
                                               use_act=True, skip_connection=skip_connection)
        if self.output_cat_time_intervals:
            output_dim += self.hidden_size
        if self.output_cat_repeat_times:
            output_dim += self.hidden_size

        self.output_layer = MLP(num_layers=num_output_layer, input_dim=output_dim, hidden_dim=output_dim,
                                output_dim=1, dropout=self.hidden_dropout_prob, use_act=True,
                                skip_connection=skip_connection)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_time_intervals = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_repeat_times = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout(self.emb_dropout_prob)
        self.loss_type = loss_type

        if self.loss_type == "BCE":
            self.loss_fct = nn.BCELoss()
        elif self.loss_type == "BPR":
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

    def extract_intent_and_filter_topk(self, src_neighb_seq, src_neighb_interact_times):
        """
        Extracts 1D behavioral intent vector from M recent neighbors and retains Top-K most similar neighbors.
        
        :param src_neighb_seq: Tensor of shape (bs, M) with node IDs
        :param src_neighb_interact_times: Tensor of shape (bs, M) with interaction timestamps
        :return: (topk_seq, topk_times, topk_len, intent_vector)
        """
        bs, seq_len = src_neighb_seq.shape
        k = min(self.top_k_intent, seq_len)

        # 1. Embed historical neighbor nodes
        neighb_emb = self.node_embedding(src_neighb_seq) # [bs, M, H]

        if self.use_pos:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=src_neighb_seq.device)
            position_ids = position_ids.unsqueeze(0).expand_as(src_neighb_seq)
            position_embedding = self.position_embedding(position_ids)
            h_seq = neighb_emb + position_embedding
        else:
            h_seq = neighb_emb

        h_seq = self.LayerNorm(h_seq)
        h_seq = self.emb_dropout(h_seq)

        # 2. Self-Attentive Intent Pooling to get 1D Intent Vector
        attn_logits = self.intent_attn(h_seq) # [bs, M, 1]
        pad_mask = (src_neighb_seq == 0).unsqueeze(-1) # [bs, M, 1]
        attn_logits = attn_logits.masked_fill(pad_mask, -1e9)

        attn_weights = torch.softmax(attn_logits, dim=1) # [bs, M, 1]
        
        # Zero out weights for sequences that are entirely padding
        all_zeros = (src_neighb_seq != 0).sum(dim=1, keepdim=True).unsqueeze(-1) == 0
        attn_weights = torch.where(all_zeros, torch.zeros_like(attn_weights), attn_weights)

        # Aggregate weighted embeddings to obtain 1D intent vector
        intent_vector = torch.sum(attn_weights * h_seq, dim=1) # [bs, H]
        intent_vector = self.intent_proj(intent_vector) # [bs, H]

        # 3. Calculate similarity score between intent vector and neighbor node embeddings
        # Normalized cosine similarity: [bs, 1, H] * [bs, M, H] -> [bs, M]
        norm_intent = torch.nn.functional.normalize(intent_vector.unsqueeze(1), p=2, dim=-1)
        norm_neighb = torch.nn.functional.normalize(neighb_emb, p=2, dim=-1)
        similarity = torch.sum(norm_intent * norm_neighb, dim=-1) # [bs, M]

        # Mask padding items so they have minimal score
        similarity = similarity.masked_fill(src_neighb_seq == 0, -1e9)

        # 4. Select Top-K most similar nodes
        topk_indices = torch.topk(similarity, k=k, dim=1, largest=True).indices # [bs, K]
        # Sort indices to preserve relative chronological order
        topk_indices, _ = torch.sort(topk_indices, dim=1)

        topk_seq = torch.gather(src_neighb_seq, 1, topk_indices) # [bs, K]
        topk_times = torch.gather(src_neighb_interact_times, 1, topk_indices) # [bs, K]
        topk_len = (topk_seq != 0).sum(dim=1) # [bs]

        return topk_seq, topk_times, topk_len, intent_vector

    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = torch.bmm(mask_a.unsqueeze(1).transpose(1, 2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst=None, dst_last_update_times=None):
        bs = src_neighb_seq.shape[0]

        # ── 1. Intent Extraction & Top-K Filtering ─────────────────────────────
        topk_seq, topk_times, topk_len, intent_vector = self.extract_intent_and_filter_topk(
            src_neighb_seq, neighbors_interact_times
        )
        topk_len[topk_len == 0] = 1
        k_seq_len = topk_seq.shape[1]

        # ── 2. Time intervals & repeat times features ──────────────────────────
        if self.output_cat_time_intervals:
            dst_last_update_intervals = cur_times.view(-1, 1) - dst_last_update_times
            dst_last_update_intervals[dst_last_update_times < -1] = -100000 
            dst_last_update_intervals = dst_last_update_intervals.to(self.device)
            dst_node_time_intervals_feat = self.time_projection(dst_last_update_intervals.float().view(-1, 1)).view(dst_last_update_intervals.shape[0], dst_last_update_intervals.shape[1], -1)
            dst_node_time_intervals_feat = self.LayerNorm_time_intervals(dst_node_time_intervals_feat)
            dst_node_time_intervals_feat = self.dropout(dst_node_time_intervals_feat)

        test_dst_emb = self.node_embedding(test_dst)
        test_dst_emb = self.LayerNorm(test_dst_emb.view(bs, -1, self.hidden_size))
        test_dst_emb = self.emb_dropout(test_dst_emb)

        if self.output_cat_repeat_times:
            repeat_times = test_dst.view(bs, test_dst.shape[1], 1) == topk_seq.view(bs, 1, topk_seq.shape[1])
            repeat_times = repeat_times.sum(dim=-1).unsqueeze(-1).float()
            repeat_times_feat = self.repeat_times_projection(repeat_times.float()).view(bs, -1, self.hidden_size)
            repeat_times_feat = self.LayerNorm_repeat_times(repeat_times_feat)
            repeat_times_feat = self.dropout(repeat_times_feat)

        # ── 3. Embed Top-K Filtered Neighbor Sequence ──────────────────────────
        topk_neighb_emb = self.node_embedding(topk_seq)
        if self.use_pos:
            position_ids = torch.arange(k_seq_len, dtype=torch.long, device=topk_seq.device)
            position_ids = position_ids.unsqueeze(0).expand_as(topk_seq)
            if k_seq_len <= self.top_k_intent:
                position_embedding = self.topk_position_embedding(position_ids)
            else:
                position_embedding = self.position_embedding(position_ids)
            input_emb = topk_neighb_emb + position_embedding
        else:
            input_emb = topk_neighb_emb

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.emb_dropout(input_emb)

        if self.input_cat_time_intervals:
            src_neighbor_interact_time_intervals = cur_times.view(-1, 1) - topk_times
            src_neighbor_interact_time_intervals[topk_seq == 0] = -100000
            src_neighb_time_embedding = self.time_projection(src_neighbor_interact_time_intervals.to(self.device).float().view(-1, 1)).view(topk_seq.shape[0], topk_seq.shape[1], -1)
            src_neighb_time_embedding = self.LayerNorm_time_intervals(src_neighb_time_embedding)
            src_neighb_time_embedding = self.dropout(src_neighb_time_embedding)
            input_emb = torch.cat([input_emb, src_neighb_time_embedding], dim=-1)

        # ── 4. Cross-Attention over Top-K Intent Neighbors ─────────────────────
        attention_mask = topk_seq != 0
        test_dst_mask = torch.ones(test_dst_emb.shape[0], test_dst_emb.shape[1]).to(self.device)
        extended_attention_mask = self.get_attention_mask(test_dst_mask, mask_b=attention_mask)
        output = self.cross_attention(
            test_dst_emb, extended_attention_mask, input_emb, output_all_encoded_layers=True
        )[-1]

        if self.output_cat_time_intervals:
            if output is None:
                output = dst_node_time_intervals_feat
            else:
                output = torch.cat([output, dst_node_time_intervals_feat], dim=-1).float()
        if self.output_cat_repeat_times:
            if output is None:
                output = repeat_times_feat
            else:
                output = torch.cat([output, repeat_times_feat], dim=-1).float()

        output = self.output_layer(output.view(-1, output.shape[-1])).view(output.shape[0], output.shape[1], -1)
        return output

    def predict(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times):
        src_neighb_seq = src_neighb_seq.to(self.device) - self.dst_min_idx + 1
        test_dst = test_dst.to(self.device) - self.dst_min_idx + 1
        src_neighb_seq[src_neighb_seq < 0] = 0
        src_neighb_interact_times = src_neighb_interact_times.to(self.device)
        src_neighb_seq_len = src_neighb_seq_len.to(self.device)
        logits = self.forward(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times.to(self.device), test_dst=test_dst, dst_last_update_times=dst_last_update_times.to(self.device))
        if self.loss_type == 'BPR':
            positive_probabilities = logits[:, 0].flatten()
            negative_probabilities = logits[:, 1:].flatten()
        else:
            positive_probabilities = logits[:, 0].sigmoid().flatten()
            negative_probabilities = logits[:, 1:].sigmoid().flatten()
        return positive_probabilities, negative_probabilities

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times):
        positive_probabilities, negative_probabilities = self.predict(src_neighb_seq, src_neighb_seq_len, src_neighb_interact_times, cur_pred_times, test_dst, dst_last_update_times)
        bs = test_dst.shape[0]
        if self.loss_type == 'BPR': 
            negative_probabilities = negative_probabilities.flatten()
            positive_probabilities = positive_probabilities.flatten()
            loss = self.loss_fct(positive_probabilities, negative_probabilities)
        predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
        labels = torch.cat([torch.ones(bs), torch.zeros(bs)], dim=0).to(self.device)
        if self.loss_type == 'BCE':
            loss = self.loss_fct(predicts, labels)
        elif self.loss_type != 'BPR':
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented! Only BCE and BPR are supported!")
        return loss, predicts, labels
