import torch
from torch import nn
import math
import copy
from models.modules import BPRLoss, MLP, FeedForward4CrossAttn

class CustomMultiHeadCrossAttention(nn.Module):
    """
    Custom Multi-Head Cross Attention for CRAFTV3.
    
    Mechanism:
      1. Compute initial attention matrix between Query Q and Key K.
      2. Squash the attention matrix across the query dimension by taking the mean -> 1 x seq_len.
      3. Normalize this 1D squashed attention vector.
      4. Scale each position/row of K with the corresponding index of the 1D squashed vector to get K'.
      5. Compute second attention matrix between Q and K'.
      6. Multiply with Value V to get final context representation.
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        rotary_emb_type=None,
        rotary_emb=None,
    ):
        super(CustomMultiHeadCrossAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.rotary_emb_type = rotary_emb_type
        self.rotary_emb = rotary_emb

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, query, attention_mask, key=None, interaction_time=None):
        if key is None:
            key = query
            query_time_slots = interaction_time
            key_time_slots = interaction_time
        else:
            if interaction_time is not None:
                query_time_slots = interaction_time[1]
                key_time_slots = interaction_time[0]
                
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.rotary_emb is not None:
            if self.rotary_emb_type == 'time':
                query_layer = self.rotary_emb(query_layer, input_pos=query_time_slots)
                key_layer = self.rotary_emb(key_layer, input_pos=key_time_slots)
            else:
                query_layer = self.rotary_emb(query_layer)
                key_layer = self.rotary_emb(key_layer)

        # Rearrange for matrix multiplication
        query_layer = query_layer.permute(0, 2, 1, 3) # [batch_size, num_heads, query_seq_len, head_size]
        key_layer = key_layer.permute(0, 2, 3, 1)   # [batch_size, num_heads, head_size, key_seq_len]
        value_layer = value_layer.permute(0, 2, 1, 3) # [batch_size, num_heads, key_seq_len, head_size]

        # ── Step 1: Initial Attention Scores between Q and K ──────────────────
        attention_scores_1 = torch.matmul(query_layer, key_layer)
        attention_scores_1 = attention_scores_1 / self.sqrt_attention_head_size
        attention_scores_1 = attention_scores_1 + attention_mask
        attention_probs_1 = self.softmax(attention_scores_1) # [batch_size, num_heads, query_seq_len, key_seq_len]

        # ── Step 2: Squash attention matrix by mean -> [batch_size, num_heads, 1, key_seq_len]
        squashed_attn = attention_probs_1.mean(dim=-2, keepdim=True)

        # ── Step 3: Normalize the 1D squashed attention ───────────────────────
        norm_squashed_attn = squashed_attn / (squashed_attn.sum(dim=-1, keepdim=True) + 1e-12)

        # ── Step 4: Scale each row/column of K to produce K' ──────────────────
        # key_layer is [batch_size, num_heads, head_size, key_seq_len]
        # norm_squashed_attn is [batch_size, num_heads, 1, key_seq_len]
        key_layer_prime = key_layer * norm_squashed_attn

        # ── Step 5: Compute final Attention with Q and K' ─────────────────────
        attention_scores_2 = torch.matmul(query_layer, key_layer_prime)
        attention_scores_2 = attention_scores_2 / self.sqrt_attention_head_size
        attention_scores_2 = attention_scores_2 + attention_mask
        attention_probs_2 = self.softmax(attention_scores_2)
        attention_probs_2 = self.attn_dropout(attention_probs_2)

        # ── Step 6: Multiply with Value V ─────────────────────────────────────
        context_layer = torch.matmul(attention_probs_2, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = hidden_states + query

        return hidden_states


class CustomCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        rotary_emb_type=None,
        rotary_emb=None,
        output_dim=None,
    ):
        super(CustomCrossAttentionLayer, self).__init__()
        self.multi_head_attention = CustomMultiHeadCrossAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, rotary_emb_type, rotary_emb
        )
        self.feed_forward = FeedForward4CrossAttn(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
            output_dim=output_dim
        )

    def forward(self, query, attention_mask, key=None, interaction_time=None):
        attention_output = self.multi_head_attention(query, attention_mask, key, interaction_time)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class CustomCrossAttention(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        rotary_emb=None,
        rotary_emb_type=None,
        output_dim=None,
    ):
        super(CustomCrossAttention, self).__init__()
        layer = CustomCrossAttentionLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
            rotary_emb_type=rotary_emb_type,
            rotary_emb=rotary_emb,
            output_dim=output_dim,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, query, attention_mask, key=None, output_all_encoded_layers=True, interaction_time=None):
        all_encoder_layers = []
        for layer_module in self.layer:
            query = layer_module(query, attention_mask, key, interaction_time)
            if output_all_encoded_layers:
                all_encoder_layers.append(query)
        if not output_all_encoded_layers:
            all_encoder_layers.append(query)
        return all_encoder_layers


class CRAFTV3(torch.nn.Module):
    """
    CRAFTV3: CRAFT with Custom Squashed-Attention Rescaling Mechanism.
    
    1. Computes initial cross-attention scores between Query Q and Key K.
    2. Squashes attention matrix by mean to form a 1D vector of shape (1 x seq_len).
    3. Normalizes the 1D vector.
    4. Rescales rows of K by the normalized 1D vector to produce K'.
    5. Computes final cross-attention between Q and K', and multiplies with V.
    """

    def __init__(self, n_layers, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob,
                 hidden_act, layer_norm_eps, initializer_range, n_nodes, max_seq_length,
                 device, loss_type, use_pos=True, input_cat_time_intervals=False,
                 output_cat_time_intervals=True, output_cat_repeat_times=False,
                 num_output_layer=1, emb_dropout_prob=0.1, skip_connection=False):
        super(CRAFTV3, self).__init__()
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
        self.output_cat_time_intervals = output_cat_time_intervals
        self.output_cat_repeat_times = output_cat_repeat_times
        self.emb_dropout_prob = emb_dropout_prob
        self.node_embedding = nn.Embedding(
            self.n_nodes + 1, self.hidden_size, padding_idx=0
        )
        self.use_pos = use_pos
        self.input_cat_time_intervals = input_cat_time_intervals
        if use_pos:
            self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        output_dim = 0 
        if self.input_cat_time_intervals:
            trm_input_dim = self.hidden_size * 2
        else:
            trm_input_dim = self.hidden_size
            
        self.cross_attention = CustomCrossAttention(
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
        self.device = device
        
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

    def forward(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times, cur_times, test_dst=None, dst_last_update_times=None):
        bs = src_neighb_seq.shape[0]
        src_neighb_seq_len[src_neighb_seq_len == 0] = 1
        neighb_emb = self.node_embedding(src_neighb_seq)
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
            repeat_times = test_dst.view(bs, test_dst.shape[1], 1) == src_neighb_seq.view(bs, 1, src_neighb_seq.shape[1])
            repeat_times = repeat_times.sum(dim=-1).unsqueeze(-1).float()
            repeat_times_feat = self.repeat_times_projection(repeat_times.float()).view(bs, -1, self.hidden_size)
            repeat_times_feat = self.LayerNorm_repeat_times(repeat_times_feat)
            repeat_times_feat = self.dropout(repeat_times_feat)
        if self.use_pos:
            position_ids = torch.arange(
                src_neighb_seq.size(1), dtype=torch.long, device=src_neighb_seq.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(src_neighb_seq)
            position_embedding = self.position_embedding(position_ids)
            input_emb = neighb_emb + position_embedding
        else:
            input_emb = neighb_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.emb_dropout(input_emb)
        if self.input_cat_time_intervals:
            src_neighbor_interact_time_intervals = cur_times.view(-1, 1) - neighbors_interact_times
            src_neighbor_interact_time_intervals[src_neighb_seq == 0] = -100000
            src_neighb_time_embedding = self.time_projection(src_neighbor_interact_time_intervals.to(self.device).float().view(-1, 1)).view(src_neighb_seq.shape[0], src_neighb_seq.shape[1], -1)
            src_neighb_time_embedding = self.LayerNorm_time_intervals(src_neighb_time_embedding)
            src_neighb_time_embedding = self.dropout(src_neighb_time_embedding)
            input_emb = torch.cat([input_emb, src_neighb_time_embedding], dim=-1)
        
        attention_mask = src_neighb_seq != 0
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
    
    def get_attention_mask(self, mask_a, mask_b):
        extended_attention_mask = torch.bmm(mask_a.unsqueeze(1).transpose(1, 2), mask_b.unsqueeze(1).float()).bool().unsqueeze(1)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
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
