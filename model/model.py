import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True, drop=None):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.drop = drop
        if drop is not None:
            self.dropout = nn.Dropout(p=drop)
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        if self.drop is not None:
            x = self.dropout(x)
        for conv in self.convs:
            x = conv(x)
        return x


def get_activation(activation_str):
    if activation_str == 'relu':
        return F.relu
    elif activation_str == 'leaky_relu':
        return lambda x: F.leaky_relu(x, negative_slope=0.2)
    elif activation_str == 'sigmoid':
        return torch.sigmoid
    elif activation_str == 'tanh':
        return torch.tanh
    elif activation_str == 'none':
        return None
    else:
        raise ValueError(f"Unsupported activation: {activation_str}")


# model.py
class GatedMultiHeadFusion(nn.Module):
    def __init__(self, num_heads, d_model, bn_decay=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.total_dim = num_heads * d_model
        self.gate_net = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim // 2),
            nn.ReLU(),
            nn.Linear(self.total_dim // 2, num_heads),
            nn.Softmax(dim=-1)
        )
        self.bn = nn.BatchNorm2d(self.total_dim, momentum=bn_decay)
        self.dim_recovery = nn.Sequential(
            nn.Linear(d_model, self.total_dim),
            nn.ReLU()
        ) if d_model != self.total_dim else nn.Identity()

    def forward(self, multi_head_output):
    
        batch_size, num_steps, num_vertex, total_dim = multi_head_output.shape
        gate_weights = self.gate_net(multi_head_output)
        reshaped = multi_head_output.view(batch_size, num_steps, num_vertex,
                                          self.num_heads, self.d_model)

        gated = reshaped * gate_weights.unsqueeze(-1)

        fused = gated.view(batch_size, num_steps, num_vertex, total_dim)

        fused = fused.permute(0, 3, 1, 2)
        fused = self.bn(fused)
        fused = fused.permute(0, 2, 3, 1) 

        return fused


class STEmbedding(nn.Module):
    def __init__(self, D, args, drop=None):
        super(STEmbedding, self).__init__()
        self.D = D 
        se_bn_decay =0.1
        se_act1 = 'relu'
        se_act2 = 'none'
        se_drop = 0.01
        te_bn_decay = 0.1
        te_act1 ='relu'
        te_act2 ='none'
        te_drop =0.01

        self.FC_se = FC(
            input_dims=[64, D],
            units=[D, D],
            activations=[get_activation(se_act1), get_activation(se_act2)],
            bn_decay=se_bn_decay,
            drop=se_drop
        )
        self.FC_te = nn.Sequential(
            FC(
                input_dims=[3, D],
                units=[D, D],
                activations=[get_activation(te_act1), get_activation(te_act2)],
                bn_decay=te_bn_decay,
                drop=te_drop
            ),
            FC(
                input_dims=[D, D],
                units=[D, D],
                activations=[get_activation(te_act1), get_activation(te_act2)],
                bn_decay=te_bn_decay,
                drop=te_drop
            )
        )

    def forward(self, SE, TE):
 
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        TE = TE.unsqueeze(2).float()
        TE = self.FC_te(TE)
        STE = SE + TE

        return STE


class FourierFilter(nn.Module):
    def __init__(self, args, filter_types=['adaptive', 'low', 'high'], drop=None):
        super(FourierFilter, self).__init__()
        self.filter_types = filter_types
        self.num_filters = len(filter_types)
        self.args = args

        fft_bn_decay = 0.1
        fft_act1 ='relu'
        fft_act2 ='none'
        fft_drop =0.01

        if 'adaptive' in filter_types:
            self.adaptive_weights = nn.Parameter(torch.randn(1))
        self.distribution_correction = FC(
            input_dims=[1, 1],
            units=[1, 1],
            activations=[get_activation(fft_act1), get_activation(fft_act2)],
            bn_decay=fft_bn_decay,
            drop=fft_drop
        )

    def forward(self, x):
        original = x
        batch_size, num_steps, num_nodes, feat_dim = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        freqs = torch.fft.rfftfreq(num_steps, d=1 / num_steps, device=x.device)
        filtered_results = []

        if 'adaptive' in self.filter_types:
            adaptive_filter = torch.sigmoid(self.adaptive_weights)
            adaptive_filter = adaptive_filter.view(1, -1, 1, 1)
            adaptive_result = x_fft * adaptive_filter
            filtered_results.append(torch.fft.irfft(adaptive_result, n=num_steps, dim=1, norm='ortho'))
        if 'low' in self.filter_types:
            cutoff = 0.1 * freqs.max()
            lowpass = torch.where(freqs <= cutoff, 1.0, 0.0).view(1, -1, 1, 1)
            lowpass_result = x_fft * lowpass
            filtered_results.append(torch.fft.irfft(lowpass_result, n=num_steps, dim=1, norm='ortho'))
        if 'high' in self.filter_types:
            cutoff = 0.1 * freqs.max()
            highpass = torch.where(freqs >= cutoff, 1.0, 0.0).view(1, -1, 1, 1)
            highpass_result = x_fft * highpass
            filtered_results.append(torch.fft.irfft(highpass_result, n=num_steps, dim=1, norm='ortho'))

        filtered_results.append(original)
        fused = torch.cat(filtered_results, dim=-1)
        gate = torch.softmax(torch.randn_like(fused), dim=-1)
        fused = torch.sum(fused * gate, dim=-1, keepdim=True)
        corrected = self.distribution_correction(fused)
        mean_original = torch.mean(original, dim=(0, 1, 2), keepdim=True)
        std_original = torch.std(original, dim=(0, 1, 2), keepdim=True)
        mean_corrected = torch.mean(corrected, dim=(0, 1, 2), keepdim=True)
        std_corrected = torch.std(corrected, dim=(0, 1, 2), keepdim=True)
        corrected = (corrected - mean_corrected) / (std_corrected + 1e-8)
        corrected = corrected * std_original + mean_original

        return corrected


# 图扩散卷积
class GraphDiffusionConv(nn.Module):
    def __init__(self, adj_matrix, input_dim, output_dim, bn_decay=0.1):
        super(GraphDiffusionConv, self).__init__()
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        self.register_buffer('adj_matrix', adj_matrix)
        self.num_nodes = adj_matrix.shape[0]
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        nn.init.xavier_uniform_(self.W)
        self.norm_adj = self.normalize_adjacency()
        self.bn = nn.BatchNorm2d(output_dim, momentum=bn_decay)

    def normalize_adjacency(self):
        adj = self.adj_matrix
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    def forward(self, X):
        batch_size, num_steps, num_nodes, feat_dim = X.shape
        if self.adj_matrix.device != X.device:
            self.adj_matrix = self.adj_matrix.to(X.device)
        X_reshaped = X.reshape(batch_size * num_steps, num_nodes, feat_dim)
        self.norm_adj = self.norm_adj.to(dtype=torch.float32)
        diffused = torch.einsum('ij,bjk->bik', self.norm_adj, X_reshaped)
        transformed = torch.einsum('bik,kl->bil', diffused, self.W)
        transformed = F.relu(transformed)
        transformed = transformed.reshape(batch_size, num_steps, num_nodes, -1)
        transformed = transformed.permute(0, 3, 1, 2)
        transformed = self.bn(transformed)
        transformed = transformed.permute(0, 2, 3, 1)

        return transformed


class spatialAttention(nn.Module):
    def __init__(self, K, d, args, drop=None):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.D = D
        self.args = args
        q_bn_decay = 0.1
        q_act ='relu'
        q_drop =0.01
        k_bn_decay = 0.1
        k_act = 'relu'
        k_drop =0.01
        v_bn_decay =0.1
        v_act ='relu'
        v_drop =0.01
        fc_bn_decay = 0.1
        fc_act = 'relu'
        fc_drop = 0.01
        self.FC_q = FC(input_dims=2 * D, units=D, activations=get_activation(q_act),
                       bn_decay=q_bn_decay, drop=q_drop)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=get_activation(k_act),
                       bn_decay=k_bn_decay, drop=k_drop)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=get_activation(v_act),
                       bn_decay=v_bn_decay, drop=v_drop)
        self.gated_fusion = GatedMultiHeadFusion(K, d, 0.1)
        self.FC = FC(input_dims=D, units=D, activations=get_activation(fc_act),
                     bn_decay=fc_bn_decay, drop=fc_drop)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        num_steps = X.shape[1]
        num_vertex = X.shape[2]
        X = torch.cat((X, STE), dim=-1)
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.view(self.K, batch_size, num_steps, num_vertex, self.d)
        X = X.permute(1, 2, 3, 0, 4).contiguous()
        X = X.view(batch_size, num_steps, num_vertex, self.K * self.d)
        X = self.gated_fusion(X)
        X = self.FC(X)
        return X


class AuxiliaryPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, args, drop=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.args = args
        aux_bn_decay =0.1
        aux_act1 ='relu'
        aux_act2 ='none'
        aux_drop =0.01
        main_bn_decay =0.1
        main_act1 = 'relu'
        main_act2 ='none'
        main_drop =0.01
        sim_bn_decay =0.1
        sim_act1 = 'relu'
        sim_act2 = 'sigmoid'
        sim_drop = 0.01
        attn_bn_decay = 0.1
        attn_act1 = 'relu'
        attn_act2 = 'none'
        attn_drop = 0.01
        fusion_bn_decay = 0.1
        fusion_act1 = 'relu'
        fusion_act2 = 'none'
        fusion_drop = 0.01
        self.aux_transform = FC(
            input_dims=[1, output_dim],
            units=[output_dim, output_dim],
            activations=[get_activation(aux_act1), get_activation(aux_act2)],
            bn_decay=aux_bn_decay,
            drop=aux_drop
        )
        self.main_transform = FC(
            input_dims=[input_dim, output_dim],
            units=[output_dim, output_dim],
            activations=[get_activation(main_act1), get_activation(main_act2)],
            bn_decay=main_bn_decay,
            drop=main_drop
        )
        self.sim_weight_net = FC(
            input_dims=[1, 16],
            units=[16, 1],
            activations=[get_activation(sim_act1), get_activation(sim_act2)],
            bn_decay=sim_bn_decay,
            drop=sim_drop
        )
        self.attention = FC(
            input_dims=[output_dim * 2, output_dim],
            units=[output_dim, 1],
            activations=[get_activation(attn_act1), get_activation(attn_act2)],
            bn_decay=attn_bn_decay,
            drop=attn_drop
        )
        self.fusion = FC(
            input_dims=[output_dim * 2, output_dim],
            units=[output_dim, output_dim],
            activations=[get_activation(fusion_act1), get_activation(fusion_act2)],
            bn_decay=fusion_bn_decay,
            drop=fusion_drop
        )
        self.bn = nn.BatchNorm2d(output_dim, momentum=0.1)

    def forward(self, main_features, auxiliary_info, similarity_scores):

        batch_size, num_steps, num_nodes, feat_dim = main_features.shape
        if auxiliary_info.dim() == 3:
            auxiliary_info = auxiliary_info.unsqueeze(-1)
        transformed_aux = self.aux_transform(auxiliary_info)
        transformed_main = self.main_transform(main_features)
        sim_weights = self.sim_weight_net(similarity_scores.view(batch_size, 1, 1, 1))
        weighted_aux = transformed_aux * sim_weights

        if weighted_aux.shape[1] < num_steps:
            padding = torch.zeros(batch_size, num_steps - weighted_aux.shape[1],
                                  num_nodes, self.output_dim, device=main_features.device)
            weighted_aux = torch.cat([weighted_aux, padding], dim=1)
        elif weighted_aux.shape[1] > num_steps:
            weighted_aux = weighted_aux[:, :num_steps]

        combined = torch.cat([transformed_main, weighted_aux], dim=-1)
        attention_scores = self.attention(combined)
        attention_weights = torch.sigmoid(attention_scores)
        attended_aux = weighted_aux * attention_weights
        fused = torch.cat([transformed_main, attended_aux], dim=-1)
        enhanced_features = self.fusion(fused)
        enhanced_features = enhanced_features.permute(0, 3, 1, 2)
        enhanced_features = self.bn(enhanced_features)
        enhanced_features = enhanced_features.permute(0, 2, 3, 1)

        return enhanced_features

class TemporalAware(nn.Module):
    def __init__(self, input_dim, output_dim, bn_decay=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.glu_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GLU(dim=-1)
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.residual_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.bn = nn.BatchNorm2d(output_dim, momentum=bn_decay)

    def forward(self, X):
        batch_size, num_steps, num_vertex, _ = X.shape
        glu_out = self.glu_layer(X)  # [batch, time, nodes, output_dim]
        position = torch.arange(num_steps, device=X.device).float()
        position = position.view(1, num_steps, 1, 1).repeat(batch_size, 1, num_vertex, 1)
        position_aware = glu_out + position
        mlp_out = self.position_mlp(position_aware)
        if self.input_dim == self.output_dim:
            output = X + self.residual_scale * mlp_out
        else:
            output = mlp_out
        output = output.permute(0, 3, 1, 2)
        output = self.bn(output)
        return output.permute(0, 2, 3, 1)  


class TemporalAttention(nn.Module):
    def __init__(self, input_dim, output_dim, args, drop=None):
        super(TemporalAttention, self).__init__()

        q_bn_decay = 0.1
        q_act ='relu'
        q_drop =0.01
        k_bn_decay = 0.1
        k_act = 'relu'
        k_drop = 0.01
        v_bn_decay = 0.1
        v_act = 'relu'
        v_drop =0.01
        fusion_bn_decay = 0.1
        fusion_act = 'relu'
        fusion_drop = 0.01
        self.W_q = FC(input_dims=input_dim * 2, units=output_dim, activations=get_activation(q_act),
                      bn_decay=q_bn_decay, drop=q_drop)
        self.W_k = FC(input_dims=input_dim * 2, units=output_dim, activations=get_activation(k_act),
                      bn_decay=k_bn_decay, drop=k_drop)
        self.W_v = FC(input_dims=input_dim * 2, units=output_dim, activations=get_activation(v_act),
                      bn_decay=v_bn_decay, drop=v_drop)
        self.fusion = FC(input_dims=output_dim * 2, units=output_dim, activations=get_activation(fusion_act),
                         bn_decay=fusion_bn_decay, drop=fusion_drop)

    def forward(self, X, STE, mask_future=True):

        batch_size, num_steps, num_nodes, feat_dim = X.shape
        original_X = X
        X = torch.cat((X, STE), dim=-1)
        Q = self.W_q(X)  
        K = self.W_k(X)  
        V = self.W_v(X)
        attn_scores = torch.einsum('btnd,bknd->btnk', Q, K) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        if mask_future:
            mask = torch.tril(torch.ones(num_steps, num_steps, device=X.device), diagonal=0)
            mask = mask.view(1, num_steps, 1, num_steps)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.einsum('btnk,bknd->btnd', attn_weights, V)
        fused = self.fusion(torch.cat([original_X, context], dim=-1))

        return fused


class NavAttention(nn.Module):
    def __init__(self, output_dim, args, drop=None):
        super(NavAttention, self).__init__()
        self.output_dim = output_dim
        self.args = args

        q_bn_decay = 0.1
        q_act = 'relu'
        q_drop = 0.01
        k_bn_decay = 0.1
        k_act = 'relu'
        k_drop = 0.01
        v_bn_decay = 0.1
        v_act = 'relu'
        v_drop =0.1
        map_bn_decay = 0.1
        map_act = 'relu'
        map_drop =0.1
        fusion_bn_decay =0.1
        fusion_act = 'relu'
        fusion_drop =0.01
        self.W_q = FC(input_dims=output_dim, units=output_dim, activations=get_activation(q_act),
                      bn_decay=q_bn_decay, drop=q_drop)
        self.W_k = FC(input_dims=output_dim, units=output_dim, activations=get_activation(k_act),
                      bn_decay=k_bn_decay, drop=k_drop)
        self.W_v = FC(input_dims=output_dim, units=output_dim, activations=get_activation(v_act),
                      bn_decay=v_bn_decay, drop=v_drop)
        self.mapping = FC(input_dims=self.output_dim, units=self.output_dim,
                          activations=get_activation(map_act), bn_decay=map_bn_decay, drop=map_drop)
        self.fusion = FC(input_dims=output_dim * 2, units=output_dim,
                         activations=get_activation(fusion_act), bn_decay=fusion_bn_decay, drop=fusion_drop)

    def forward(self, traffic, nav):

        Q = self.W_q(nav) 
        K = self.W_k(traffic)
        V = self.W_v(traffic)

        attn_scores = torch.einsum('btrd,btnd->btrn', Q, K) / torch.sqrt(
            torch.tensor(self.output_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.einsum('btrn,btnd->btrd', attn_weights, V)
        mapped = self.mapping(attended) 
        updated_traffic = traffic + mapped.mean(dim=2, keepdim=True)

        return updated_traffic



class STBlock(nn.Module):
    def __init__(self, adj_matrix, input_dim, output_dim, args, bn_decay=0.1, drop=0.1):
        super(STBlock, self).__init__()
        self.args = args
        self.spatialAttention = spatialAttention(args.K, args.d, args, drop)
        self.graph_conv = GraphDiffusionConv(adj_matrix, input_dim, output_dim, bn_decay)
        self.temporal_attn = TemporalAttention(input_dim, output_dim, args, drop)
        self.time_series_pred = TemporalAware(input_dim, output_dim, bn_decay=bn_decay)
        self.spatial_fusion = FC(
            input_dims=output_dim*2,
            units=output_dim,
            activations=get_activation('relu'),
            bn_decay=0.1,
            drop=0.01
        )
        self.temporal_fusion = FC(
            input_dims=output_dim*2,
            units=output_dim,
            activations=get_activation('relu'),
            bn_decay=0.1,
            drop=0.01
        )
        self.final_fusion = FC(
            input_dims=output_dim * 3,
            units=output_dim,
            activations=get_activation('relu'),
            bn_decay=0.1,
            drop=0.01
        )

        self.res_gate = FC(
            input_dims=[output_dim, output_dim],
            units=[output_dim, 1],
            activations=[get_activation('relu'), get_activation('none')],
            bn_decay=0.1,
            drop=0.01
        )
        self.distribution_correction = nn.LayerNorm(output_dim)
        self.aux_predictor = AuxiliaryPredictor(input_dim, output_dim, args, drop)

    def forward(self, X, STE, auxiliary_info=None, similarity_scores=None, is_encoder=True):

        aux_output = self.aux_predictor(X, auxiliary_info, similarity_scores)
        spatial_attention = self.spatialAttention(X, STE)
        graph_output = self.graph_conv(X)
        spatial_output = self.spatial_fusion(torch.cat([spatial_attention,graph_output], dim=-1))
        attn_output = self.temporal_attn(X, STE, mask_future=is_encoder)
        pred_output = self.time_series_pred(X)
        temporal_output = self.temporal_fusion(torch.cat([attn_output,pred_output], dim=-1))
        st_output = self.final_fusion(torch.cat([spatial_output, temporal_output,aux_output], dim=-1))
        st_output = self.distribution_correction(st_output)
        gate_weight = self.res_gate(st_output)
        output = gate_weight * X + (1 - gate_weight) * st_output
        return output


# 转换注意力
class TransformAttention(nn.Module):
    def __init__(self, K, d, args):
        super(TransformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.args = args
        q_bn_decay = 0.1
        q_act = 'relu'
        q_drop =0.1
        k_bn_decay =0.1
        k_act ='relu'
        k_drop = 0.1
        v_bn_decay = 0.1
        v_act = 'relu'
        v_drop = 0.1
        fc_bn_decay =0.1
        fc_act ='relu'
        fc_drop =0.1
        self.FC_q = FC(input_dims=D, units=D, activations=get_activation(q_act),
                       bn_decay=q_bn_decay, drop=q_drop)
        self.FC_k = FC(input_dims=D, units=D, activations=get_activation(k_act),
                       bn_decay=k_bn_decay, drop=k_drop)
        self.FC_v = FC(input_dims=D, units=D, activations=get_activation(v_act),
                       bn_decay=v_bn_decay, drop=v_drop)
        self.FC = FC(input_dims=D, units=D, activations=get_activation(fc_act),
                     bn_decay=fc_bn_decay, drop=fc_drop)

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class HQEST(nn.Module):
    def __init__(self, SE, args, adj_matrix, bn_decay=0.1, drop=0.1):
        super(HQEST, self).__init__()
        self.args = args
        self.register_buffer('SE', SE)

        input_dim = 1
        output_dim = args.d * args.K

        self.fourier_filter = FourierFilter(args)

        self.traffic_linear = FC(
            input_dims=[input_dim, output_dim],
            units=[output_dim, output_dim],
            activations=[get_activation('relu'), get_activation('none')],
            bn_decay=0.1,
            drop=0.01
        )

        self.query_linear = FC(
            input_dims=[2, output_dim],
            units=[output_dim, output_dim],
            activations=[get_activation('relu'), get_activation('none')],
            bn_decay=0.1,
            drop=0.01
        )
        self.nav_attention = NavAttention(output_dim, args)
        self.st_embedding = STEmbedding(output_dim, args, drop)
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        if SE.device != adj_matrix.device:
            adj_matrix = adj_matrix.to(SE.device)

        self.encoder_blocks = nn.ModuleList([
            STBlock(adj_matrix, output_dim, output_dim, self.args, bn_decay=bn_decay, drop=drop)
            for _ in range(args.L)
        ])

        self.transform_attn = TransformAttention(args.K, args.d, args)

        self.decoder_blocks = nn.ModuleList([
            STBlock(adj_matrix, output_dim, output_dim, self.args, bn_decay=bn_decay, drop=drop)
            for _ in range(args.L)
        ])

        self.output_layer = FC(
            input_dims=[output_dim, output_dim],
            units=[output_dim, 1],
            activations=[get_activation('relu'), get_activation('none')],
            bn_decay=0.1,
            drop=0.01
        )

    def forward(self, X, TE, query_data, auxiliary_info_his=None, auxiliary_info_pred=None, similarity_scores=None,
                is_training=None):

        X = torch.unsqueeze(X, -1)

        filtered_X = self.fourier_filter(X)
        traffic_transformed = self.traffic_linear(filtered_X)
        query = query_data.reshape(query_data.shape[0], query_data.shape[1], -1, query_data.shape[4])
        query_transformed = self.query_linear(query)

        updated_traffic = self.nav_attention(traffic_transformed, query_transformed)

        STE = self.st_embedding(self.SE, TE)
        STE_his = STE[:, :self.args.num_his]
        STE_pred = STE[:, self.args.num_his:]

        encoder_output = updated_traffic
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output, STE_his,
                                   auxiliary_info_his,
                                   similarity_scores,
                                   is_encoder=True)

        transformed = self.transform_attn(encoder_output, STE_his, STE_pred)

        decoder_output = transformed
        for block in self.decoder_blocks:
            decoder_output = block(decoder_output, STE_pred,
                                   auxiliary_info_his,
                                   similarity_scores,
                                   is_encoder=False)

        output = self.output_layer(decoder_output)
        return output.squeeze(-1)