import torch
import torch.nn as nn
from torch import nn, einsum
from .ProFound_utils import *
from .build import MODELS

# Feature Extractor
class VectorFusionAttention(nn.Module):
    def __init__(self, in_channel = 128, dim = 64, n_knn = 16, attn_hidden_multiplier = 4):
        super(VectorFusionAttention, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )
        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, query, support):
        pq, fq = query
        ps, fs = support

        identity = fq 
        query, key, value = self.conv_query(fq), self.conv_key(fs), self.conv_value(fs) 
        
        B, D, N = query.shape

        pos_flipped_1 = ps.permute(0, 2, 1).contiguous() 
        pos_flipped_2 = pq.permute(0, 2, 1).contiguous() 
        idx_knn = query_knn(self.n_knn, pos_flipped_1, pos_flipped_2)

        key = grouping_operation(key, idx_knn) 
        qk_rel = query.reshape((B, -1, N, 1)) - key  

        pos_rel = pq.reshape((B, -1, N, 1)) - grouping_operation(ps, idx_knn)  
        pos_embedding = self.pos_mlp(pos_rel) 

        attention = self.attn_mlp(qk_rel + pos_embedding) 
        attention = torch.softmax(attention, -1)

        value = grouping_operation(value, idx_knn) + pos_embedding  
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  
        output = self.conv_end(agg) + identity
        
        return output

class HCFF(nn.Module):
    def __init__(self, dim_in = [512, 256, 128], is_inter = True, down_rates = [1, 4, 2], knns = [16, 12, 8]):
        super(HCFF, self).__init__()
        self.down_rates = down_rates
        self.is_inter = is_inter
        self.num_scale = len(down_rates)

        self.attn_lists = nn.ModuleList()
        self.q_mlp_lists = nn.ModuleList()
        self.s_mlp_lists = nn.ModuleList()
        for i in range(self.num_scale):
            self.attn_lists.append(VectorFusionAttention(in_channel = dim_in[i], dim = 64, n_knn = knns[i]))

        for i in range(self.num_scale - 1):
            self.q_mlp_lists.append(MLP_Res(in_dim = 128*2, hidden_dim = 128, out_dim = 128))
            self.s_mlp_lists.append(MLP_Res(in_dim = 128*2, hidden_dim = 128, out_dim = 128))

    def forward(self, point, graph):
        pp, pf = point
        gp, gf = graph
        
        pre_pos = []
        pre_f = []
        
        for i in range(self.num_scale - 1, -1, -1):
            _pos1, _f1, _pos2, _f2 = pp[i], pf[i], gp[i], gf[i]
            f = self.attn_lists[i]([_pos1, _f1], [_pos2, _f2])
            pre_pos.append(_pos1)
            pre_f.append(f)

        return pre_pos, pre_f

class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024, n_knn=20):
        """
            Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(256, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_3 = PointNet_SA_Module_KNN(128, 16, 256, [256, 512], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_4 = PointNet_SA_Module_KNN(None, None, 512, [512, out_dim], group_all=True, if_bn=False)

        self.gcn_1 = EdgeConv(3, 128, 16)
        self.gcn_2 = EdgeConv(128, 256, 8)
        self.gcn_3 = EdgeConv(256, 512, 4)

        self.hcff = HCFF(dim_in = [128, 256, 512], is_inter = False, down_rates = [1, 2, 2], knns = [16, 12, 8])

    def forward(self, xyz):
        """
        Args:
             xyz: b, 3, n
        Returns:
            l3_points: (B, out_dim, 1)
        """
        # PointNet++ feature extraction
        P_P = []
        F_P = []
        l_xyz, l_points = xyz, xyz
        sa_modules = [self.sa_module_1, self.sa_module_2, self.sa_module_3]

        for sa_module in sa_modules:
            l_xyz, l_points, _ = sa_module(l_xyz, l_points)
            P_P.append(l_xyz)
            F_P.append(l_points)

        _, point_feat = self.sa_module_4(P_P[-1], F_P[-1])  # (B, 3, 1), (B, out_dim, 1)

        # GCN feature extraction
        P_G = []
        F_G = []
        x, p = xyz, xyz
        gcn_modules = [self.gcn_1, self.gcn_2, self.gcn_3]
        points = [512, 256, 128]

        for i, gcn_layer in enumerate(gcn_modules):
            x = gcn_layer(x)
            idx = pointnet2_utils.furthest_point_sample(p.permute(0, 2, 1).contiguous(), points[i])
            p = pointnet2_utils.gather_operation(p.contiguous(), idx)
            f = pointnet2_utils.gather_operation(x, idx)
            P_G.append(p)
            F_G.append(f)

        pos, feat = self.hcff([P_P, F_P], [P_G, F_G])

        return pos[0], feat[0], point_feat, feat[-1]


# Matrx Guided DenseAgg MultiRF Seed Generator Module
class MatrixGuided(nn.Module):
    def __init__(self, dim=512, hidd_dim=64, num_dicts=128, init_value=0.0):
        super(MatrixGuided, self).__init__()
        self.num_dicts = num_dicts
        self.matrix = nn.Parameter(torch.full((dim, num_dicts), init_value))
        self.query = nn.Conv1d(dim, hidd_dim, 1)
        self.key = nn.Conv1d(dim, hidd_dim, 1)
    
    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        B,C,N = feat.size()

        dict_fea = self.matrix
        dict_fea = dict_fea.unsqueeze(0).repeat(B,1,1)
        
        q = self.query(feat)  # (b, 64, N)
        value = dict_fea 
        k = self.key(value) # (1, 64, 128)
        
        d_k = k.size(1)

        # Attention
        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(d_k) # (b, N, 128)
        scores = torch.softmax(scores, dim=-1) # (b, N, 128)
        
        output = torch.matmul(scores, value.transpose(-2, -1)) # (b, N, dim)

        return (output.transpose(-2, -1)+feat)/2

class Mini_PN(nn.Module):
    def __init__(self, feat_dim = 512):
        super(Mini_PN, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])

    def forward(self, xyz):
        feat = self.mlp_1(xyz)
        global_feat = torch.max(feat, 2, keepdim=False)[0]  # (b, 512)
        global_feat = global_feat.unsqueeze(-1)  # (b, 512, 1)
        # print(global_feat.shape)
        # sys.exit()
        return global_feat

class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.mini_pn = Mini_PN(feat_dim = dim_feat)
        self.uptrans = UpTransformer(512, 128, dim=64, n_knn=20, use_upfeat=False, attn_channel=True, up_factor=2, scale_layer=None)
        self.refine = MatrixGuided(dim=640,hidd_dim=256,num_dicts=256) 
        self.mlp_1 = MLP_Res(in_dim=1024, hidden_dim=512, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128 *2, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, xyz, patch_xyz, patch_feat, point_feat):
        """
        Args:
            point_feat: Tensor (b, dim_feat, 1)
            completion: Tensor (b, 3, n)
        """
        # print(xyz.shape)
        boundary_points_batch = batch_process_point_clouds(xyz, 20, 75)
        boundary_points = torch.stack(boundary_points_batch, dim=0).permute(0, 2, 1).contiguous()
        
        # print(type(xyz),xyz.permute(0, 2, 1).contiguous().shape)
        # print(type(boundary_points),boundary_points.shape)

        # save_tensor_to_pcd(xyz.permute(0, 2, 1).contiguous(),'./point/source')
        # save_tensor_to_pcd(boundary_points,'./point/BP')
        # exit()
        boundary_feat = self.mini_pn(boundary_points)

        x1 = self.uptrans(patch_xyz, patch_feat, patch_feat, upfeat=None)
        x2 = self.mlp_2(x1)
        r1 = torch.cat([point_feat.repeat((1, 1, x1.size(2))), boundary_feat.repeat((1, 1, x1.size(2)))], 1)
        r2 = self.refine(r1)
        
        x3 = self.mlp_3(torch.cat([x2, r2], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)

        return completion, x3


# Point Refinement Module
class UpTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, dim, n_knn=20, up_factor=2, use_upfeat=True, 
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1

        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        # attention layers
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor,1), (up_factor,1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = nn.Upsample(scale_factor=(up_factor,1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # residual connection
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos, key, query, upfeat):
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        """

        value = self.mlp_v(torch.cat([key, query], 1)) # (B, dim, N)
        identity = value
        key = self.conv_key(key) # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        # upfeat embedding
        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat) # (B, dim, N)
            upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn) # (B, dim, N, k)
        else:
            upfeat_rel = torch.zeros_like(qk_rel)

        # attention
        attention = self.attn_mlp(qk_rel + pos_embedding + upfeat_rel) # (B, dim, N*up_factor, k)

        # softmax function
        attention = self.scale(attention)

        # knn value is correct
        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel # (B, dim, N, k)
        value = self.upsample1(value) # (B, dim, N*up_factor, k)

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)
        y = self.conv_end(agg) # (B, out_dim, N*up_factor)

        # shortcut
        identity = self.residual_layer(identity) # (B, out_dim, N)
        identity = self.upsample2(identity) # (B, out_dim, N*up_factor)

        return y+identity
     
class PointRefineLayer(nn.Module):
    def __init__(self, dim_feat=512,dim=128, up_factor=2, i=0, radius=1):
        """Snowflake Point Deconvolution"""
        super(PointRefineLayer, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim * 2 , layer_dims=[256, 128])

        self.uptrans1 = UpTransformer(dim, dim, dim=64, n_knn=20, use_upfeat=True, up_factor=None)
        self.uptrans2 = UpTransformer(dim, dim, dim=64, n_knn=20, use_upfeat=True, attn_channel=True, up_factor=self.up_factor)

        self.up_sampler = nn.Upsample(scale_factor=up_factor)

        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])


    def forward(self, pcd_prev, gcn_feat, seed, seed_feat, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            gcn_feat: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape

        # three interpolate
        idx, dis = get_nearest_index(pcd_prev, seed, k=3, return_dis=True) # (B, N_prev, 3), (B, N_prev, 3)
        dist_recip = 1.0 / (dis + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True) # (B, N_prev, 1)
        weight = dist_recip / norm # (B, N_prev, 3)
        feat_upsample = torch.sum(indexing_neighbor(seed_feat, idx) * weight.unsqueeze(1), dim=-1) # (B, seed_dim, N_prev)

        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            gcn_feat.repeat((1, 1, int((feat_1.size(2))/(gcn_feat.size(2))))),
                            feat_upsample], 1)

        Q = self.mlp_2(feat_1)
        # Upsample Transformers
        H = self.uptrans1(pcd_prev, K_prev if K_prev is not None else Q, Q, upfeat=feat_upsample) # (B, 128, N_prev)
        feat_child = self.uptrans2(pcd_prev, K_prev if K_prev is not None else H, H, upfeat=feat_upsample) # (B, 128, N_prev*up_factor)

        # Get current features K
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius ** self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta
        
        return pcd_child, K_curr

class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(PointRefineLayer(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, patch_xyz, patch_feat, point_feat, gcn_feat, partial, return_P0=False):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        seed, seed_feat = self.decoder_coarse(partial, patch_xyz, patch_feat, point_feat)  # (B, num_pc, 3)
        seed = seed.permute(0, 2, 1).contiguous() # (B, num_pc, 3)
        arr_pcd.append(seed)
        pcd = fps_subsample(torch.cat([seed, partial], 1), self.num_p0)
        if return_P0:
            arr_pcd.append(pcd)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        seed = seed.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, gcn_feat, seed, seed_feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
        
        return arr_pcd


@MODELS.register_module()
class ProFound(nn.Module):
    def __init__(self, config, **kwargs):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_pc: int
            num_p0: int
            radius: searching radius
            up_factors: list of int
        """
        super().__init__()
        dim_feat = config.dim_feat
        num_pc = config.num_pc
        num_p0 = config.num_p0
        radius = config.radius
        up_factors = config.up_factors

        self.feat_extractor = FeatureExtractor(out_dim=dim_feat)
        self.decoder = Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0, radius=radius, up_factors=up_factors)

    def forward(self, point_cloud,condition=None,return_P0=False):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        # print("point_cloud:",point_cloud.shape)
        pcd_bnc = point_cloud
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        patch_xyz, patch_feat, point_feat, gcn_feat = self.feat_extractor(point_cloud)
        # print("feat:",feat.shape)
        out = self.decoder(patch_xyz, patch_feat,point_feat, gcn_feat, pcd_bnc, return_P0=return_P0)
        if self.training:
            out = (*out, point_cloud.permute(0, 2, 1).contiguous())
        else:
            out = (point_cloud.permute(0, 2, 1).contiguous(), *out)    
        
        # print("out:",out[0].shape, out[1].shape, out[2].shape, out[3].shape, out[4].shape, out[5].shape)
        # sys.exit()
        return out

