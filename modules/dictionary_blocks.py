import torch
from einops import rearrange
from torch import nn


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x):
        x = rearrange(x, "b h w c -> b c h w")
        x = self.dwconv(x)
        x = rearrange(x, "b c h w -> b h w c")

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = hidden_features // 2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x)) * v
        x = self.fc2(x)
        return x


class ConvWithDW(nn.Module):
    def __init__(self, input_dim=320, output_dim=320):
        super(ConvWithDW, self).__init__()
        self.in_trans = nn.Conv2d(
            input_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True
        )
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(
            output_dim,
            output_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=output_dim,
            bias=True,
        )
        self.act2 = nn.GELU()
        self.out_trans = nn.Conv2d(
            output_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True
        )

    def forward(self, x):
        x = self.in_trans(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        x = self.out_trans(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, dim=320):
        super(DenseBlock, self).__init__()
        self.layer_num = 3
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GELU(),
                    ConvWithDW(dim, dim),
                )
                for i in range(self.layer_num)
            ]
        )
        self.proj = nn.Conv2d(
            dim * (self.layer_num + 1),
            dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
        )

    def forward(self, x):
        outputs = [x]
        for i in range(self.layer_num):
            outputs.append(self.conv_layers[i](outputs[-1]))
        x = self.proj(torch.cat(outputs, dim=1))
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAggregation, self).__init__()
        self.s = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = DenseBlock(dim)

    def forward(self, x):
        x = rearrange(x, "b h w c -> b c h w")
        s = self.s(x)
        s_out = self.dense(s)
        x = s_out * self.spatial_atte(s_out)
        x = rearrange(x, "b c h w -> b h w c")
        return x


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class MultiScaleDictionaryCrossAttentionGLU(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=20, qkv_bias=True):
        super().__init__()

        dict_dim = 32 * head_num
        self.head_num = head_num

        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)

        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)

        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)

        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)

        self.mlp = ConvolutionalGLU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Sequential(nn.Linear(dict_dim, output_dim))
        self.softmax = torch.nn.Softmax(dim=-1)

        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

    def forward(self, x, dt):
        B, C, H, W = x.size()
        x = rearrange(x, "b c h w -> b h w c")
        x = self.x_trans(x)

        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)

        shortcut = x
        x = self.lnx(x)
        x = self.q_trans(x)
        x = rearrange(x, "b h w c -> b c h w")

        q = rearrange(x, "b (e c) h w -> b e (h w) c", e=self.head_num)
        dt = self.dict_ln(dt)
        k = self.k(dt)
        k = rearrange(k, "b n (e c) -> b e n c", e=self.head_num)
        dt = rearrange(dt, "b n (e c) -> b e n c", e=self.head_num)
        sim = torch.einsum("benc,bedc->bend", q, k)
        sim = sim * self.scale
        probs = self.softmax(sim)
        output = torch.einsum("bend,bedc->benc", probs, dt)
        output = rearrange(output, "b e (h w) c -> b h w (e c) ", h=H, w=W)

        output = self.linear(output) + self.res_scale_2(shortcut)

        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)

        output = self.output_trans(output)
        output = rearrange(
            output,
            "b h w c -> b c h w",
        )
        return output
