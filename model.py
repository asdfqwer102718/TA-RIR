import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool as gap, global_mean_pool as gmp
from utils.FiLM import FiLM
import torch.nn.functional as F

from utils.audio import (
    get_octave_filters,
)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, use_layernorm=True
                 , sequence_length=None):
        """

        :param in_channels:
        :param out_channels:
        :param use_batchnorm:
        :param use_layernorm: 在use_batchnorm为True的情况下，若为True则将batchnorm替换为layernorm
        """
        super(EncoderBlock, self).__init__()
        if use_batchnorm:
            if use_layernorm:
                self.norm_layer = nn.LayerNorm([out_channels, sequence_length])
            else:
                self.norm_layer = nn.BatchNorm1d(out_channels, track_running_stats=True)
        else:
            self.norm_layer = nn.Identity()     # 恒等映射

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
            self.norm_layer,
            nn.PReLU(),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            self.norm_layer,
        )

    def forward(self, x):
        out = self.conv(x)
        skip_out = self.skip_conv(x)
        skip_out = out + skip_out
        return skip_out


class Encoder(nn.Module):
    """
    使用Encoder生成512维的向量，然后fc成128维的z（z=128）
    """

    def __init__(self, use_layer_norm):
        """

        :param use_layer_norm: 是否用LN代替BN
        """
        super(Encoder, self).__init__()
        block_list = []
        channels = [1, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512]

        # 初始序列长度
        sequence_length = 131072
        for i in range(0, len(channels) - 1):
            if (i + 1) % 3 == 0:
                use_batchnorm = True
            else:
                use_batchnorm = False
            in_channels = channels[i]
            out_channels = channels[i + 1]
            # 计算当前块的输出序列长度
            sequence_length //= 2
            curr_block = EncoderBlock(in_channels, out_channels,
                use_batchnorm, use_layer_norm, sequence_length=sequence_length)
            block_list.append(curr_block)

        self.encode = nn.Sequential(*block_list)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        b, c, l = x.size()
        out = self.encode(x)
        out = self.pooling(out)
        out = out.view(b, -1)
        out = self.fc(out)
        return out


class UpsampleNet(nn.Module):
    def __init__(self, input_size, output_size, upsample_factor):
        super(UpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer = nn.ConvTranspose1d(
            input_size,
            output_size,
            upsample_factor * 2,
            upsample_factor,
            padding=upsample_factor // 2,
            )
        nn.init.orthogonal_(layer.weight)
        self.layer = spectral_norm(layer)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs


class ConditionalBatchNorm1d(nn.Module):
    """Conditional Batch Normalization"""

    def __init__(self, num_features, condition_length):
        super().__init__()

        self.num_features = num_features
        self.condition_length = condition_length
        self.norm = nn.BatchNorm1d(num_features, affine=True, track_running_stats=True)

        self.layer = spectral_norm(nn.Linear(condition_length, num_features * 2))
        self.layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.layer.bias.data.zero_()  # Initialise bias at 0

    def forward(self, inputs, noise):
        outputs = self.norm(inputs)
        gamma, beta = self.layer(noise).chunk(2, 1)
        gamma = gamma.view(-1, self.num_features, 1)
        beta = beta.view(-1, self.num_features, 1)

        outputs = gamma * outputs + beta

        return outputs


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor, condition_length):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_length = condition_length
        self.upsample_factor = upsample_factor

        # Block A
        self.film1 = FiLM(in_channels, condition_length)

        self.first_stack = nn.Sequential(
            nn.PReLU(),
            UpsampleNet(in_channels, in_channels, upsample_factor),
            nn.Conv1d(in_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.film2 = FiLM(out_channels, condition_length)

        self.second_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.residual1 = nn.Sequential(
            UpsampleNet(in_channels, in_channels, upsample_factor),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
        )
        # Block B
        self.film3 = FiLM(out_channels, condition_length)

        self.third_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=4, padding=28),
        )

        self.film4 = FiLM(out_channels, condition_length)

        self.fourth_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=8, padding=56),
        )

    def forward(self, enc_out, condition):
        inputs = enc_out

        outputs = self.film1(inputs, condition)
        outputs = self.first_stack(outputs)
        outputs = self.film2(outputs, condition)
        outputs = self.second_stack(outputs)

        residual_outputs = self.residual1(inputs) + outputs

        # Decoder Blcok B
        outputs = self.film3(residual_outputs, condition)
        outputs = self.third_stack(outputs)
        outputs = self.film4(outputs, condition)
        outputs = self.fourth_stack(outputs)

        outputs = outputs + residual_outputs

        return outputs


class Decoder(nn.Module):
    def __init__(self, num_filters, cond_length):
        super(Decoder, self).__init__()

        self.preprocess = nn.Conv1d(1, 512, kernel_size=15, padding=7)
        self.blocks = nn.ModuleList(
            [
                # cond_length是noise_condition_length + z_size = 16 + 128 = 144
                DecoderBlock(512, 512, 1, cond_length),
                DecoderBlock(512, 256, 2, cond_length),
                DecoderBlock(256, 256, 3, cond_length),
                DecoderBlock(256, 128, 4, cond_length),
                DecoderBlock(128, 64, 5, cond_length),
            ]
        )

        self.postprocess = nn.Sequential(nn.Conv1d(64, num_filters + 1, kernel_size=15, padding=7))

        self.sigmoid = nn.Sigmoid()

    def forward(self, v, condition):
        inputs = self.preprocess(v)
        outputs = inputs
        for i, layer in enumerate(self.blocks):
            outputs = layer(outputs, condition)
        outputs = self.postprocess(outputs)

        direct_early = outputs[:, 0:1]
        late = outputs[:, 1:]
        late = self.sigmoid(late)

        return direct_early, late


class TARIR_main(nn.Module):
    def __init__(self, config):
        super(TARIR_main, self).__init__()
        if config.room_info_only:
            config.graph_module.use_graph_module = False
        self.config = config

        self.rir_length = int(self.config.rir_duration * self.config.sr)
        self.min_snr, self.max_snr = config.min_snr, config.max_snr

        # Learned decoder input
        self.decoder_input = nn.Parameter(torch.randn((1, 1, config.decoder_input_length)))  # 1,1,400
        self.condition_length = config.noise_condition_length
        if config.reverberation_module.use_reverberation_module:
            self.condition_length += config.z_size
            self.encoder = Encoder(config.use_layer_norm)
        if config.room_info_only:
            self.condition_length += config.room_info_len_after_padding
        if not config.room_info_only and config.graph_module.use_graph_module:
            self.condition_length += config.graph_module.graph_embedding_length
            self.gcn = TopologyEncoder(config.graph_module.graph_embedding_length)
        if config.reverberation_module.use_reverberation_module:
            self.condition_length = config.noise_condition_length + config.z_size
            if config.graph_module.use_graph_module:
                self.alpha = nn.Parameter(torch.tensor(1.0))  # Initialize scaling factor
                self.fusion_layer = nn.Linear(config.z_size + config.graph_module.graph_embedding_length, config.z_size)
            if config.room_info_only:
                self.fusion_layer = nn.Linear(config.z_size + config.room_info_len_after_padding, config.z_size)
            if config.graph_module.use_graph_module or config.room_info_only:
                self.condition_prelu = nn.PReLU()    # 用于对condition进行PReLu处理


        self.decoder = Decoder(config.num_filters, self.condition_length)

        # Learned "octave-band" like filter
        self.filter = nn.Conv1d(
            config.num_filters,
            config.num_filters,
            kernel_size=config.filter_order,
            stride=1,
            padding='same',
            groups=config.num_filters,
            bias=False,
        )

        # Octave band pass initialization
        octave_filters = get_octave_filters()
        self.filter.weight.data = torch.FloatTensor(octave_filters)

        # self.filter.bias.data.zero_()

        # Mask for direct and early part
        mask = torch.zeros((1, 1, self.rir_length))
        mask[:, :, : self.config.early_length] = 1.0
        self.register_buffer("mask", mask)
        self.output_conv = nn.Conv1d(config.num_filters + 1, 1, kernel_size=1, stride=1)

    def forward(self, x, stochastic_noise, noise_condition, graph, room_info_vector):
        """
        args:
            x : Reverberant speech. shape=(batch_size, 1, input_samples)
            stochastic_noise : Random normal noise for late reverb synthesis. shape=(batch_size, n_freq_bands, length_of_rir)
            noise_condition : Noise used for conditioning. shape=(batch_size, noise_cond_length)
            graph: 图
            room_info_vector: 房间信息直接拼接成的向量
        return
            rir: shape=(batch_size, 1, rir_samples)
            condition: 输入decoder的condition，用于绘图
        """
        b, _, _ = x.size()      # 这个x的维度是(batch_size, 1, 131072)

        # Filter random noise signal
        filtered_noise = self.filter(stochastic_noise)

        # Encode the reverberated speech
        if self.config.reverberation_module.use_reverberation_module:
            z = self.encoder(x)

        # 生成图embedding [batch_size, 32]
        if self.config.graph_module.use_graph_module:
            graph_embedding = self.gcn(graph)

        if self.config.graph_module.use_graph_module and self.config.reverberation_module.use_reverberation_module:
            scale_z = z.norm(p=2, dim=1).mean()
            bias_z = z.mean(dim=1, keepdim=True)

            scale_graph_embedding = graph_embedding.norm(p=2, dim=1).mean()
            bias_graph_embedding = graph_embedding.mean(dim=1, keepdim=True)

            # Adjust e_B to match z
            graph_embedding_adjusted = (graph_embedding - bias_graph_embedding) * (scale_z / scale_graph_embedding) + bias_z

            # Apply learnable scaling factor
            graph_embedding_scaled = self.alpha * graph_embedding_adjusted

            # Make condition vector
            condition = torch.cat([z, graph_embedding_scaled], dim=-1)
            condition = self.fusion_layer(condition)
            condition = self.condition_prelu(condition)
            condition = torch.cat([condition, noise_condition], dim=-1)
        elif self.config.room_info_only and self.config.reverberation_module.use_reverberation_module:
            condition = torch.cat([z, room_info_vector], dim=-1)
            condition = self.fusion_layer(condition)
            condition = self.condition_prelu(condition)
            condition = torch.cat([condition, noise_condition], dim=-1)
        elif self.config.graph_module.use_graph_module:
            condition = torch.cat([graph_embedding, noise_condition], dim=-1)
        else:
            condition = torch.cat([z, noise_condition], dim=-1)

        # Learnable decoder input. Repeat it in the batch dimension.
        decoder_input = self.decoder_input.repeat(b, 1, 1)

        # Generate RIR
        direct_early, late_mask = self.decoder(decoder_input, condition)

        # Apply mask to the filtered noise to get the late part
        late_part = filtered_noise * late_mask

        # Zero out sample beyond 2400 for direct early part
        direct_early = torch.mul(direct_early, self.mask)
        # Concat direct,early with late and perform convolution
        rir = torch.cat((direct_early, late_part), 1)

        # Sum
        rir = self.output_conv(rir)

        return rir, condition


class TopologyEncoder(nn.Module):
    def __init__(self, config):
        super(TopologyEncoder, self).__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 32)
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, config)

    def forward(self, graph: Data):
        batch = graph.batch  
        x = self.conv1(graph.x, graph.edge_index)
        x = F.relu(x)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.conv2(x, graph.edge_index)
        x = F.relu(x)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.linear1(x1 + x2)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x).squeeze(1)
        return x
