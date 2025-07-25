import copy
import time
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"

import tensordict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse
import torchsparse.nn as tsnn
from genesis import gs
from graphs import GraphEncoder, GraphEncoderWithGlobalFeatures
from n_step_sampler import StepAndNextSampler
from singleton_graph_buffer import GraphBuffer
from sparse_voxel_buffer import SparseVoxelBuffer
from torch_geometric.data import Batch
from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    TensorDictReplayBuffer,
    TensorStorage,
)
from torchsparse_util import sparse_coo_to_torchsparse

from examples.box_to_pos_task import MoveBoxSetup
from examples.ten_holes_14 import TenHoles


class SACActor(nn.Module):
    """
    PyTorch implementation of SAC Actor network.
    """

    def __init__(
        self,
        action_dim,
        electronics_graph_in_dim,
        mechanics_graph_in_dim,
        electronics_graph_out_dim,
        mechanics_graph_out_dim,
        device=None,
        dtype=torch.bfloat16,
    ):
        super(SACActor, self).__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --- Voxel encoding helpers for torchsparse ---
        @torch._dynamo.disable
        def voxel_encode(x):
            x = self.voxel_conv1(x)
            x = self.voxel_act(x)
            x = self.voxel_bn1(x)
            x = self.voxel_conv2(x)
            x = self.voxel_act(x)
            x = self.voxel_bn2(x)
            x = self.voxel_conv3(x)
            x = self.voxel_act(x)
            x = self.voxel_bn3(x)
            x_dense = x.dense()
            return x_dense.view(x_dense.size(0), -1).to(torch.bfloat16)

        self._voxel_encode = voxel_encode
        # Voxel encoder (3D conv layers)
        self.voxel_conv1 = tsnn.Conv3d(1, 2, kernel_size=(6, 6, 6), stride=(4, 4, 4))
        self.voxel_bn1 = tsnn.BatchNorm(2)
        self.voxel_act = tsnn.SiLU()
        self.voxel_conv2 = tsnn.Conv3d(2, 4, kernel_size=(6, 6, 6), stride=(4, 4, 4))
        self.voxel_bn2 = tsnn.BatchNorm(4)
        self.voxel_conv3 = tsnn.Conv3d(4, 8, kernel_size=(6, 6, 6), stride=(4, 4, 4))
        self.voxel_bn3 = tsnn.BatchNorm(8)
        # Video encoder (2D conv layers)
        self.video1_conv1 = nn.Conv2d(
            7, 10, kernel_size=(6, 6), stride=(4, 4), dtype=torch.bfloat16
        )
        self.vid1_bn1 = nn.BatchNorm2d(10, dtype=torch.bfloat16)
        self.video1_conv2 = nn.Conv2d(
            10, 12, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16
        )
        self.vid1_bn2 = nn.BatchNorm2d(12, dtype=torch.bfloat16)
        self.video1_conv3 = nn.Conv2d(
            12, 14, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16
        )
        self.vid1_bn3 = nn.BatchNorm2d(14, dtype=torch.bfloat16)

        self.video2_conv1 = nn.Conv2d(
            7, 10, kernel_size=(6, 6), stride=(4, 4), dtype=torch.bfloat16
        )
        self.vid2_bn1 = nn.BatchNorm2d(10, dtype=torch.bfloat16)
        self.video2_conv2 = nn.Conv2d(
            10, 12, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16
        )
        self.vid2_bn2 = nn.BatchNorm2d(12, dtype=torch.bfloat16)
        self.video2_conv3 = nn.Conv2d(
            12, 14, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16
        )
        self.vid2_bn3 = nn.BatchNorm2d(14, dtype=torch.bfloat16)

        # graph
        assert mechanics_graph_out_dim % 2 == 0, (
            "mechanics_graph_out_dim must be even for global features"
        )
        self.mech_graph_encoder = GraphEncoderWithGlobalFeatures(
            num_features_graph=mechanics_graph_in_dim,
            hidden_dim_graph=256,
            out_dim_graph=mechanics_graph_out_dim // 2,
            global_embedding_in_dim=8,
            hidden_dim_global=256,
            out_dim_global=mechanics_graph_out_dim // 2,
            heads=2,
            dtype=torch.bfloat16,
        )
        self.elec_graph_encoder = GraphEncoder(
            num_features=electronics_graph_in_dim,
            hidden_dim=256,
            out_dim=electronics_graph_out_dim,
            heads=2,
            dtype=torch.bfloat16,
        )

        # combined
        compressed_video_shape = 350
        compressed_voxel_shape = 216
        combined_dim = (
            compressed_video_shape * 2
            + compressed_voxel_shape * 2
            + electronics_graph_out_dim * 2
            + mechanics_graph_out_dim * 2
        )
        self.combine_bn1 = nn.BatchNorm1d(combined_dim, dtype=torch.bfloat16)
        self.fc1 = nn.Linear(combined_dim, 256, dtype=torch.bfloat16)
        self.combine_bn2 = nn.BatchNorm1d(256, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(256, 256, dtype=torch.bfloat16)
        self.combine_bn3 = nn.BatchNorm1d(256, dtype=torch.bfloat16)
        self.out_mean = nn.Linear(256, action_dim, dtype=torch.bfloat16)
        self.out_log_std = nn.Linear(256, action_dim, dtype=torch.bfloat16)

    def forward(
        self,
        voxel_init_obs,
        voxel_des_obs,
        video_obs,
        mech_graph_init_obs: Batch,
        mech_graph_des_obs: Batch,
        elec_graph_init_obs: Batch,
        elec_graph_des_obs: Batch,
    ):
        # TODO add support for mech graphs.

        # assume voxel_init_obs shape [B, D, H, W]
        x_vox_i = self._voxel_encode(voxel_init_obs)
        x_vox_d = self._voxel_encode(voxel_des_obs)

        # assume video_obs shape [B, C, H, W]
        vid_1 = video_obs[:, 0]
        x_vid1 = F.silu(self.video1_conv1(vid_1.to(torch.bfloat16) / 255))
        x_vid1 = self.vid1_bn1(x_vid1)
        x_vid1 = F.silu(self.video1_conv2(x_vid1))
        x_vid1 = self.vid1_bn2(x_vid1)
        x_vid1 = F.silu(self.video1_conv3(x_vid1))
        x_vid1 = self.vid1_bn3(x_vid1)
        x_vid1 = x_vid1.reshape(x_vid1.size(0), -1)

        vid_2 = video_obs[:, 1]  # (4, 7, 256, 256)
        x_vid2 = F.silu(self.video2_conv1(vid_2.to(torch.bfloat16) / 255))
        x_vid2 = self.vid2_bn1(x_vid2)
        x_vid2 = F.silu(self.video2_conv2(x_vid2))
        x_vid2 = self.vid2_bn2(x_vid2)
        x_vid2 = F.silu(self.video2_conv3(x_vid2))
        x_vid2 = self.vid2_bn3(x_vid2)
        x_vid2 = x_vid2.reshape(x_vid2.size(0), -1)

        # observe graphs:
        batch_shape = x_vid1.shape[0]
        encoded_mech_graph_i = self.mech_graph_encoder(
            mech_graph_init_obs, batch_shape
        )  # not x_graph because graphs have their own x
        encoded_mech_graph_d = self.mech_graph_encoder(
            mech_graph_des_obs, batch_shape
        )  # not x_graph because graphs have their own x

        encoded_elec_graph_i = self.elec_graph_encoder(elec_graph_init_obs, batch_shape)
        encoded_elec_graph_d = self.elec_graph_encoder(elec_graph_des_obs, batch_shape)

        # concatenate all features
        x = torch.cat(
            [
                x_vox_i,
                x_vox_d,
                x_vid1,
                x_vid2,
                encoded_mech_graph_i,
                encoded_mech_graph_d,
                encoded_elec_graph_i,
                encoded_elec_graph_d,
            ],
            dim=-1,
        ).to(torch.bfloat16)
        x = self.combine_bn1(x)
        x = F.silu(self.fc1(x))
        x = self.combine_bn2(x)
        x = F.silu(self.fc2(x))
        x = self.combine_bn3(x)
        mean = self.out_mean(x)
        log_std = torch.clamp(self.out_log_std(x), -5.0, 2.0)
        return mean, log_std

    def sample_action(
        self,
        voxel_init_obs,
        voxel_des_obs,
        video_obs,
        mech_graph_init_obs,
        mech_graph_des_obs,
        elec_graph_init_obs,
        elec_graph_des_obs,
        deterministic=False,
    ):
        mean, log_std = self.forward(
            voxel_init_obs,
            voxel_des_obs,
            video_obs,
            mech_graph_init_obs.to(self.device),
            mech_graph_des_obs.to(self.device),
            elec_graph_init_obs.to(self.device),
            elec_graph_des_obs.to(self.device),
        )
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
        else:
            noise = torch.randn_like(mean)
            pre_tanh = mean + noise * std
        action = torch.tanh(pre_tanh)
        log_prob = -0.5 * (
            ((pre_tanh - mean) / std) ** 2
            + 2 * log_std
            + torch.log(torch.tensor(2 * torch.pi))
        )
        # correction for tanh
        log_prob = log_prob.sum(dim=-1, keepdim=True) - torch.log(
            1 - action.pow(2) + 1e-6
        ).sum(dim=-1, keepdim=True)
        return action.to(torch.bfloat16), log_prob.to(torch.bfloat16)


import torch

if hasattr(torch, "_dynamo"):
    torch_compile_disable = torch._dynamo.disable
else:

    def torch_compile_disable(fn):
        return fn


class SACCritic(nn.Module):
    """
    PyTorch implementation of twin Q-function critic.
    """

    def __init__(
        self,
        action_dim,
        mechanics_graph_in_dim,
        electronics_graph_in_dim,
        electronics_graph_out_dim,
        mechanics_graph_out_dim,
        device=None,
    ):
        super(SACCritic, self).__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --- Voxel encoding helpers for torchsparse ---
        @torch._dynamo.disable
        def voxel_encode_q1(x):
            x = self.conv3d_q1(x)
            x_dense = x.dense()
            return x_dense.view(x_dense.size(0), -1).to(torch.bfloat16)

        @torch._dynamo.disable
        def voxel_encode_q2(x):
            x = self.conv3d_q2(x)
            x_dense = x.dense()
            return x_dense.view(x_dense.size(0), -1).to(torch.bfloat16)

        self._voxel_encode_q1 = voxel_encode_q1
        self._voxel_encode_q2 = voxel_encode_q2
        # Shared conv encoders for Q1
        self.conv3d_q1 = nn.Sequential(
            tsnn.Conv3d(1, 2, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            tsnn.SiLU(),
            tsnn.BatchNorm(2),
            tsnn.Conv3d(2, 4, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            tsnn.SiLU(),
            tsnn.BatchNorm(4),
            tsnn.Conv3d(4, 8, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            tsnn.SiLU(),
            tsnn.BatchNorm(8),
        )  # note: they are float16 by default. Could set to tf32 for better performance. bfloat16 is not supported.
        # Given input shape (D0, H0, W0) = (256, 256, 256) and 3 Conv3d layers each with kernel_size=6, stride=4:
        # Layer 1: D1 = floor((256 - 6) / 4) + 1 = 63,  same for H1, W1
        # Layer 2: D2 = floor((63 - 6) / 3) + 1 = 15,   same for H2, W2
        # Layer 3: D3 = floor((15 - 6) / 3) + 1 = 3,    same for H3, W3
        # Output channels after last Conv3d = 8
        # Total flattened output dim = 8 * 3 * 3 * 3 = 216
        self.vid1_q1 = nn.Sequential(
            nn.Conv2d(7, 10, kernel_size=(6, 6), stride=(4, 4), dtype=torch.bfloat16),
            nn.BatchNorm2d(10, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Conv2d(10, 12, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16),
            nn.BatchNorm2d(12, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Conv2d(12, 14, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16),
            nn.BatchNorm2d(14, dtype=torch.bfloat16),
            nn.SiLU(),
        )  # note: BBF (bigger better faster, 2023) used impala-style convs, which have 15 conv layers and are better for video.
        # it's more sample-efficient, although notably more expensive.
        # perhaps consider using the impala-style convs.
        # though there is no evidence of this improving anything, as BRO paper said.
        # Simba/simba2 did not use vision at all.
        self.vid2_q1 = nn.Sequential(
            nn.Conv2d(7, 10, kernel_size=(6, 6), stride=(4, 4), dtype=torch.bfloat16),
            nn.BatchNorm2d(10, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Conv2d(10, 12, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16),
            nn.BatchNorm2d(12, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Conv2d(12, 14, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16),
            nn.BatchNorm2d(14, dtype=torch.bfloat16),
            nn.SiLU(),
        )
        compressed_video_shape = 350  # 300 # I don't know why - there is a mismatch between actor and critic on this value.
        compressed_voxel_shape = 216
        combined_q_dim = (
            compressed_video_shape * 2
            + compressed_voxel_shape * 2
            + electronics_graph_out_dim * 2
            + mechanics_graph_out_dim * 2
            + action_dim
        )
        self.q1_fc = nn.Sequential(
            nn.Linear(combined_q_dim, 256, dtype=torch.bfloat16),
            nn.BatchNorm1d(256, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(256, 256, dtype=torch.bfloat16),
            nn.BatchNorm1d(256, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(256, 1, dtype=torch.bfloat16),
        )
        self.mech_graph_encoder_q1 = GraphEncoderWithGlobalFeatures(
            num_features_graph=mechanics_graph_in_dim,
            hidden_dim_graph=256,
            out_dim_graph=mechanics_graph_out_dim // 2,
            global_embedding_in_dim=8,
            hidden_dim_global=256,
            out_dim_global=mechanics_graph_out_dim // 2,
            heads=2,
            dtype=torch.bfloat16,
        )
        self.elec_graph_encoder_q1 = GraphEncoder(
            num_features=electronics_graph_in_dim,
            hidden_dim=256,
            out_dim=electronics_graph_out_dim,
            heads=2,
            dtype=torch.bfloat16,
        )

        # Twin Q2
        self.conv3d_q2 = nn.Sequential(
            tsnn.Conv3d(1, 2, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            tsnn.BatchNorm(2),
            tsnn.SiLU(),
            tsnn.Conv3d(2, 4, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            tsnn.BatchNorm(4),
            tsnn.SiLU(),
            tsnn.Conv3d(4, 8, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            tsnn.BatchNorm(8),
            tsnn.SiLU(),
        )
        self.vid1_q2 = nn.Sequential(  # 7 channels!
            nn.Conv2d(7, 10, kernel_size=(6, 6), stride=(4, 4), dtype=torch.bfloat16),
            nn.BatchNorm2d(10, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Conv2d(10, 12, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16),
            nn.BatchNorm2d(12, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Conv2d(12, 14, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16),
            nn.BatchNorm2d(14, dtype=torch.bfloat16),
            nn.SiLU(),
        )
        self.vid2_q2 = nn.Sequential(  # 7 channels!
            nn.Conv2d(7, 10, kernel_size=(6, 6), stride=(4, 4), dtype=torch.bfloat16),
            nn.BatchNorm2d(10, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Conv2d(10, 12, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16),
            nn.BatchNorm2d(12, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Conv2d(12, 14, kernel_size=(6, 6), stride=(3, 3), dtype=torch.bfloat16),
            nn.BatchNorm2d(14, dtype=torch.bfloat16),
            nn.SiLU(),
        )
        self.q2_fc = nn.Sequential(
            nn.Linear(combined_q_dim, 256, dtype=torch.bfloat16),
            nn.BatchNorm1d(256, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(256, 256, dtype=torch.bfloat16),
            nn.BatchNorm1d(256, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(256, 1, dtype=torch.bfloat16),
        )
        self.mech_graph_encoder_q2 = GraphEncoderWithGlobalFeatures(
            num_features_graph=mechanics_graph_in_dim,
            hidden_dim_graph=256,
            out_dim_graph=mechanics_graph_out_dim // 2,
            global_embedding_in_dim=8,
            hidden_dim_global=256,
            out_dim_global=mechanics_graph_out_dim // 2,
            heads=2,
            dtype=torch.bfloat16,
        )
        self.elec_graph_encoder_q2 = GraphEncoder(
            num_features=electronics_graph_in_dim,
            hidden_dim=256,
            out_dim=electronics_graph_out_dim,
            heads=2,
            dtype=torch.bfloat16,
        )

    def forward(
        self,
        voxel_init_obs,
        voxel_des_obs,
        video_obs,
        mech_graph_init_obs,
        mech_graph_des_obs,
        elec_graph_init_obs,
        elec_graph_des_obs,
        action,
    ):
        # Encoder Q1
        # voxels
        x_vox_init_q1 = self._voxel_encode_q1(voxel_init_obs)
        x_vox_des_q1 = self._voxel_encode_q1(voxel_des_obs)

        # vid
        x_vid1 = self.vid1_q1(video_obs[:, 0]).view(video_obs.size(0), -1)
        x_vid2 = self.vid2_q1(video_obs[:, 1]).view(video_obs.size(0), -1)

        # graph
        batch_shape = x_vid1.shape[0]
        mech_graph_init_q1 = self.mech_graph_encoder_q1(
            mech_graph_init_obs, batch_shape
        )
        mech_graph_des_q1 = self.mech_graph_encoder_q1(mech_graph_des_obs, batch_shape)
        elec_graph_init_q1 = self.elec_graph_encoder_q1(
            elec_graph_init_obs, batch_shape
        )
        elec_graph_des_q1 = self.elec_graph_encoder_q1(elec_graph_des_obs, batch_shape)

        x1 = torch.cat(
            [
                x_vox_init_q1,
                x_vox_des_q1,
                x_vid1,
                x_vid2,
                mech_graph_init_q1,
                mech_graph_des_q1,
                elec_graph_init_q1,
                elec_graph_des_q1,
                action,
            ],
            dim=-1,
        ).to(dtype=torch.bfloat16)

        q1 = self.q1_fc(x1)
        # Encoder Q2
        # voxels
        x_vox_init_q2 = self._voxel_encode_q2(voxel_init_obs)
        x_vox_des_q2 = self._voxel_encode_q2(voxel_des_obs)

        # vid
        x_vid1_q2 = self.vid1_q2(video_obs[:, 0])
        x_vid1_q2 = x_vid1_q2.view(x_vid1_q2.size(0), -1).to(torch.bfloat16)
        x_vid2_q2 = self.vid2_q2(video_obs[:, 1])
        x_vid2_q2 = x_vid2_q2.view(x_vid2_q2.size(0), -1).to(torch.bfloat16)
        # graph
        batch_shape = x_vid1_q2.shape[0]
        mech_graph_init_q2 = self.mech_graph_encoder_q2(
            mech_graph_init_obs, batch_shape
        )
        mech_graph_des_q2 = self.mech_graph_encoder_q2(mech_graph_des_obs, batch_shape)
        elec_graph_init_q2 = self.elec_graph_encoder_q2(
            elec_graph_init_obs, batch_shape
        )
        elec_graph_des_q2 = self.elec_graph_encoder_q2(elec_graph_des_obs, batch_shape)
        x2 = torch.cat(
            [
                x_vox_init_q2,
                x_vox_des_q2,
                x_vid1_q2,
                x_vid2_q2,
                mech_graph_init_q2,
                mech_graph_des_q2,
                elec_graph_init_q2,
                elec_graph_des_q2,
                action,
            ],
            dim=-1,
        ).to(dtype=torch.bfloat16)
        q2 = self.q2_fc(x2)

        return q1, q2
