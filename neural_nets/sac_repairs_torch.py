import copy
import time

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


# ===== SAC Trainer =====
class SACTrainer:
    def __init__(
        self,
        action_dim,
        electronics_graph_encoded_dim,
        mechanics_graph_encoded_dim,
        device=None,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        buffer_size=100000,
        batch_size=256,
        singleton_buffer_size=20000,
        sample_batch_size=256,
        cleanup_freq: int = 10,  # How often to clean up unused voxels and electronics graphs
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.actor = SACActor(
            action_dim,
            electronics_graph_in_dim=4,
            electronics_graph_out_dim=electronics_graph_encoded_dim,
            mechanics_graph_in_dim=8,
            mechanics_graph_out_dim=mechanics_graph_encoded_dim,
        ).to(self.device)
        self.critic = SACCritic(
            action_dim,
            electronics_graph_in_dim=4,
            electronics_graph_out_dim=electronics_graph_encoded_dim,
            mechanics_graph_in_dim=8,
            mechanics_graph_out_dim=mechanics_graph_encoded_dim,
        ).to(self.device)
        # Try to speed-up networks with torch.compile (PyTorch ≥ 2.0). If it
        # fails (e.g., unsupported ops), fall back to eager modules.
        try:
            # self.actor = torch.compile(self.actor, mode="default")
            # self.critic = torch.compile(self.critic, mode="default")
            """ """
        except Exception as err:
            print(
                f"[SACTrainer] torch.compile failed: {err}. Falling back to eager execution."
            )
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = -action_dim
        self.gamma = gamma
        self.tau = tau
        self.cleanup_freq = cleanup_freq
        self.steps_since_cleanup = 0

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Replay buffer setup
        # tensor_dict = tensordict.TensorDict(
        #     {
        #         "init_voxel_id": torch.zeros(
        #             (buffer_size,), dtype=torch.int
        #         ),  # Stores voxel IDs instead of full tensors
        #         "des_voxel_id": torch.zeros((buffer_size,), dtype=torch.int),
        #         "init_electronics_graph_id": torch.zeros(
        #             (buffer_size,), dtype=torch.int
        #         ),
        #         "des_electronics_graph_id": torch.zeros(
        #             (buffer_size,), dtype=torch.int
        #         ),
        #         "video_obs": torch.zeros(
        #             (buffer_size, 2, 7, 256, 256), dtype=torch.uint8
        #         ),  # Example shape, adjust as needed
        #         "reward": torch.zeros((buffer_size,), dtype=torch.bfloat16),
        #         "action": torch.zeros((buffer_size, action_dim), dtype=torch.bfloat16),
        #         "done": torch.zeros((buffer_size,), dtype=torch.bool),
        #     },
        #     batch_size=(buffer_size,),  # batch size= buffer size???
        #     device=self.device,
        # )

        self.buffer_storage = LazyMemmapStorage(
            max_size=buffer_size
        )  # device=self.device,  # storage=tensor_dict,

        self.replay_buffer = TensorDictReplayBuffer(
            storage=self.buffer_storage,
            batch_size=batch_size,
            sampler=StepAndNextSampler(n=1, gamma=0.99, batch_size=sample_batch_size),
        )  # FIXME: I should store this buffer in the ROM and prefetch it to GPU only due to video obs being large.

        self.replay_buffer.append_transform(lambda x: x.to(self.device))
        # ^ or at least video_obs.
        # Singleton storages
        # Sparse voxel & graph storage
        self.voxel_buffer = SparseVoxelBuffer(
            buffer_size=singleton_buffer_size,
            batch_size=batch_size,
            voxel_shape=(256, 256, 256),
            device=self.device,
        )
        self.elec_graph_buffer = GraphBuffer(
            node_feat_dim=4, store_global_feat=False, device=self.device
        )
        self.mech_graph_buffer = GraphBuffer(
            node_feat_dim=8,
            store_global_feat=True,
            max_globals=12,
            global_feat_dim=8,
            device=self.device,
        )
        # voxels ids
        self.voxel_ids = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.des_voxel_ids = torch.zeros(
            batch_size, dtype=torch.int, device=self.device
        )
        # electronics graphs ids
        self.init_elec_graph_ids = torch.zeros(
            batch_size, dtype=torch.int, device=self.device
        )
        self.des_elec_graph_ids = torch.zeros(
            batch_size, dtype=torch.int, device=self.device
        )
        # mechanics graphs ids
        self.init_mech_graph_ids = torch.zeros(
            batch_size, dtype=torch.int, device=self.device
        )
        self.des_mech_graph_ids = torch.zeros(
            batch_size, dtype=torch.int, device=self.device
        )

    def select_action(
        self,
        voxel_init_obs,
        voxel_des_obs,
        video_obs,
        elec_graph_init_obs,
        elec_graph_des_obs,
        mech_graph_init_obs,
        mech_graph_des_obs,
        deterministic=False,
    ):
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(
                voxel_init_obs,
                voxel_des_obs,
                video_obs.to(self.device),
                elec_graph_init_obs.to(self.device),
                elec_graph_des_obs.to(self.device),
                mech_graph_init_obs.to(self.device),
                mech_graph_des_obs.to(self.device),
            )
        self.actor.train()
        return action.cpu()

    # @torch.compile()  # dynamic=true? but is it?
    def update(
        self,
        vox_init,
        vox_des,
        vid_obs,
        elec_g_init,
        elec_g_des,
        mech_g_init,
        mech_g_des,
        a,
        r,
        next_vid,
        d,
    ):
        with torch.no_grad():
            na, nlp = self.actor.sample_action(
                vox_init,
                vox_des,
                next_vid,
                mech_g_init,
                mech_g_des,
                elec_g_init,
                elec_g_des,
            )
            q1n, q2n = self.critic_target(
                vox_init,
                vox_des,
                vid_obs,
                mech_g_init,
                mech_g_des,
                elec_g_init,
                elec_g_des,
                na,
            )
            qn = torch.min(q1n, q2n) - self.log_alpha.exp() * nlp
            target = (
                r.unsqueeze(-1)
                + self.gamma * (1 - d.unsqueeze(-1).to(torch.bfloat16)) * qn
            ).to(torch.bfloat16)
        q1, q2 = self.critic(
            vox_init,
            vox_des,
            vid_obs,
            mech_g_init,
            mech_g_des,
            elec_g_init,
            elec_g_des,
            a,
        )
        cl = (F.mse_loss(q1, target) + F.mse_loss(q2, target)).to(torch.bfloat16)
        self.critic_optimizer.zero_grad()
        cl.backward()
        self.critic_optimizer.step()
        a2, lp = self.actor.sample_action(
            vox_init,
            vox_des,
            vid_obs,
            mech_g_init,
            mech_g_des,
            elec_g_init,
            elec_g_des,
        )
        q1n, q2n = self.critic(
            vox_init,
            vox_des,
            vid_obs,
            mech_g_init,
            mech_g_des,
            elec_g_init,
            elec_g_des,
            a2,
        )
        al = (self.log_alpha.exp() * lp - torch.min(q1n, q2n)).mean()
        self.actor_optimizer.zero_grad()
        al.backward()
        self.actor_optimizer.step()
        xal = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        xal.backward()
        self.alpha_optimizer.step()
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        return cl.item(), al.item(), self.log_alpha.exp().item()

    # end of SACTrainer

    def _add_to_buffer(
        self,
        video_obs: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor,
        voxel_init_obs: torch.Tensor,
        voxel_des_obs: torch.Tensor,
        mech_graph_init_obs: Batch,
        mech_graph_des_obs: Batch,
        elec_graph_init_obs: Batch,
        elec_graph_des_obs: Batch,
    ) -> None:
        """Helper method to add transitions to the replay buffer with sparse voxel storage."""
        # Store voxel observations in the sparse buffer and get their IDs
        first_step_idx = done.to(self.device).nonzero().squeeze(1)
        voxel_init_obs = voxel_init_obs.to(self.device)
        voxel_des_obs = voxel_des_obs.to(self.device)
        if done.any():
            self.voxel_ids[first_step_idx] = self.voxel_buffer.add(
                voxel_init_obs.index_select(0, first_step_idx)
            ).int()  # note ^: index_select because getitem is not implemented on sparse arrays
            self.des_voxel_ids[first_step_idx] = self.voxel_buffer.add(
                voxel_des_obs.index_select(0, first_step_idx)
            ).int()
            # graphs
            batch_init_graph_obs = mech_graph_init_obs.to_data_list()
            batch_des_graph_obs = mech_graph_des_obs.to_data_list()
            self.init_mech_graph_ids[first_step_idx] = self.mech_graph_buffer.add(
                Batch.from_data_list([batch_init_graph_obs[i] for i in first_step_idx])
            ).int()
            self.des_mech_graph_ids[first_step_idx] = self.mech_graph_buffer.add(
                Batch.from_data_list([batch_des_graph_obs[i] for i in first_step_idx])
            ).int()
            # electronics graphs
            batch_init_graph_obs = elec_graph_init_obs.to_data_list()
            batch_des_graph_obs = elec_graph_des_obs.to_data_list()
            self.init_elec_graph_ids[first_step_idx] = self.elec_graph_buffer.add(
                Batch.from_data_list([batch_init_graph_obs[i] for i in first_step_idx])
            ).int()
            self.des_elec_graph_ids[first_step_idx] = self.elec_graph_buffer.add(
                Batch.from_data_list([batch_des_graph_obs[i] for i in first_step_idx])
            ).int()

        # Add to replay buffer
        self.replay_buffer.extend(  # debug note: `extend`, not `add.`
            tensordict.TensorDict(
                {
                    "init_voxel_id": self.voxel_ids,
                    "des_voxel_id": self.des_voxel_ids,
                    "video_obs": video_obs,
                    "init_electronics_graph_id": self.init_elec_graph_ids,
                    "des_electronics_graph_id": self.des_elec_graph_ids,
                    "init_mech_graph_id": self.init_mech_graph_ids,
                    "des_mech_graph_id": self.des_mech_graph_ids,
                    "action": action.to(torch.bfloat16),
                    "reward": rewards.to(torch.bfloat16),
                    "done": done,
                },
                batch_size=(done.size()[0],),
                device=self.device,
            )
        )

        # Periodically clean up unused voxels
        self.steps_since_cleanup += 1
        if self.steps_since_cleanup >= self.cleanup_freq:
            self._cleanup_singleton_buffers()
            self.steps_since_cleanup = 0

    def _cleanup_singleton_buffers(self) -> None:
        """Clean up unused voxels from the buffer."""
        # Get all active voxel IDs from the replay buffer
        all_voxel_ids = torch.cat([self.voxel_ids, self.des_voxel_ids]).unique()
        # note: it should be fairly safe to use unique on only first 20% of tensors or so, since they are more or less sorted.

        # Clean up unused voxels
        self.voxel_buffer.cleanup(all_voxel_ids)

        # Clean up unused electronics graphs
        all_graph_ids = torch.cat(
            [self.init_elec_graph_ids, self.des_elec_graph_ids]
        ).unique()
        self.mech_graph_buffer.cleanup(all_graph_ids)

    def get_batch_voxels(
        self, init_voxel_ids: torch.Tensor, des_voxel_ids: torch.Tensor
    ) -> tuple[torchsparse.SparseTensor, torchsparse.SparseTensor]:
        """Retrieve voxel tensors from the sparse buffer for a batch."""
        assert init_voxel_ids.shape == des_voxel_ids.shape, "Batch sizes must match"
        assert init_voxel_ids.ndim == 1, "Batch sizes must be 1D"
        # Get unique voxel IDs in this batch and get from buffer
        init_voxels = self.voxel_buffer.get(init_voxel_ids)
        des_voxels = self.voxel_buffer.get(des_voxel_ids)
        # Convert to torchsparse
        init_voxels = sparse_coo_to_torchsparse(init_voxels)
        des_voxels = sparse_coo_to_torchsparse(des_voxels)
        return init_voxels, des_voxels

    def get_batch_electronics_graphs(
        self, init_graph_ids: torch.Tensor, des_graph_ids: torch.Tensor
    ) -> tuple[Batch, Batch]:
        """Retrieve electronics graph tensors from the singleton buffer for a batch."""
        assert init_graph_ids.shape == des_graph_ids.shape, "Batch sizes must match"
        assert (
            torch.isin(init_graph_ids, des_graph_ids, invert=True)
            | (init_graph_ids == -1)
        ).all(), "Desired and initial graph IDs must never match"
        return self.elec_graph_buffer.get(init_graph_ids).to(
            self.device
        ), self.elec_graph_buffer.get(des_graph_ids).to(self.device)

    def get_batch_mechanical_graphs(
        self, init_graph_ids: torch.Tensor, des_graph_ids: torch.Tensor
    ) -> tuple[Batch, Batch]:
        """Retrieve mechanical graph tensors from the singleton buffer for a batch."""
        assert init_graph_ids.shape == des_graph_ids.shape, "Batch sizes must match"
        assert (
            torch.isin(init_graph_ids, des_graph_ids, invert=True)
            | (init_graph_ids == -1)
        ).all(), "Desired and initial graph IDs must never match"
        return self.mech_graph_buffer.get(init_graph_ids).to(
            self.device
        ), self.mech_graph_buffer.get(des_graph_ids).to(self.device)


# ===== Training Orchestrator =====
def run_training(
    env_setups,
    tasks,
    env_cfg,
    obs_cfg,
    reward_cfg,
    command_cfg,
    ml_batch_dim,
    sample_batch_size,
    action_dim,
    electronics_graph_out_dim,
    mechanics_graph_out_dim,
    num_steps=100000,
    prefill_steps=1000,
    buffer_size=10000,
    singleton_buffer_size=1000,
):
    """
    Orchestrates environment interaction, replay buffer filling, and training steps.
    """
    from repairs_components.training_utils.gym_env import RepairsEnv

    env = RepairsEnv(
        env_setups=env_setups,
        tasks=tasks,
        ml_batch_dim=ml_batch_dim,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        io_cfg=io_cfg,
    )
    trainer = SACTrainer(
        action_dim,
        electronics_graph_out_dim,
        mechanics_graph_out_dim,
        buffer_size=buffer_size,
        singleton_buffer_size=singleton_buffer_size,
        batch_size=ml_batch_dim,
        sample_batch_size=sample_batch_size,
    )
    # FIXME: I'm finding that voxel_init_obs is 2,256,256,256 when it should be 4,2,256,256,256
    (
        voxel_init_obs,
        voxel_des_obs,
        video_obs,
        mech_graph_init_obs,
        mech_graph_des_obs,
        elec_graph_init_obs,
        elec_graph_des_obs,
    ) = env.reset()

    prev_video_obs = video_obs
    # fill singleton buffers with one transition to avoid empty buffer error
    done = torch.ones(
        (ml_batch_dim,), dtype=torch.bool
    )  # all true to write into buffers.
    trainer._add_to_buffer(
        video_obs=video_obs,
        action=torch.zeros((ml_batch_dim, action_dim), dtype=torch.bfloat16),
        rewards=torch.zeros((ml_batch_dim,), dtype=torch.bfloat16),
        done=done,
        voxel_init_obs=voxel_init_obs,
        voxel_des_obs=voxel_des_obs,
        elec_graph_init_obs=elec_graph_init_obs,
        elec_graph_des_obs=elec_graph_des_obs,
        mech_graph_init_obs=mech_graph_init_obs,
        mech_graph_des_obs=mech_graph_des_obs,
    )

    prefill_start_time = time.time()
    # debug
    camera = env.__dict__["concurrent_scenes_data"][0].cameras[0]
    camera.start_recording()
    action_bound_min = torch.tensor(command_cfg["min_bounds"])
    action_bound_max = torch.tensor(command_cfg["max_bounds"])

    # Prefill replay buffer with random actions
    for _ in range(prefill_steps):
        rand_action = (
            torch.rand(
                (ml_batch_dim, action_dim), device=trainer.device, dtype=torch.bfloat16
            )
            * (action_bound_max - action_bound_min)
            + action_bound_min
        )
        # note: action should probably be rescaled to franka arm space.
        (
            voxel_init_obs,
            voxel_des_obs,
            video_obs,
            elec_graph_init_obs,
            elec_graph_des_obs,
            mech_graph_init_obs,
            mech_graph_des_obs,
            rewards,
            dones,
            info,
        ) = env.step(rand_action)

        trainer._add_to_buffer(
            # note: I'm not sure if it was correct to reset environments without observing them.
            voxel_init_obs=voxel_init_obs,
            voxel_des_obs=voxel_des_obs,
            video_obs=prev_video_obs,
            elec_graph_init_obs=elec_graph_init_obs,
            elec_graph_des_obs=elec_graph_des_obs,
            mech_graph_init_obs=mech_graph_init_obs,
            mech_graph_des_obs=mech_graph_des_obs,
            action=rand_action,
            rewards=rewards,
            # next_voxel_init_obs=voxel_init,
            # next_voxel_des_obs=voxel_des,
            # next_video_obs=video_obs,
            # next_graph_curr_obs=graph_obs,
            # next_graph_des_obs=graph_des,
            done=dones,
        )
        prev_video_obs = video_obs

    print(
        f"Buffer prefill steps ended. Elapsed time: {time.time() - prefill_start_time}"
    )
    camera.stop_recording(save_to_filename="video.mp4", fps=50)

    # Main training loop
    for step in range(num_steps):
        # Get action from policy
        action = trainer.select_action(
            # don't convert globally because buffer stores voxel obs as coo.
            sparse_coo_to_torchsparse(voxel_init_obs),
            sparse_coo_to_torchsparse(voxel_des_obs),
            video_obs,
            elec_graph_init_obs.to(trainer.device),
            elec_graph_des_obs.to(trainer.device),
            mech_graph_init_obs.to(trainer.device),
            mech_graph_des_obs.to(trainer.device),
        )

        # Step environment
        (
            voxel_init_obs,
            voxel_des_obs,
            video_obs,
            elec_graph_init_obs,
            elec_graph_des_obs,
            mech_graph_init_obs,
            mech_graph_des_obs,
            rewards,
            dones,
            info,
        ) = env.step(action)

        # Store transition in replay buffer
        trainer._add_to_buffer(
            voxel_init_obs=voxel_init_obs,
            voxel_des_obs=voxel_des_obs,
            video_obs=prev_video_obs,
            elec_graph_init_obs=elec_graph_init_obs,
            elec_graph_des_obs=elec_graph_des_obs,
            mech_graph_init_obs=mech_graph_init_obs,
            mech_graph_des_obs=mech_graph_des_obs,
            action=action.to(trainer.device),
            rewards=rewards,
            done=dones,
        )

        batch = trainer.replay_buffer.sample()
        prev_step_batch = {k: v[:, 0] for k, v in batch.items()}
        next_step_batch = {k: v[:, 1] for k, v in batch.items()}

        # Get voxel tensors for the batch
        init_voxels, des_voxels = trainer.get_batch_voxels(
            prev_step_batch["init_voxel_id"],
            prev_step_batch["des_voxel_id"],  # don't take the "next".
        )

        # Get electronics graph tensors for the batch
        init_graphs, des_graphs = trainer.get_batch_electronics_graphs(
            prev_step_batch["init_electronics_graph_id"],
            prev_step_batch["des_electronics_graph_id"],
        )

        init_mech_graphs, des_mech_graphs = trainer.get_batch_mechanical_graphs(
            prev_step_batch["init_mech_graph_id"],
            prev_step_batch["des_mech_graph_id"],
        )

        # Update networks
        cl, al, alpha = trainer.update(
            vox_init=init_voxels,
            vox_des=des_voxels,
            vid_obs=prev_step_batch["video_obs"].to(torch.bfloat16) / 255,
            elec_g_init=init_graphs,
            elec_g_des=des_graphs,
            mech_g_init=init_mech_graphs,
            mech_g_des=des_mech_graphs,
            a=prev_step_batch["action"],
            r=prev_step_batch["reward"],
            next_vid=next_step_batch["video_obs"],
            d=prev_step_batch["done"],
        )

        if step % 1000 == 0:
            print(f"Step {step}: critic_loss={cl}, actor_loss={al}, alpha={alpha}")

        # Update observations
        prev_video_obs = video_obs  # only prev_video_obs is stored
        action = action.to(trainer.device)


if __name__ == "__main__":
    # Example setup for training
    from repairs_components.processing.tasks import AssembleTask, DisassembleTask

    # Initialize Genesis
    gs.init(
        backend=gs.cuda, logging_level="warning"
    )  # note: logging level "warning" because genesis spams step speed logs during training.

    # Create task and environment setup
    tasks = [AssembleTask(), DisassembleTask()]
    env_setups = [MoveBoxSetup()]

    debug = True  # True
    force_recreate_data = False  # True
    # Note: set force_recreate_data to True after non-debug runs to remove large config files.

    # Environment configuration
    env_cfg = {
        "num_actions": 9,  # [x, y, z, quat_w, quat_x, quat_y, quat_z, gripper_force_left, gripper_force_right]
        "joint_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ],
        "default_joint_angles": {
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": 0.0,
            "joint4": -2.0,
            "joint5": 0.0,
            "joint6": 2.0,
            "joint7": 0.79,  # no "hand" here? there definitely was hand.
            "finger_joint1": 0.04,
            "finger_joint2": 0.04,
        },
        "min_bounds": (-0.6, -0.7, -0.1),
        "max_bounds": (0.5, 0.5, 2),
    }

    obs_cfg = {
        "num_obs": 3,  # RGB, depth, segmentation
        "res": (256, 256) if not debug else (64, 64),
        "use_random_textures": False,
    }

    reward_cfg = {
        "success_reward": 10.0,
        "progress_reward_scale": 1.0,
        "progressive": True,  # TODO : if progressive, use progressive reward calc instead.
    }

    io_cfg = {
        "generate_number_of_configs_per_scene": 164
        if not debug
        else 8,  # note: strange shape to debug
        "dataloader_settings": {
            "prefetch_memory_size": 512
            if not debug
            else 4  # 256 environments per scene.
        },  # note^ 4 is for faster env spinup.
        "data_dir": "/workspace/data",
        "save_obs": {
            # "video": True,
            # "voxel": True,
            # "electronic_graph": True,
            # "path": "./obs/",
            "video": False,  # not flooding the disk..
            "voxel": False,
            "electronic_graph": False,
            "mechanics_graph": False,
            "path": "/workspace/data/obs/",
        },
        "force_recreate_data": force_recreate_data,
        "env_setup_ids": list(range(1)),  # 1 scene now.
    }

    command_cfg = {
        "min_bounds": [
            *(-0.8, -0.8, 0),  # XYZ position min
            *(-1.0, -1.0, -1.0, -1.0),  # Quaternion components (w,x,y,z) min
            *(0.0, 0.0),  # Gripper control min
        ],
        "max_bounds": [
            *(0.8, 0.8, 1.0),  # XYZ position max
            # ^note: xyz is dep
            *(1.0, 1.0, 1.0, 1.0),
            *(1.0, 1.0),  # Quaternion components (w,x,y,z) max
        ],
    }

    action_dim = env_cfg["num_actions"]
    num_cameras = 2
    vision_obs_dim = (
        num_cameras,
        256,
        256,
        7,
    )  # 2 cameras, 7 channels (RGB, depth, segmentation)
    electronics_graph_encoded_dim = 64  # latent dim from graph encoder
    mechanics_graph_encoded_dim = 128
    electronics_graph_feat = 4  # number of features in graph.x
    voxel_obs_dim = (2, 256, 256, 256)  # start and finish # should be sparse?

    batch_size = (
        128
        if not debug
        else 4  # 16 if jax.default_backend() == "cpu" else 64  # 256 # note:debug atm.
    )
    train_steps = (
        10_000_000 if torch.cuda.is_available() and not debug else 3000
    ) // batch_size
    # train_steps = (10_000_000 if jax.default_backend() == "gpu" else 3000) // batch_size
    buffer_size = (
        100 if debug else 200_000
    )  # was 200_000, reduced due to GPU constraints.
    singleton_buffer_size = (
        20 if debug else 10_000
    )  # was 200_000, reduced due to GPU constraints.
    min_buffer_len = 40 if debug else 10_000
    prefill_steps = min_buffer_len // batch_size + 1
    # ^46gb at 2*256*256*7*int8 res!!! (w/o sparsity.)
    sample_batch_size = 256 if not debug else 4

    run_training(
        env_setups=env_setups,
        tasks=tasks,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        ml_batch_dim=batch_size,
        action_dim=action_dim,
        electronics_graph_out_dim=electronics_graph_encoded_dim,
        mechanics_graph_out_dim=mechanics_graph_encoded_dim,
        buffer_size=buffer_size,
        singleton_buffer_size=singleton_buffer_size,
        prefill_steps=prefill_steps,
        sample_batch_size=sample_batch_size,
    )
