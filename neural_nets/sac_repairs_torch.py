import copy
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse.nn as tsnn
from torch_geometric.data import Batch, Data
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, TensorDict
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.tensor_specs import TensorSpec

from neural_nets.utils import hard_update, soft_update
from neural_nets.singleton_graph_buffer import GraphBuffer
from neural_nets.sparse_voxel_buffer import SparseVoxelBuffer
from neural_nets.graphs import GraphEncoder
import tensordict


class SACActor(nn.Module):
    """
    PyTorch implementation of SAC Actor network.
    """

    def __init__(self, action_dim, electronics_graph_encoded_dim, device=None):
        super(SACActor, self).__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Voxel encoder (3D conv layers)
        self.voxel_conv1 = tsnn.Conv3d(1, 2, kernel_size=(6, 6, 6), stride=(4, 4, 4))
        self.voxel_conv2 = tsnn.Conv3d(2, 4, kernel_size=(6, 6, 6), stride=(4, 4, 4))
        self.voxel_conv3 = tsnn.Conv3d(4, 8, kernel_size=(6, 6, 6), stride=(4, 4, 4))
        # Video encoder (2D conv layers)
        self.video_conv1 = nn.Conv2d(3, 6, kernel_size=(6, 6), stride=(4, 4))
        self.video_conv2 = nn.Conv2d(6, 8, kernel_size=(6, 6), stride=(4, 4))
        self.video_conv3 = nn.Conv2d(8, 12, kernel_size=(6, 6), stride=(4, 4))
        # Graph observations MLP
        combined_dim = 216 + 324 + electronics_graph_encoded_dim
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out_mean = nn.Linear(256, action_dim)
        self.out_log_std = nn.Linear(256, action_dim)
        self.graph_encoder = GraphEncoder(
            216, 256, electronics_graph_encoded_dim, heads=2
        )

    def forward(self, voxel_obs, video_obs, graph_obs: Batch):
        # TODO add support for mech graphs.

        # assume voxel_obs shape [B, 1, D, H, W]
        x_v = F.silu(self.voxel_conv1(voxel_obs))
        x_v = F.silu(self.voxel_conv2(x_v))
        x_v = F.silu(self.voxel_conv3(x_v))
        x_v = x_v.view(x_v.size(0), -1)

        # assume video_obs shape [B, C, H, W]
        x_vid = F.silu(self.video_conv1(video_obs))
        x_vid = F.silu(self.video_conv2(x_vid))
        x_vid = F.silu(self.video_conv3(x_vid))
        x_vid = x_vid.view(x_vid.size(0), -1)

        # observe graphs:
        encoded_graph = self.graph_encoder(
            graph_obs
        )  # not x_graph because graphs have their own x

        # concatenate all features
        x = torch.cat([x_v, x_vid, encoded_graph], dim=-1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        mean = self.out_mean(x)
        log_std = torch.clamp(self.out_log_std(x), -5.0, 2.0)
        return mean, log_std

    def sample_action(self, voxel_obs, video_obs, graph_obs, deterministic=False):
        mean, log_std = self.forward(voxel_obs, video_obs, graph_obs)
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
        return action, log_prob


class SACCritic(nn.Module):
    """
    PyTorch implementation of twin Q-function critic.
    """

    def __init__(self, action_dim, electronics_graph_out_dim, device=None):
        super(SACCritic, self).__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Shared conv encoders for Q1
        self.conv3d_q1 = nn.Sequential(
            tsnn.Conv3d(1, 2, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
            tsnn.Conv3d(2, 4, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
            tsnn.Conv3d(4, 8, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
        )
        self.conv2d_q1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
            nn.Conv2d(6, 8, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
            nn.Conv2d(8, 12, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
        )
        combined_q_dim = 216 + 324 + electronics_graph_out_dim + action_dim
        self.q1_fc = nn.Sequential(
            nn.Linear(combined_q_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        self.graph_encoder_q1 = GraphEncoder(
            216, 256, electronics_graph_out_dim, heads=2
        )

        # Twin Q2
        self.conv3d_q2 = nn.Sequential(
            tsnn.Conv3d(1, 2, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
            tsnn.Conv3d(2, 4, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
            tsnn.Conv3d(4, 8, kernel_size=(6, 6, 6), stride=(4, 4, 4)),
            nn.SiLU(),
        )
        self.conv2d_q2 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
            nn.Conv2d(6, 8, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
            nn.Conv2d(8, 12, kernel_size=(6, 6), stride=(4, 4)),
            nn.SiLU(),
        )
        self.q2_fc = nn.Sequential(
            nn.Linear(combined_q_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        self.graph_encoder_q2 = GraphEncoder(
            216, 256, electronics_graph_out_dim, heads=2
        )

    def forward(self, voxel_obs, video_obs, graph_obs, action):
        # Encoder Q1
        x_v1 = self.conv3d_q1(voxel_obs).view(voxel_obs.size(0), -1)
        x_vid1 = self.conv2d_q1(video_obs).view(video_obs.size(0), -1)
        graph_obs_q1 = self.graph_encoder_q1(graph_obs)
        x1 = torch.cat([x_v1, x_vid1, graph_obs_q1, action], dim=-1)
        q1 = self.q1_fc(x1)
        # Encoder Q2
        x_v2 = self.conv3d_q2(voxel_obs).view(voxel_obs.size(0), -1)
        x_vid2 = self.conv2d_q2(video_obs).view(video_obs.size(0), -1)
        graph_obs_q2 = self.graph_encoder_q2(graph_obs)
        x2 = torch.cat([x_v2, x_vid2, graph_obs_q2, action], dim=-1)
        q2 = self.q2_fc(x2)

        return q1, q2


# ===== SAC Trainer =====
class SACTrainer:
    def __init__(
        self,
        action_dim,
        electronics_graph_encoded_dim,
        device=None,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        buffer_size=100000,
        batch_size=256,
        cleanup_freq: int = 10,  # How often to clean up unused voxels
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.actor = SACActor(action_dim, electronics_graph_encoded_dim).to(self.device)
        self.critic = SACCritic(action_dim, electronics_graph_encoded_dim).to(
            self.device
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = -action_dim
        self.gamma = gamma
        self.tau = tau
        self.cleanup_freq = cleanup_freq
        self.steps_since_cleanup = 0

        # Sparse voxel storage
        self.voxel_buffer = SparseVoxelBuffer(device=self.device)

        # Initialize graph buffer for electronics observations
        self.graph_buffer = GraphBuffer(device=str(self.device))  # Convert to string for device specification

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Replay buffer setup
        tensor_dict = tensordict.TensorDict(
            {
                "init_voxel_id": torch.zeros(
                    (buffer_size,), dtype=torch.long
                ),  # Stores voxel IDs instead of full tensors
                "des_voxel_id": torch.zeros(
                    (buffer_size,), dtype=torch.long
                ),
                "init_graph_id": torch.zeros(
                    (buffer_size,), dtype=torch.long
                ),  # Store graph IDs instead of full tensors
                "des_graph_id": torch.zeros(
                    (buffer_size,), dtype=torch.long
                ),
                "video_obs": torch.zeros(
                    (buffer_size, 2, 7, 256, 256), dtype=torch.uint8
                ),  # Example shape, adjust as needed
                "reward": torch.zeros((buffer_size,), dtype=torch.float32),
                "action": torch.zeros((buffer_size, action_dim), dtype=torch.float32),
                "done": torch.zeros((buffer_size,), dtype=torch.bool),
            },
            batch_size=[buffer_size],
            device=self.device,
        )

        self.buffer_storage = TensorStorage(
            storage=tensor_dict, max_size=buffer_size, device=self.device
        )
        self.replay_buffer = TensorDictReplayBuffer(
            storage=self.buffer_storage, batch_size=batch_size
        )
        self.voxel_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.des_voxel_ids = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )

    def select_action(self, voxel_obs, video_obs, graph_obs, deterministic=False):
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(
                voxel_obs.to(self.device),
                video_obs.to(self.device),
                graph_obs.to(self.device),
            )
        self.actor.train()
        return action.cpu()

    def update(self, batch):
        v, vid, g, a, r, nv, nvid, ng, d = batch
        v, vid, g, a, r, nv, nvid, ng, d = [
            x.to(self.device) for x in (v, vid, g, a, r, nv, nvid, ng, d)
        ]
        with torch.no_grad():
            na, nlp = self.actor.sample_action(nv, nvid, ng)
            q1n, q2n = self.critic_target(nv, nvid, ng, na)
            qn = torch.min(q1n, q2n) - self.log_alpha.exp() * nlp
            target = r.unsqueeze(-1) + self.gamma * (1 - d.unsqueeze(-1)) * qn
        q1, q2 = self.critic(v, vid, g, a)
        cl = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        cl.backward()
        self.critic_optimizer.step()
        a2, lp = self.actor.sample_action(v, vid, g)
        q1n, q2n = self.critic(v, vid, g, a2)
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
        is_first_step: torch.BoolTensor,
        video_obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        init_graph_obs: Batch,
        des_graph_obs: Batch,
        done: torch.Tensor,
        voxel_init_obs: torch.Tensor,
        voxel_des_obs: torch.Tensor,
    ) -> None:
        """Helper method to add transitions to the replay buffer with sparse voxel storage."""
        # Store voxel observations in the sparse buffer and get their IDs
        for env_id in torch.nonzero(is_first_step).squeeze():
            self.voxel_ids[env_id] = self.voxel_buffer.add(voxel_init_obs[env_id])
            self.des_voxel_ids[env_id] = self.voxel_buffer.add(voxel_des_obs[env_id])

        # Add graphs to the graph buffer and get their IDs
        # Convert PyG Batch/Data to dict if needed
        def get_graph_components(graph):
            if isinstance(graph, (Batch, Data)):
                return {
                    'edge_index': graph.edge_index,
                    'node_features': getattr(graph, 'x', None),
                    'edge_features': getattr(graph, 'edge_attr', None)
                }
            return graph
            
        init_graph = get_graph_components(init_graph_obs)
        des_graph = get_graph_components(des_graph_obs)
        
        init_graph_id = self.graph_buffer.add(
            edge_index=init_graph['edge_index'],
            node_features=init_graph.get('node_features'),
            edge_features=init_graph.get('edge_features')
        )
        des_graph_id = self.graph_buffer.add(
            edge_index=des_graph['edge_index'],
            node_features=des_graph.get('node_features'),
            edge_features=des_graph.get('edge_features')
        )

        # Add to replay buffer
        batch_data = {
            "init_voxel_id": self.voxel_ids.unsqueeze(0).to(self.device),
            "des_voxel_id": self.des_voxel_ids.unsqueeze(0).to(self.device),
            "init_graph_id": torch.tensor([[init_graph_id]], device=self.device),
            "des_graph_id": torch.tensor([[des_graph_id]], device=self.device),
            "video_obs": video_obs.unsqueeze(0).to(self.device) if video_obs is not None else None,
            "action": action.unsqueeze(0).to(self.device) if action is not None else None,
            "reward": torch.tensor([reward], device=self.device) if reward is not None else None,
            "done": done.unsqueeze(0).to(self.device) if done is not None else None,
        }
        batch_data = {k: v for k, v in batch_data.items() if v is not None}
        
        batch = TensorDict(
            batch_data,
            batch_size=[1],  # Single transition
            device=self.device,
        )
        self.replay_buffer.add(batch)

        # Periodically clean up unused voxels
        self.steps_since_cleanup += 1
        if self.steps_since_cleanup >= self.cleanup_freq:
            self._cleanup_voxel_buffer()
            self.steps_since_cleanup = 0

    def _cleanup_voxel_buffer(self) -> None:
        """Clean up unused voxels from the buffer."""
        # Get all active voxel IDs from the replay buffer
        all_voxel_ids = torch.cat([self.voxel_ids, self.des_voxel_ids]).unique()
        # note: it should be fairly safe to use unique on only first 20% of tensors or so, since they are more or less sorted.

        # Clean up unused voxels
        self.voxel_buffer.cleanup(all_voxel_ids)

    def get_batch_voxels(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve voxel tensors from the sparse buffer for a batch."""
        # Get unique voxel IDs in this batch
        voxel_ids = torch.cat([batch["init_voxel_id"], batch["des_voxel_id"]]).unique()

        # Create a mapping from ID to index for fast lookup
        id_to_idx = {vid.item(): i for i, vid in enumerate(voxel_ids)}

        # Get all unique voxels
        voxels = torch.stack([self.voxel_buffer.get(vid.item()) for vid in voxel_ids])

        # Create index tensors for batch lookups
        batch_voxel_indices = torch.tensor(
            [id_to_idx[vid.item()] for vid in batch["init_voxel_id"]], device=self.device
        )
        des_batch_voxel_indices = torch.tensor(
            [id_to_idx[vid.item()] for vid in batch["des_voxel_id"]],
            device=self.device,
        )

        return voxels, batch_voxel_indices, des_batch_voxel_indices


# ===== Training Orchestrator =====
def run_training(
    env_setups,
    tasks,
    env_cfg,
    obs_cfg,
    reward_cfg,
    command_cfg,
    ml_batch_dim,
    action_dim,
    electronics_graph_dim,
    num_steps=10000,
    prefill_steps=1000,
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
    )
    trainer = SACTrainer(
        action_dim,
        electronics_graph_dim,
        buffer_size=buffer_size,
        batch_size=ml_batch_dim,
    )

    voxel_init_obs, voxel_des_obs, video_obs, graph_curr_obs, graph_des_obs = (
        env.reset()
    )

    prev_video_obs = video_obs

    # Prefill replay buffer with random actions
    for _ in range(prefill_steps):
        rand_action = torch.randn((ml_batch_dim, action_dim), device=trainer.device)
        # note: action should probably be rescaled to franka arm space.
        (
            voxel_init,
            voxel_des,
            video_obs,
            graph_obs,
            graph_des,
            rewards,
            dones,
            info,
        ) = env.step(rand_action)

        trainer._add_to_buffer(
            is_first_step=dones,  # note: here done is true if the episode is over and reset.
            # however I'm not sure if it was correct to reset environments without observing them.
            voxel_init_obs=voxel_init_obs,
            voxel_des_obs=voxel_des_obs,
            video_obs=prev_video_obs,
            init_graph_obs=graph_curr_obs,
            des_graph_obs=graph_des_obs,
            action=rand_action,
            reward=rewards,
            # next_voxel_init_obs=voxel_init,
            # next_voxel_des_obs=voxel_des,
            # next_video_obs=video_obs,
            # next_graph_curr_obs=graph_obs,
            # next_graph_des_obs=graph_des,
            done=dones,
        )
        prev_video_obs = video_obs

    # Main training loop
    for step in range(num_steps):
        # Get action from policy
        action = trainer.select_action(
            voxel_init_obs, voxel_des_obs, video_obs, graph_curr_obs
        )

        # Step environment
        (
            voxel_init,
            voxel_des,
            video_obs,
            graph_obs,
            graph_des,
            rewards,
            dones,
            info,
        ) = env.step(action)

        # Store transition in replay buffer
        trainer._add_to_buffer(
            voxel_init_obs=voxel_init_obs,
            voxel_des_obs=voxel_des_obs,
            video_obs=prev_video_obs,
            init_graph_obs=graph_curr_obs,
            des_graph_obs=graph_des_obs,
            action=action.to(trainer.device),
            reward=rewards,
            next_voxel_init_obs=voxel_init,
            next_voxel_des_obs=voxel_des,
            next_video_obs=video_obs,
            next_graph_curr_obs=graph_obs,
            next_graph_des_obs=graph_des,
            done=dones,
        )

        # Update policy if we have enough samples
        if step >= prefill_steps:
            batch = trainer.replay_buffer.sample()

            # Get voxel tensors for the batch
            voxels, voxel_indices, next_voxel_indices = trainer.get_batch_voxels(batch)

            # Get graph data from buffer using stored IDs
            init_graph_batch = trainer.graph_buffer.get_batch(batch["init_graph_id"].flatten())
            des_graph_batch = trainer.graph_buffer.get_batch(batch["des_graph_id"].flatten())

            # Combine graph features for current and next states
            graph_obs = torch.cat([
                init_graph_batch['node_features'],
                des_graph_batch['node_features']
            ], dim=1)

            next_graph_batch = trainer.graph_buffer.get_batch(batch["des_graph_id"].flatten())
            next_graph_obs = torch.cat([
                next_graph_batch['node_features'],
                des_graph_batch['node_features']  # Keep desired graph for next state
            ], dim=1)

            # Update networks
            cl, al, alpha = trainer.update(
                voxels=voxels,
                video_obs=batch["video_obs"],
                graph_obs=graph_obs,
                action=batch["action"],
                reward=batch["reward"],
                next_voxels=voxels[next_voxel_indices],
                next_video_obs=batch["video_obs"],
                next_graph_obs=next_graph_obs,
                done=batch["done"],
                voxel_indices=voxel_indices,
            )

            if step % 1000 == 0:
                print(f"Step {step}: critic_loss={cl}, actor_loss={al}, alpha={alpha}")
                print(
                    f"Buffer: {len(trainer.replay_buffer)} transitions, "
                    f"{len(trainer.voxel_buffer)} unique voxels"
                )

        # Update observations
        prev_video_obs = video_obs  # only prev_video_obs is stored
        action = action.to(trainer.device)


if __name__ == "__main__":
    # Example setup for training
    from repairs_components.processing.tasks import AssembleTask, DisassembleTask

    # Initialize Genesis
    gs.init(backend=gs.cuda)

    # Create task and environment setup
    tasks = [AssembleTask(), DisassembleTask()]
    env_setups = [MoveBoxSetup()]

    debug = True

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
        "dataloader_settings": {
            "prefetch_memory_size": 256
            if not debug
            else 4  # 256 environments per scene.
        },  # note^ 4 is for faster env spinup.
        "min_bounds": (-0.6, -0.7, -0.1),
        "max_bounds": (0.5, 0.5, 2),
        "save_obs": {
            # "video": True,
            # "voxel": True,
            # "electronic_graph": True,
            # "path": "./obs/",
            "video": False,  # not flooding the disk..
            "voxel": False,
            "electronic_graph": False,
            "path": "./obs/",
        },
    }

    obs_cfg = {
        "num_obs": 3,  # RGB, depth, segmentation
        "res": (256, 256),
    }

    reward_cfg = {
        "success_reward": 10.0,
        "progress_reward_scale": 1.0,
        "progressive": True,  # TODO : if progressive, use progressive reward calc instead.
    }

    command_cfg = {}

    action_dim = env_cfg["num_actions"]
    num_cameras = 2
    vision_obs_dim = (
        num_cameras,
        256,
        256,
        7,
    )  # 2 cameras, 7 channels (RGB, depth, segmentation)
    electronics_graph_encoded_dim = 64  # latent dim from graph encoder
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
        500 if debug else 200_000
    )  # was 200_000, reduced due to GPU constraints.
    min_buffer_len = 300 if debug else 10_000
    # ^46gb at 2*256*256*7*int8 res!!!
    sample_batch_size = 256

    run_training(
        env_setups=env_setups,
        tasks=tasks,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        ml_batch_dim=batch_size,
        action_dim=action_dim,
        electronics_graph_dim=electronics_graph_encoded_dim,
    )
