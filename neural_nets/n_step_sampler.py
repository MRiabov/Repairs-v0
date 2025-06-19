import pickle
import tensordict
from torchrl.data.replay_buffers.samplers import Sampler
import torch


class StepAndNextSampler(Sampler):
    def __init__(self, n: int = 1, gamma: float = 0.99, batch_size: int = 32):
        super().__init__()
        self.n = n
        self.gamma = gamma
        self.batch_size = batch_size

    def sample(self, buffer, batch_size: int = None):
        # get the highest filled index
        # max_idx = buffer["index"].nonzero().squeeze(1)[-1] # below is faster.
        max_idx = min(buffer["index"].max().item(), buffer["index"].shape[0])
        assert max_idx > self.n, "Buffer too small for n-step sampling."
        if batch_size is None:
            batch_size = self.batch_size

        selected_idxs = torch.randint(0, max_idx, (batch_size,), device=buffer.device)

        # Construct [batch_size, n+1] index matrix
        steps = torch.arange(self.n + 1, device=buffer.device)
        index_matrix = selected_idxs.unsqueeze(1) + steps.unsqueeze(0)  # [B, n+1]

        return index_matrix, {}

    def state_dict(self):
        return {
            "n": self.n,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "rng_state": self._rng.get_state(),
        }

    def load_state_dict(self, state_dict):
        self.n = state_dict["n"]
        self.gamma = state_dict["gamma"]
        self.batch_size = state_dict["batch_size"]
        self._rng.set_state(state_dict["rng_state"])

    def _empty(self):
        """Create an uninitialized clone."""
        return StepAndNextSampler(self.n, self.gamma, self.batch_size)

    def dumps(self) -> bytes:
        """Serialize the sampler state."""
        return pickle.dumps(self.state_dict())

    def loads(self, state: bytes):
        """Deserialize the sampler state."""
        self.load_state_dict(pickle.loads(state))
