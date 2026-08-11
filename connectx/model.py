"""
The four learned pieces: an encoder (real state -> latent), a dynamics model
(latent + action -> predicted next latent + reward), a value head (latent ->
estimated cost-to-go), and a diagnostic decoder (latent -> reconstructed
state, used only to sanity-check the latent isn't collapsing -- never
consulted for a real decision at inference time).

Config-driven: nothing here is hardcoded to Connect-4 beyond `state_dim`/
`num_actions`, so this file would work for any domain with a fixed-length
tuple state and a fixed discrete action space.
"""
import torch
import torch.nn as nn


def mlp(dims, out_activation=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if not is_last:
            layers.append(nn.ReLU())
        elif out_activation is not None:
            layers.append(out_activation)
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = mlp([state_dim, hidden_dim, hidden_dim, latent_dim])

    def forward(self, state):
        return self.net(state)


class DynamicsModel(nn.Module):
    def __init__(self, latent_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.num_actions = num_actions
        self.trunk = mlp([latent_dim + num_actions, hidden_dim, hidden_dim])
        self.next_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, action_idx):
        action_onehot = nn.functional.one_hot(action_idx, self.num_actions).float()
        h = self.trunk(torch.cat([z, action_onehot], dim=-1))
        next_z = self.next_latent_head(h)
        reward = self.reward_head(h).squeeze(-1)
        return next_z, reward


class ValueHead(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = mlp([latent_dim, hidden_dim, hidden_dim, 1])

    def forward(self, z):
        return self.net(z).squeeze(-1)


class Decoder(nn.Module):
    def __init__(self, latent_dim, state_dim, hidden_dim=128):
        super().__init__()
        self.net = mlp([latent_dim, hidden_dim, hidden_dim, state_dim])

    def forward(self, z):
        return self.net(z)


class WorldModel(nn.Module):
    def __init__(self, state_dim, num_actions, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.encoder = Encoder(state_dim, latent_dim, hidden_dim)
        self.dynamics = DynamicsModel(latent_dim, num_actions, hidden_dim)
        self.value = ValueHead(latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, state_dim, hidden_dim)
        # Value-target normalization stats: NOT learned, set once by
        # whichever value-training pass runs (see verifier.py's
        # train_mc_value_onpolicy) from the actual label distribution it
        # sees. Buffers (not plain attributes) so they save/load with the
        # checkpoint automatically.
        self.register_buffer("value_target_mean", torch.tensor(0.0))
        self.register_buffer("value_target_std", torch.tensor(1.0))

    def encode(self, state):
        return self.encoder(state)

    def imagine_step(self, z, action_idx):
        return self.dynamics(z, action_idx)

    def evaluate(self, z):
        """Always returns real-scale value estimates (remaining cost, same
        units `imagine_step`'s predicted reward uses) -- the head internally
        predicts a normalized target, denormalized here so no caller needs
        to know normalization is happening."""
        raw = self.value(z)
        return raw * self.value_target_std + self.value_target_mean

    def reconstruct(self, z):
        return self.decoder(z)
