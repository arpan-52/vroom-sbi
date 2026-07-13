"""
Circular reparameterization of chi0 for NPE training.

chi0 is a polarization angle, defined modulo pi, but the flow trains on an
unbounded real line: for truths near the prior edges (0 or pi) the true
posterior mass wraps across the boundary, and an unbounded flow must assign
near-zero density to one side. Those cases produce catastrophic per-sample
losses that dominate the validation mean exactly when the posterior is sharp
(most visibly faraday_thin n=1), driving early stopping toward deliberately
blurry — under-confident — checkpoints.

Fix: the flow never sees chi0. Each component's chi0 is expanded to
(sin 2chi0, cos 2chi0), which is continuous across the 0 <-> pi wrap, and
posterior samples are contracted back via chi0 = atan2(s, c) / 2 mod pi.
The physical parameterization (and every downstream consumer: SBC, TARP,
paper figures, inference outputs) is unchanged.

chi0 is the *last* parameter of every component tuple in all four physical
models (see torch_physics), so the transform is uniform: physical
params-per-component ppc -> flow ppc + 1, with sin at slot ppc-1 and cos at
slot ppc.
"""

import math

import torch

from .torch_physics import params_per_comp


def flow_n_params(model_type: str, n_components: int) -> int:
    """Flow-space dimensionality: one extra dimension per component."""
    return (params_per_comp(model_type) + 1) * n_components


def expand_theta(
    theta: torch.Tensor, n_components: int, model_type: str
) -> torch.Tensor:
    """Physical theta (…, C*ppc) -> flow theta (…, C*(ppc+1)).

    Replaces each component's chi0 with (sin 2chi0, cos 2chi0).
    """
    ppc = params_per_comp(model_type)
    lead = theta.shape[:-1]
    t = theta.reshape(-1, n_components, ppc)
    two_chi = 2.0 * t[:, :, -1:]
    out = torch.cat([t[:, :, :-1], torch.sin(two_chi), torch.cos(two_chi)], dim=2)
    return out.reshape(*lead, n_components * (ppc + 1))


def contract_theta(
    theta_flow: torch.Tensor, n_components: int, model_type: str
) -> torch.Tensor:
    """Flow theta (…, C*(ppc+1)) -> physical theta (…, C*ppc).

    chi0 = atan2(sin, cos) / 2 mod pi. atan2 projects radially, so off-circle
    flow samples map to a well-defined angle with no ambiguity.
    """
    ppc = params_per_comp(model_type)
    lead = theta_flow.shape[:-1]
    t = theta_flow.reshape(-1, n_components, ppc + 1)
    chi0 = 0.5 * torch.atan2(t[:, :, -2:-1], t[:, :, -1:])
    chi0 = torch.remainder(chi0, math.pi)
    out = torch.cat([t[:, :, :-2], chi0], dim=2)
    return out.reshape(*lead, n_components * ppc)


def expand_bounds(
    low: torch.Tensor, high: torch.Tensor, n_components: int, model_type: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical prior box -> flow-space box.

    The chi0 slot becomes two slots bounded [-1, 1]. The box over-covers the
    unit circle the samples actually live on; DirectPosterior only uses it
    for support checks, so over-coverage means no wrongful rejections.
    """
    ppc = params_per_comp(model_type)
    lo = low.reshape(n_components, ppc)
    hi = high.reshape(n_components, ppc)
    ones = torch.ones(n_components, 1, dtype=low.dtype, device=low.device)
    lo_f = torch.cat([lo[:, :-1], -ones, -ones], dim=1)
    hi_f = torch.cat([hi[:, :-1], ones, ones], dim=1)
    return lo_f.reshape(-1), hi_f.reshape(-1)


class Chi0ReparamSimulator:
    """GPUSimulator wrapper that emits flow-space theta.

    Drop-in for OnlineNPETrainer: ``generate_batch`` returns
    ``(expand_theta(theta), x)``; ``low``/``high``/``n_params`` are exposed
    in flow space so prior construction stays uniform.
    """

    def __init__(self, simulator):
        self.simulator = simulator
        self.model_type = simulator.model_type
        self.n_components = simulator.n_components
        self.n_freq = simulator.n_freq
        self.device = simulator.device
        self.n_params = flow_n_params(self.model_type, self.n_components)
        self.low, self.high = expand_bounds(
            simulator.low, simulator.high, self.n_components, self.model_type
        )

    def generate_batch(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta, x = self.simulator.generate_batch(batch_size, generator=generator)
        return expand_theta(theta, self.n_components, self.model_type), x


class Chi0ReparamPosterior:
    """Posterior wrapper returning samples in physical parameter space.

    Wraps a posterior whose flow was trained on expanded theta; ``sample``
    contracts back so SBC/TARP/inference consumers stay in physical space.
    """

    def __init__(self, posterior, n_components: int, model_type: str):
        self.posterior = posterior
        self.n_components = n_components
        self.model_type = model_type

    def sample(self, sample_shape, x, **kwargs) -> torch.Tensor:
        s = self.posterior.sample(sample_shape, x=x, **kwargs)
        return contract_theta(s, self.n_components, self.model_type)
