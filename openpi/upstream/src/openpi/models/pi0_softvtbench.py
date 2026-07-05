import dataclasses

import einops
from flax import traverse_util
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, posemb_sincos
from openpi.shared import array_typing as at


class ResidualMLPBlock(nnx.Module):
    """Checkpoint-compatible 3-layer residual MLP block used by SoftVTBench tactile encoders."""

    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.kernels = nnx.Dict({f"kernel_{i}": nnx.Linear(in_dim, out_dim, rngs=rngs) for i in range(3)})
        if in_dim != out_dim:
            self.residual_proj = nnx.Linear(in_dim, out_dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b d"]:
        residual = self.residual_proj(x) if hasattr(self, "residual_proj") else x
        ys = [nnx.swish(kernel(x)) for kernel in self.kernels.values()]
        y = sum(ys) / (len(ys) ** 0.5)
        return nnx.swish(y + residual)


class TactilePrefixEncoder(nnx.Module):
    def __init__(self, *, tactile_dim: int, prefix_width: int, hidden_dim: int, rngs: nnx.Rngs):
        self.blocks = nnx.Dict(
            {
                "block_0": ResidualMLPBlock(tactile_dim, hidden_dim, rngs=rngs),
                "block_1": ResidualMLPBlock(hidden_dim, hidden_dim, rngs=rngs),
            }
        )
        self.out_proj = nnx.Linear(hidden_dim, prefix_width, rngs=rngs)

    def __call__(self, tactile_prefix: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        x = self.blocks["block_0"](tactile_prefix)
        x = self.blocks["block_1"](x)
        return self.out_proj(x)


class TactileSuffixEncoder(nnx.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.proj_in = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.proj_out = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, tactile_suffix: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        x = self.proj_in(tactile_suffix)
        x = nnx.swish(x)
        return self.proj_out(x)


@dataclasses.dataclass(frozen=True)
class SoftVTBenchPi0Config(pi0_config.Pi0Config):
    tactile_prefix_dim: int = 396
    tactile_suffix_dim: int = 396
    tactile_suffix_history: int = 2
    tactile_prefix_width: int = 2048
    tactile_hidden_dim: int = 4096
    tactile_suffix_hidden_dim: int = 2048
    tactile_suffix_width: int = 1024

    @override
    def create(self, rng: at.KeyArrayLike) -> "SoftVTBenchPi0":
        return SoftVTBenchPi0(self, rngs=nnx.Rngs(rng))

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "SoftVTBenchPi0":
        """Load SoftVTBench weights, allowing smoke wrappers to omit LoRA/tactile-only params.

        Official SoftVTBench checkpoints contain these params. Base OpenPI checkpoints do not, so
        for interface smoke tests we materialize missing LoRA/tactile weights as zeros.
        """
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        ref = state.to_pure_dict()
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(ref, params)

        flat_ref = traverse_util.flatten_dict(ref, sep="/")
        flat_params = traverse_util.flatten_dict(params, sep="/")
        missing = [key for key in flat_ref if key not in flat_params]
        unsupported = [
            key for key in missing if not (key.startswith("tactile_") or "lora" in key)
        ]
        if unsupported:
            preview = ", ".join(unsupported[:8])
            raise ValueError(f"SoftVTBenchPi0 checkpoint is missing non-optional params: {preview}")
        for key in missing:
            spec = flat_ref[key]
            flat_params[key] = jnp.zeros(spec.shape, spec.dtype)

        params = traverse_util.unflatten_dict(flat_params, sep="/")
        at.check_pytree_equality(expected=ref, got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                tactile_prefix=jax.ShapeDtypeStruct([batch_size, self.tactile_prefix_dim], jnp.float32),
                tactile_suffix=jax.ShapeDtypeStruct(
                    [batch_size, self.tactile_suffix_history, self.tactile_suffix_dim], jnp.float32
                ),
            )
        return observation_spec, action_spec


class SoftVTBenchPi0(Pi0):
    def __init__(self, config: SoftVTBenchPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.tactile_suffix_history = config.tactile_suffix_history
        self.tactile_prefix_encoder = TactilePrefixEncoder(
            tactile_dim=config.tactile_prefix_dim,
            prefix_width=config.tactile_prefix_width,
            hidden_dim=config.tactile_hidden_dim,
            rngs=rngs,
        )
        self.tactile_suffix_encoder = TactileSuffixEncoder(
            in_dim=config.tactile_suffix_dim * config.tactile_suffix_history,
            hidden_dim=config.tactile_suffix_hidden_dim,
            out_dim=config.tactile_suffix_width,
            rngs=rngs,
        )
        self.tactile_suffix_to_prefix_proj = nnx.Linear(
            config.tactile_suffix_width, config.tactile_prefix_width, rngs=rngs
        )

    def _require_tactile(self, obs: _model.Observation) -> tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b d"]]:
        if obs.tactile_prefix is None:
            raise ValueError("SoftVTBenchPi0 requires observation.tactile_prefix")
        if obs.tactile_suffix is None:
            raise ValueError("SoftVTBenchPi0 requires observation.tactile_suffix")
        return obs.tactile_prefix, obs.tactile_suffix

    def _encode_tactile_suffix(self, tactile_suffix: at.Float[at.Array, "b ..."]) -> at.Float[at.Array, "b d"]:
        suffix = tactile_suffix.reshape((tactile_suffix.shape[0], -1))
        expected = self.tactile_suffix_encoder.proj_in.in_features
        if suffix.shape[-1] == self.tactile_suffix_encoder.proj_out.out_features:
            return suffix
        if suffix.shape[-1] < expected:
            repeat = expected // suffix.shape[-1]
            if expected % suffix.shape[-1] == 0:
                suffix = jnp.tile(suffix, (1, repeat))
            else:
                pad = [(0, 0)] * suffix.ndim
                pad[-1] = (expected - suffix.shape[-1], 0)
                suffix = jnp.pad(suffix, pad)
        elif suffix.shape[-1] > expected:
            suffix = suffix[..., -expected:]
        return self.tactile_suffix_encoder(suffix)

    @override
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        tactile_prefix, tactile_suffix = self._require_tactile(obs)
        tokens, input_mask, ar_mask = super().embed_prefix(obs)
        tactile_prefix_token = self.tactile_prefix_encoder(tactile_prefix)[:, None, :]
        tactile_suffix_prefix_token = self.tactile_suffix_to_prefix_proj(self._encode_tactile_suffix(tactile_suffix))
        tactile_suffix_prefix_token = tactile_suffix_prefix_token[:, None, :]
        tactile_tokens = jnp.concatenate([tactile_prefix_token, tactile_suffix_prefix_token], axis=1)
        tokens = jnp.concatenate([tokens, tactile_tokens], axis=1)
        tactile_mask = jnp.ones((tokens.shape[0], tactile_tokens.shape[1]), dtype=jnp.bool_)
        input_mask = jnp.concatenate([input_mask, tactile_mask], axis=1)
        ar_mask = jnp.concatenate([ar_mask, jnp.array([False] * tactile_tokens.shape[1])], axis=0)
        return tokens, input_mask, ar_mask

    @override
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        _, tactile_suffix = self._require_tactile(obs)
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tactile_suffix_token = self._encode_tactile_suffix(tactile_suffix)[:, None, :]
            tokens.extend([state_token, tactile_suffix_token])
            input_mask.append(jnp.ones((obs.state.shape[0], 2), dtype=jnp.bool_))
            ar_mask += [True, False]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond
