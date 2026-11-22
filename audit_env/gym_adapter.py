"""Optional Gymnasium-compatible adapter (zero runtime dependencies).

If `gymnasium` is installed, `GymAuditEnv` inherits from `gymnasium.Env`;
otherwise it exposes the identical `reset(seed, options)` / `step(action)` API
so RL training stacks (TRL, RLlib, Prime Intellect verifiers, SB3) can plug in
without adding any dependency to the offline core package.
"""
from .environment import AuditEnv

try:  # pragma: no cover - optional dependency
    import gymnasium as _gym
    _BASE = _gym.Env
except ImportError:
    _gym = None
    _BASE = object


class GymAuditEnv(_BASE):
    """Thin Gymnasium-compatible wrapper around AuditEnv."""

    metadata = {"render_modes": []}

    def __init__(self, *, training=True, timeout_seconds=60, key=None, default_config=None):
        self._env = AuditEnv(training=training, timeout_seconds=timeout_seconds, key=key)
        self._default_config = default_config
        # Actions and observations are JSON objects, not tensors: their contract is the JSON Schema in
        # schemas/action.schema.json and the observation description in docs/protocol.md. A text policy
        # (an LLM) consumes them directly; a tensor policy needs its own encoder, so no Space is claimed.
        self.action_space = None
        self.observation_space = None
        self.action_schema = "schemas/action.schema.json"

    def reset(self, *, seed=None, options=None):
        episode_seed = 0 if seed is None else int(seed)
        config = (options or {}).get("config", self._default_config)
        return self._env.reset(episode_seed, config)

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)
        # Gymnasium expects a numeric float when training; closed mode preserves None.
        return observation, reward, terminated, truncated, info

    @property
    def outcome(self):
        return self._env.outcome

    def close(self):
        return None


class SyncVectorAuditEnv:
    """Pure-stdlib Gymnasium-style vectorized wrapper for batched multi-episode rollouts."""

    def __init__(self, num_envs, *, training=True, timeout_seconds=60, key=None, default_config=None):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = num_envs
        self.envs = [GymAuditEnv(training=training, timeout_seconds=timeout_seconds, key=key,
                                 default_config=default_config) for _ in range(num_envs)]

    def reset(self, *, seeds=None, options=None):
        if seeds is None:
            seeds = list(range(self.num_envs))
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        obs_list, info_list = [], []
        for env, seed in zip(self.envs, seeds):
            obs, info = env.reset(seed=seed, options=options)
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            rewards.append(reward)
            terms.append(terminated)
            truncs.append(truncated)
            infos.append(info)
        return obs_list, rewards, terms, truncs, infos

    def close(self):
        for env in self.envs:
            env.close()
