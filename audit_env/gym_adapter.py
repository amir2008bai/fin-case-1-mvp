"""Optional Gymnasium-compatible adapter (zero runtime dependencies).

If `gymnasium` is installed, `GymAuditEnv` inherits from `gymnasium.Env` and
exposes real `gymnasium.spaces.Text` action and observation spaces over UTF-8
JSON strings, plus `encode`/`decode` helpers between `dict` and JSON. Without
`gymnasium`, it exposes the identical `reset(seed, options)` / `step(action)`
API with `action_space = None` and `observation_space = None` so the offline
core package keeps zero external dependencies.
"""
import json
import random

from .contracts import MAX_FRAME_BYTES
from .environment import AuditEnv

try:  # pragma: no cover - optional dependency
    import gymnasium as _gym
    _BASE = _gym.Env
except ImportError:
    _gym = None
    _BASE = object

OBSERVATION_MAX_CHARS = 8 * MAX_FRAME_BYTES  # escaped Cyrillic takes six characters per letter
# Printable ASCII + JSON whitespace + Cyrillic block + common accounting/punctuation symbols.
JSON_CHARSET = (
    "".join(chr(c) for c in range(32, 127))
    + "\t\n\r"
    + "".join(chr(c) for c in range(0x0370, 0x0500))
    + "".join(chr(c) for c in range(0x2000, 0x2300))
    + "№«»·×±§©®™"
)


class GymAuditEnv(_BASE):
    """Thin Gymnasium-compatible wrapper around AuditEnv."""

    metadata = {"render_modes": []}

    def __init__(self, *, training=True, timeout_seconds=60, key=None, default_config=None, json_text=False):
        """`json_text=False` (default): actions and observations are dicts, as in AuditEnv; no Space is claimed.

        `json_text=True`: actions and observations are JSON strings (ASCII, non-ASCII escaped), and
        `action_space`/`observation_space` are `gymnasium.spaces.Text` that really contain them, so the
        environment passes `gymnasium.utils.env_checker.check_env`. A string that is not a JSON object is
        an invalid action: a bounded error that costs a step, never an exception.
        """
        self._env = AuditEnv(training=training, timeout_seconds=timeout_seconds, key=key)
        self._default_config = default_config
        self.json_text = json_text
        if json_text and _gym is not None:  # pragma: no cover - optional dependency
            self.action_space = _gym.spaces.Text(min_length=0, max_length=MAX_FRAME_BYTES, charset=JSON_CHARSET)
            self.observation_space = _gym.spaces.Text(min_length=2, max_length=OBSERVATION_MAX_CHARS,
                                                      charset=JSON_CHARSET)
        else:
            self.action_space = None
            self.observation_space = None
        self.action_schema = "schemas/action.schema.json"

    @staticmethod
    def encode(payload):
        """Serialize a dict observation or action to compact JSON; non-ASCII is escaped, so the text is
        always within JSON_CHARSET."""
        if not isinstance(payload, dict):
            raise TypeError("expected dict payload")
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    @staticmethod
    def decode(text):
        """Parse a UTF-8 JSON string back into a dict observation or action."""
        if not isinstance(text, str):
            raise TypeError("expected JSON str")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("decoded JSON payload must be an object")
        return parsed

    def reset(self, *, seed=None, options=None):
        if _gym is not None:  # pragma: no cover - seeds Env.np_random as gymnasium expects
            super().reset(seed=seed)
        # Gymnasium semantics: reset(seed=s) re-seeds; reset() without a seed draws the next episode seed
        # from that stream instead of replaying episode 0 forever.
        # reset(seed=s) plays episode s, the same as AuditEnv.reset(s).
        if seed is not None:
            self._seed_stream = random.Random(int(seed))
            episode_seed = int(seed)
        else:
            if not hasattr(self, "_seed_stream"):
                self._seed_stream = random.Random(0)
            episode_seed = self._seed_stream.randrange(2**31)
        config = (options or {}).get("config", self._default_config)
        observation, info = self._env.reset(episode_seed, config)
        return (self.encode(observation) if self.json_text else observation), info

    def step(self, action):
        if isinstance(action, str):
            try:
                action = self.decode(action)
            except (ValueError, TypeError, RecursionError):
                action = None  # AuditEnv answers invalid_action and charges the step
        observation, reward, terminated, truncated, info = self._env.step(action)
        if self.json_text:
            observation = self.encode(observation)
            reward = 0.0 if reward is None else reward  # closed mode: the reward stays with the evaluator
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
