"""The evaluator owns the deadline, records and answer key; agents speak JSONL over pipes.

Portable: a reader thread and a queue replace select() on pipes (which Windows lacks),
a writer thread keeps a stuck agent from blocking the evaluator past its deadline.
"""
import json
import math
import os
import queue
import subprocess
import threading
import uuid
from contextlib import suppress
from time import perf_counter

from .contracts import MAX_FRAME_BYTES


class ProtocolError(Exception):
    pass


def strict_loads(payload):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError("non-finite number")

    def finite_float(raw):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return value

    return json.loads(payload, object_pairs_hook=object_pairs, parse_constant=reject_constant,
                      parse_float=finite_float)


def wire_frame(observation, reward, terminated, truncated, info):
    return {
        "observation": observation, "reward": reward, "terminated": terminated,
        "truncated": truncated, "info": info,
    }


def record(episode_id, step, action, frame):
    return {"episode_id": episode_id, "step": step, "action": action,
            **{key: value for key, value in frame.items() if key != "info"}}


def execute(env, observation, policy, observer=None):
    frame = wire_frame(observation, 0.0, False, False, {})
    episode_id = observation["manifest"]["episode_id"]
    trajectory = [record(episode_id, 0, {"type": "reset"}, frame)]
    start = perf_counter()
    if observer is not None:
        observer(trajectory[0], 0.0)
    while not (frame["terminated"] or frame["truncated"]):
        started_step = perf_counter()
        try:
            action = policy(frame)
            response = env.step(action)
        except TimeoutError:
            action, response = None, env.expire()
        except ProtocolError:
            action, response = None, env.expire("agent_protocol")
        frame = wire_frame(*response)
        trajectory.append(record(episode_id, len(trajectory), action, frame))
        if observer is not None:
            observer(trajectory[-1], perf_counter() - started_step)
    return env.outcome, trajectory, perf_counter() - start


def run_policy(env, observation, policy, observer=None):
    return execute(env, observation, lambda frame: policy(frame["observation"]), observer)


class JsonAgent:
    def __init__(self, command, clock, deadline):
        self.command = command
        self.clock = clock
        self.deadline = deadline

    def __enter__(self):
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self.read_loop, daemon=True)
        self.reader.start()
        return self

    def __exit__(self, *_):
        with suppress(OSError):
            if self.process.poll() is None:
                self.process.kill()
        self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            with suppress(OSError):
                stream.close()

    def read_loop(self):
        buffer = bytearray()
        handle = self.process.stdout.fileno()
        try:
            while True:
                chunk = os.read(handle, 65536)
                if not chunk:
                    self.lines.put(("eof", None))
                    return
                buffer.extend(chunk)
                while True:
                    delimiter = buffer.find(b"\n")
                    if delimiter < 0:
                        if len(buffer) > MAX_FRAME_BYTES:
                            self.lines.put(("too_large", None))
                            return
                        break
                    if delimiter > MAX_FRAME_BYTES:
                        self.lines.put(("too_large", None))
                        return
                    self.lines.put(("line", bytes(buffer[:delimiter])))
                    del buffer[:delimiter + 1]
        except OSError:
            self.lines.put(("eof", None))

    def left(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    def write(self, payload):
        failure = []

        def send():
            try:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
            except OSError as exc:
                failure.append(exc)

        writer = threading.Thread(target=send, daemon=True)
        writer.start()
        writer.join(self.left())
        if writer.is_alive():
            raise TimeoutError
        if failure:
            raise ProtocolError("agent I/O failed")

    def __call__(self, frame):
        payload = (json.dumps(frame, ensure_ascii=True, allow_nan=False) + "\n").encode()
        self.write(payload)
        try:
            kind, line = self.lines.get(timeout=self.left())
        except queue.Empty:
            raise TimeoutError from None
        if kind == "eof":
            raise ProtocolError("agent closed stdout")
        if kind == "too_large":
            raise ProtocolError("action too large")
        try:
            return strict_loads(line.decode("utf-8"))
        except (ValueError, RecursionError, UnicodeDecodeError) as exc:
            raise ProtocolError("invalid JSON action") from exc


def run_subprocess(env, observation, command, observer=None):
    with JsonAgent(command, env.clock, env.deadline) as agent:
        return execute(env, observation, agent, observer)


def run_container(env, observation, image, policy, observer=None):
    name = f"audit-agent-{uuid.uuid4().hex}"
    command = [
        "docker", "run", "--rm", "-i", "--name", name, "--pull=never",
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--cpus=2", "--memory=4g", "--memory-swap=4g", "--pids-limit=64",
        "--user=65534:65534", "--log-driver=none", "--tmpfs=/tmp:rw,nosuid,noexec,size=32m",
        "--entrypoint=python", image, "-m", "audit_env.agents", policy,
    ]
    try:
        return run_subprocess(env, observation, command, observer)
    finally:
        # Killing the client alone does not necessarily stop its container.
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=5)
