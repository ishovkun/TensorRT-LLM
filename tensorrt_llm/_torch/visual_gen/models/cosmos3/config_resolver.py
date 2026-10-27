# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One place that turns a Cosmos3 checkpoint and the serving config into settings.

A Cosmos3 export carries the same facts in several files, written by different
exporters that are not kept in sync: the training config (``config.json``), the
diffusers pipeline index (``model_index.json``), the export metadata
(``checkpoint.json``) and the component configs (``transformer/``, ``scheduler/``,
``vae/``). :data:`RULES` names, for every setting, the sources in priority order
(the first one present wins) and what a disagreeing lower source means.
:class:`Cosmos3ConfigResolver` applies the table and returns a frozen
:class:`Cosmos3Config` whose every value remembers where it came from.

A setting no source provides and no default covers is an error. Every
:class:`Default` in the table is a value agreed on for checkpoints that omit the key.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from tensorrt_llm.logger import logger

from .sim_packing import SIM_CONFIG_KEY

CONFIG = "config.json"
CHECKPOINT = "checkpoint.json"
MODEL_INDEX = "model_index.json"
TRANSFORMER = "transformer/config.json"
SCHEDULER = "scheduler/scheduler_config.json"
VAE = "vae/config.json"

_SIM = SIM_CONFIG_KEY
# model_index.json stores non-component blocks under a leading underscore.
_SIM_INDEX = f"_{SIM_CONFIG_KEY}"

_MISSING = object()


class Conflict(Enum):
    """What a lower-priority source holding a different value means."""

    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


@dataclass(frozen=True)
class Key:
    """A dotted key path in one checkpoint file."""

    file: str
    path: str

    @property
    def label(self) -> str:
        return f"{self.file}:{self.path}"


@dataclass(frozen=True)
class Serving:
    """An attribute path in the serving config (``VisualGenArgs`` as the pipeline sees it)."""

    path: str

    @property
    def label(self) -> str:
        return f"serving:{self.path}"


@dataclass(frozen=True)
class Default:
    """The value used when no source declares the setting."""

    value: Any
    why: str

    @property
    def label(self) -> str:
        return f"default ({self.why})"


@dataclass(frozen=True)
class Rule:
    sources: tuple
    conflict: Conflict = Conflict.WARN
    required: bool = True
    """``False``: an unresolved setting is ``None`` instead of an error."""
    nullable: bool = False
    """``True``: a source holding ``null`` declares the setting (e.g. "no window");
    otherwise ``null`` counts as absent."""


@dataclass(frozen=True)
class Resolved:
    value: Any
    source: str
    disagreeing: tuple[str, ...] = ()
    """Lower-priority sources that held a different value."""


def _sim(path: str, *files: str) -> tuple[Key, ...]:
    """The same key inside the Sim block of each file, in the order given."""
    keys = []
    for f in files:
        block = _SIM_INDEX if f == MODEL_INDEX else _SIM
        keys.append(Key(f, f"{block}.{path}"))
    return tuple(keys)


_SIM_FILES = (CONFIG, TRANSFORMER, CHECKPOINT, MODEL_INDEX)

# Settings every Cosmos3 checkpoint resolves. Priority: serving config, then the
# training config (config.json), then the component configs, then the export
# metadata and the diffusers index.
RULES: dict[str, Rule] = {
    "architecture.position_embedding_type": Rule(
        (
            Key(TRANSFORMER, "position_embedding_type"),
            Default("unified_3d_mrope", "exports that predate the key"),
        )
    ),
    "architecture.max_position_embeddings": Rule(
        (
            Key(TRANSFORMER, "max_position_embeddings"),
            Default(262144, "exports that predate the key"),
        )
    ),
    "vae.temporal_compression": Rule(
        (
            Key(VAE, "scale_factor_temporal"),
            Key(TRANSFORMER, "temporal_compression_factor"),
            Default(4, "exports that predate the key"),
        ),
        conflict=Conflict.ERROR,
    ),
    "vae.spatial_compression": Rule(
        (
            Key(VAE, "scale_factor_spatial"),
            *_sim("vae_spatial_compression_factor", TRANSFORMER),
        ),
        conflict=Conflict.ERROR,
    ),
    "audio.temporal_compression": Rule(
        (
            Key(TRANSFORMER, "temporal_compression_factor_sound"),
            Default(1, "exports that predate the key"),
        )
    ),
    "prompt.use_system_prompt": Rule(
        (
            Key(CONFIG, "model.config.vlm_config.use_system_prompt"),
            Key(MODEL_INDEX, "default_use_system_prompt"),
        )
    ),
    "sampling.native_flow_schedule": Rule(
        (
            Key(MODEL_INDEX, "use_native_flow_schedule"),
            Default(False, "only Edge exports declare the native grid"),
        )
    ),
    "sampling.fixed_sigmas": Rule(
        (
            Key(SCHEDULER, "fixed_step_sampler_config.t_list"),
            *_sim("fixed_step_sampler_config.t_list", TRANSFORMER, CONFIG),
        ),
        conflict=Conflict.ERROR,
        required=False,
    ),
    "action.policy": Rule((Key(CHECKPOINT, "policy"),), required=False),
}

# Settings of autoregressive (Sim) checkpoints, which generate the video chunk by
# chunk; resolved when the transformer config carries the Sim block.
AUTOREGRESSIVE_RULES: dict[str, Rule] = {
    "prompt.text_cache_max_len": Rule(_sim("text_cache_max_len", *_SIM_FILES)),
    "action.tokens_per_frame": Rule(
        _sim("conditioning.action_tokens_per_frame", TRANSFORMER, CHECKPOINT),
        conflict=Conflict.ERROR,
    ),
    "action.embodiments": Rule(
        _sim("conditioning.embodiments", TRANSFORMER, CHECKPOINT), required=False
    ),
    "action.default_embodiment": Rule(
        _sim("conditioning.default_embodiment", TRANSFORMER, CHECKPOINT), required=False
    ),
    "action.input_action_dim": Rule(
        _sim("conditioning.input_contract.action_dim", TRANSFORMER, CHECKPOINT),
        required=False,
    ),
    "chunking.chunk_frames": Rule(
        (
            *_sim("chunk_size", *_SIM_FILES),
            Key(CONFIG, "model.config.teacher_forcing_frames_per_chunk"),
        ),
        conflict=Conflict.ERROR,
    ),
    "chunking.partition": Rule(_sim("chunk_partition", CHECKPOINT, TRANSFORMER), required=False),
    "chunking.video_temporal_causal": Rule(
        (
            Key(CONFIG, "model.config.video_temporal_causal"),
            *_sim("video_temporal_causal", *_SIM_FILES),
        ),
        conflict=Conflict.ERROR,
    ),
    # Framework semantics: frames a generated frame attends to, counting the sinks and
    # itself; null is full history. A serving value overrides the checkpoint.
    "kv_cache.inference_size": Rule(
        (
            Serving("kv_cache.kv_cache_inference_size"),
            Key(CONFIG, "model.config.kv_cache_inference_size"),
        ),
        required=False,
        nullable=True,
    ),
    # The Sim block's window: history frames counting the chunk's own first frame,
    # sinks excluded. Consulted when no inference size is declared.
    "kv_cache.window_frames": Rule(
        _sim("window_frames", *_SIM_FILES), required=False, nullable=True
    ),
    "kv_cache.sink_frames": Rule(
        (
            Serving("kv_cache.attention_sink_size"),
            Key(CONFIG, "model.config.attention_sink_size"),
            *_sim("sink_frames", *_SIM_FILES),
        )
    ),
    "kv_cache.free_gpu_memory_fraction": Rule(
        (Serving("kv_cache.free_gpu_memory_fraction"),), required=False
    ),
}


@dataclass(frozen=True)
class Cosmos3ArchitectureSettings:
    position_embedding_type: str
    max_position_embeddings: int


@dataclass(frozen=True)
class Cosmos3VAESettings:
    temporal_compression: int
    """Pixel frames per latent frame after the first, which is encoded alone."""
    spatial_compression: int


@dataclass(frozen=True)
class Cosmos3AudioSettings:
    temporal_compression: int


@dataclass(frozen=True)
class Cosmos3SamplingSettings:
    fixed_sigmas: Optional[tuple[float, ...]]
    """The distilled schedule (``t_list``); ``None`` on base checkpoints."""
    native_flow_schedule: bool


@dataclass(frozen=True)
class Cosmos3PromptSettings:
    use_system_prompt: bool
    text_cache_max_len: Optional[int]
    """Most prompt tokens a causal checkpoint's cache holds; ``None`` otherwise."""


@dataclass(frozen=True)
class Cosmos3ActionSettings:
    policy: Optional[Mapping[str, Any]]
    """The ``checkpoint.json`` policy block, unnormalized."""
    tokens_per_frame: Optional[int]
    embodiments: Mapping[str, Any] = field(default_factory=dict)
    default_embodiment: Optional[str] = None
    input_action_dim: Optional[int] = None


@dataclass(frozen=True)
class Cosmos3ChunkingSettings:
    chunk_frames: int
    partition: tuple[int, ...]
    """Leading chunk sizes in latent frames; the last one repeats and the final
    chunk is cut to what is left."""
    video_temporal_causal: bool


@dataclass(frozen=True)
class Cosmos3KVCacheSettings:
    inference_size: Optional[int]
    """Latent frames a generated frame attends to, counting the sinks and itself;
    ``None``: full history."""
    sink_frames: int
    free_gpu_memory_fraction: Optional[float]

    @property
    def history_frames(self) -> Optional[int]:
        """Recent frames before a frame that it attends to, sinks excluded;
        ``None``: all of them."""
        if self.inference_size is None:
            return None
        return self.inference_size - self.sink_frames - 1


@dataclass(frozen=True)
class Cosmos3Config:
    architecture: Cosmos3ArchitectureSettings
    vae: Cosmos3VAESettings
    audio: Cosmos3AudioSettings
    sampling: Cosmos3SamplingSettings
    prompt: Cosmos3PromptSettings
    action: Cosmos3ActionSettings
    chunking: Optional[Cosmos3ChunkingSettings]
    kv_cache: Optional[Cosmos3KVCacheSettings]
    resolved: Mapping[str, Resolved]

    @property
    def is_autoregressive(self) -> bool:
        """Generates chunk by chunk over a K/V cache (a Sim checkpoint)."""
        return self.chunking is not None

    def source_of(self, name: str) -> str:
        return self.resolved[name].source

    def describe(self) -> str:
        lines = ["Cosmos3 config:"]
        for name, r in sorted(self.resolved.items()):
            value = r.value
            if isinstance(value, Mapping):
                value = f"<{len(value)} entries>"
            line = f"  {name} = {value!r}  [{r.source}]"
            if r.disagreeing:
                line += f"  (disagrees: {', '.join(r.disagreeing)})"
            lines.append(line)
        return "\n".join(lines)


def _step(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, _MISSING)
    if obj is None or isinstance(obj, (str, int, float, bool, list, tuple)):
        return _MISSING
    return getattr(obj, key, _MISSING)


def _walk(doc: Any, path: str) -> Any:
    node = doc
    for part in path.split("."):
        node = _step(node, part)
        if node is _MISSING:
            return _MISSING
    return node


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


class Cosmos3ConfigResolver:
    """Applies :data:`RULES` (and :data:`AUTOREGRESSIVE_RULES` for Sim checkpoints).

    ``checkpoint_dir`` may be ``None``: only ``pretrained_config`` (standing in for
    ``transformer/config.json``) and ``serving`` are consulted then.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str],
        pretrained_config: Any = None,
        serving: Any = None,
    ) -> None:
        self._dir = checkpoint_dir
        self._pretrained = pretrained_config
        self._serving = serving
        self._docs: dict[str, Any] = {}

    def _doc(self, file: str) -> Any:
        if file == TRANSFORMER and self._pretrained is not None:
            return self._pretrained
        if file not in self._docs:
            doc = None
            if self._dir is not None:
                path = os.path.join(self._dir, file)
                if os.path.exists(path):
                    with open(path) as f:
                        doc = json.load(f)
            self._docs[file] = doc
        return self._docs[file]

    def _lookup(self, source: Any) -> Any:
        if isinstance(source, Key):
            doc = self._doc(source.file)
            return _MISSING if doc is None else _walk(doc, source.path)
        if isinstance(source, Serving):
            return _MISSING if self._serving is None else _walk(self._serving, source.path)
        if isinstance(source, Default):
            return source.value
        raise TypeError(f"unknown source {source!r}")

    def resolve_one(self, name: str, rule: Rule) -> Optional[Resolved]:
        winner: Optional[Resolved] = None
        disagreeing = []
        for source in rule.sources:
            if winner is not None and isinstance(source, Default):
                break
            value = self._lookup(source)
            if value is _MISSING or (value is None and not rule.nullable):
                continue
            if value is None and isinstance(source, Serving):
                continue  # an unset serving field defers to the checkpoint
            if winner is None:
                winner = Resolved(value, source.label)
                if isinstance(source, Serving):
                    break  # an override is meant to differ from the checkpoint
            elif not _same(value, winner.value):
                disagreeing.append(f"{source.label}={value!r}")
        if winner is None:
            if rule.required:
                labels = ", ".join(s.label for s in rule.sources)
                raise ValueError(f"Cosmos3 config: cannot resolve {name}; none of {labels} is set.")
            return None
        if disagreeing:
            message = (
                f"Cosmos3 config: {name} = {winner.value!r} from {winner.source}, but "
                + "; ".join(disagreeing)
            )
            if rule.conflict is Conflict.ERROR:
                raise ValueError(message)
            if rule.conflict is Conflict.WARN:
                logger.warning(message)
        return Resolved(winner.value, winner.source, tuple(disagreeing))

    def _is_autoregressive(self) -> bool:
        return _walk(self._doc(TRANSFORMER), _SIM) not in (_MISSING, None)

    def resolve(self) -> Cosmos3Config:
        rules = dict(RULES)
        autoregressive = self._is_autoregressive()
        if autoregressive:
            rules.update(AUTOREGRESSIVE_RULES)
        resolved: dict[str, Resolved] = {}
        for name, rule in rules.items():
            r = self.resolve_one(name, rule)
            if r is not None:
                resolved[name] = r

        def get(name: str) -> Any:
            r = resolved.get(name)
            return None if r is None else r.value

        sigmas = get("sampling.fixed_sigmas")
        if sigmas is not None and (
            any(not 0.0 < float(s) <= 1.0 for s in sigmas)
            or [float(s) for s in sigmas] != sorted((float(s) for s in sigmas), reverse=True)
        ):
            raise ValueError(
                f"Cosmos3 config: fixed_step_sampler_config.t_list must decrease within (0, 1]: "
                f"{list(sigmas)}."
            )
        chunking = kv_cache = None
        if autoregressive:
            if sigmas is None:
                raise ValueError(
                    "Cosmos3 config: a Sim checkpoint needs its fixed denoising schedule, "
                    "fixed_step_sampler_config.t_list; none is declared."
                )
            if not get("chunking.video_temporal_causal"):
                raise ValueError(
                    "Cosmos3 config: a Sim checkpoint must declare video_temporal_causal; "
                    "this one declares it false."
                )
            for name in (
                "chunking.chunk_frames",
                "prompt.text_cache_max_len",
                "action.tokens_per_frame",
            ):
                if int(get(name)) <= 0:
                    raise ValueError(f"Cosmos3 config: {name} must be positive, got {get(name)}.")
            chunk_frames = int(get("chunking.chunk_frames"))
            # The reference rollout cuts chunks [1, C, C, ...] and reads no partition
            # from the checkpoint; a declared one must agree with that.
            reference = (1, chunk_frames)
            partition = get("chunking.partition")
            if partition is None:
                resolved["chunking.partition"] = Resolved(
                    reference, "reference rollout [1, C, ...]"
                )
                partition = reference
            elif tuple(int(n) for n in partition) != reference:
                raise ValueError(
                    f"Cosmos3 config: chunking.partition {list(partition)} from "
                    f"{resolved['chunking.partition'].source} differs from the reference "
                    f"rollout's [1, {chunk_frames}, ...]."
                )
            chunking = Cosmos3ChunkingSettings(
                chunk_frames=chunk_frames,
                partition=tuple(int(n) for n in partition),
                video_temporal_causal=bool(get("chunking.video_temporal_causal")),
            )
            sinks = int(get("kv_cache.sink_frames"))
            if "kv_cache.inference_size" in resolved:
                size = get("kv_cache.inference_size")
            elif "kv_cache.window_frames" in resolved:
                window = get("kv_cache.window_frames")
                size = None if window is None else int(window) + sinks
                resolved["kv_cache.inference_size"] = Resolved(
                    size, resolved["kv_cache.window_frames"].source + " + sinks"
                )
            else:
                raise ValueError(
                    "Cosmos3 config: cannot resolve kv_cache.inference_size; the checkpoint "
                    "declares neither kv_cache_inference_size nor window_frames."
                )
            if size is None and sinks:
                raise ValueError(
                    f"Cosmos3 config: {sinks} attention sinks need a kv_cache_inference_size; "
                    "full history has nothing to keep them past."
                )
            if size is not None and not 0 <= sinks <= int(size) - 2:
                raise ValueError(
                    f"Cosmos3 config: {sinks} attention sinks leave no recent frame in a "
                    f"kv_cache_inference_size of {size}."
                )
            fraction = get("kv_cache.free_gpu_memory_fraction")
            kv_cache = Cosmos3KVCacheSettings(
                inference_size=None if size is None else int(size),
                sink_frames=sinks,
                free_gpu_memory_fraction=None if fraction is None else float(fraction),
            )
        text_cache = get("prompt.text_cache_max_len")
        tokens_per_frame = get("action.tokens_per_frame")
        input_dim = get("action.input_action_dim")
        default_embodiment = get("action.default_embodiment")
        return Cosmos3Config(
            architecture=Cosmos3ArchitectureSettings(
                position_embedding_type=str(get("architecture.position_embedding_type")),
                max_position_embeddings=int(get("architecture.max_position_embeddings")),
            ),
            vae=Cosmos3VAESettings(
                temporal_compression=int(get("vae.temporal_compression")),
                spatial_compression=int(get("vae.spatial_compression")),
            ),
            audio=Cosmos3AudioSettings(temporal_compression=int(get("audio.temporal_compression"))),
            sampling=Cosmos3SamplingSettings(
                fixed_sigmas=None if not sigmas else tuple(float(s) for s in sigmas),
                native_flow_schedule=bool(get("sampling.native_flow_schedule")),
            ),
            prompt=Cosmos3PromptSettings(
                use_system_prompt=bool(get("prompt.use_system_prompt")),
                text_cache_max_len=None if text_cache is None else int(text_cache),
            ),
            action=Cosmos3ActionSettings(
                policy=get("action.policy"),
                tokens_per_frame=None if tokens_per_frame is None else int(tokens_per_frame),
                embodiments=dict(get("action.embodiments") or {}),
                default_embodiment=None
                if default_embodiment is None
                else str(default_embodiment).lower(),
                input_action_dim=None if input_dim is None else int(input_dim),
            ),
            chunking=chunking,
            kv_cache=kv_cache,
            resolved=resolved,
        )


ARCHITECTURE_ATTRIBUTES = {
    "position_embedding_type": "architecture.position_embedding_type",
    "max_position_embeddings": "architecture.max_position_embeddings",
    "temporal_compression_factor": "vae.temporal_compression",
    "temporal_compression_factor_sound": "audio.temporal_compression",
}
"""``pretrained_config`` attributes the transformer reads, and the setting each holds."""


def architecture_defaults() -> dict[str, Any]:
    """The table's default for each attribute in :data:`ARCHITECTURE_ATTRIBUTES`."""
    out = {}
    for attr, name in ARCHITECTURE_ATTRIBUTES.items():
        out[attr] = next(s.value for s in RULES[name].sources if isinstance(s, Default))
    return out


def fill_pretrained_config(pretrained_config: Any) -> Any:
    """Set the transformer-construction attributes of ``pretrained_config`` that the
    export omits, from :data:`RULES`. Idempotent; returns the config.

    Runs before the pipeline loads its components, so only ``pretrained_config``
    itself is consulted.
    """
    resolver = Cosmos3ConfigResolver(None, pretrained_config)
    for attr, name in ARCHITECTURE_ATTRIBUTES.items():
        if getattr(pretrained_config, attr, None) is not None:
            continue
        rule = RULES[name]
        transformer_only = Rule(
            tuple(s for s in rule.sources if not isinstance(s, Key) or s.file == TRANSFORMER),
            rule.conflict,
            rule.required,
            rule.nullable,
        )
        r = resolver.resolve_one(name, transformer_only)
        if r is not None:
            setattr(pretrained_config, attr, r.value)
    return pretrained_config
