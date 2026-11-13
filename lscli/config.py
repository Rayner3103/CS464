"""`config.toml` (spec 3.1): defaults, loading, validation, `lscli config KEY VALUE`, and the two documents derived
from it for processes that must not parse TOML (the client may run on Python 3.10, which has no `tomllib`):
`client.json` (client settings + the environment fields a winner is keyed on).

Every key has a default and a type; an unknown key or a value of the wrong type is an error, never silently ignored.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

DEFAULTS: dict[str, dict] = {
    "idle": {"window": "01:00-07:00", "gpu_idle_minutes": 10, "gpu_util_max": 5, "require_ac": True,
             "backoff_minutes": 10},
    "search": {"k": 3, "k_frequent": 5, "frequent_calls": 1_000_000, "lt_n": 16, "lt_rank": 5, "half_life_days": 30.0},
    "client": {"dispatch": True, "inline_search": True, "min_kernel_us": 50, "min_saving_us": 5},
    "clock": {"lock_mhz": 0},
    "train": {"enabled": True, "mode": "full"},
}
CHOICES = {"train.mode": ("full", "quick")}
_WINDOW = re.compile(r"^$|^([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d$")


class ConfigError(ValueError):
    pass


def _check(key: str, value):
    sec, name = key.split(".", 1)
    want = type(DEFAULTS[sec][name])
    if want is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if want is int and isinstance(value, bool) or not isinstance(value, want):
        raise ConfigError(f"{key}: expected {want.__name__}, got {value!r}")
    if key in CHOICES and value not in CHOICES[key]:
        raise ConfigError(f"{key}: one of {', '.join(CHOICES[key])}, got {value!r}")
    if key == "idle.window" and not _WINDOW.fullmatch(value):
        raise ConfigError(f"idle.window: 'HH:MM-HH:MM' or '', got {value!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
        raise ConfigError(f"{key}: must be >= 0, got {value!r}")
    if key == "idle.gpu_util_max" and value > 100:
        raise ConfigError(f"{key}: a percentage, got {value!r}")
    return value


def validate(cfg: dict) -> dict:
    out = copy.deepcopy(DEFAULTS)
    for sec, vals in cfg.items():
        if sec not in DEFAULTS or not isinstance(vals, dict):
            raise ConfigError(f"unknown section [{sec}]")
        for name, v in vals.items():
            if name not in DEFAULTS[sec]:
                raise ConfigError(f"unknown key {sec}.{name}")
            out[sec][name] = _check(f"{sec}.{name}", v)
    s = out["search"]
    if not 1 <= s["k"] <= s["k_frequent"]:
        raise ConfigError(f"search: need 1 <= k <= k_frequent (k {s['k']}, k_frequent {s['k_frequent']}; "
                          "raise k_frequent first)")
    return out


def load(path: Path | None = None) -> dict:
    """Defaults overlaid with the file (a missing file = defaults)."""
    from lscli.paths import config_path
    path = Path(path) if path else config_path()
    if not path.exists():
        return copy.deepcopy(DEFAULTS)
    import tomllib
    try:
        with open(path, "rb") as f:
            return validate(tomllib.load(f))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None


def parse_value(key: str, text: str):
    """A command-line string -> the key's type ("true"/"false" for booleans)."""
    if "." not in key or key.split(".", 1)[0] not in DEFAULTS or key.split(".", 1)[1] not in DEFAULTS[key.split(".", 1)[0]]:
        raise ConfigError(f"unknown key {key}")
    sec, name = key.split(".", 1)
    want = type(DEFAULTS[sec][name])
    try:
        if want is float:
            return _check(key, float(text))
        if want is bool:
            if text.lower() not in ("true", "false"):
                raise ValueError
            return _check(key, text.lower() == "true")
        return _check(key, want(text))
    except ValueError:
        raise ConfigError(f"{key}: expected {want.__name__}, got {text!r}") from None


def dumps(cfg: dict) -> str:
    """TOML for a validated config (scalars only, so a small writer is enough)."""
    out = []
    for sec, vals in cfg.items():
        out.append(f"[{sec}]")
        for name, v in vals.items():
            out.append(f"{name} = {str(v).lower() if isinstance(v, bool) else json.dumps(v)}")
        out.append("")
    return "\n".join(out)


def set_value(key: str, text: str, path: Path | None = None) -> dict:
    """Set one key, validate the whole config, and replace the file atomically (comments in the file are not kept: it
    is rewritten from the validated values). Returns the new config."""
    import os
    import uuid
    from lscli.paths import config_path
    path = Path(path) if path else config_path()
    value = parse_value(key, text)               # rejects unknown keys before anything is read or written
    cfg = load(path)
    sec, name = key.split(".", 1)
    cfg[sec][name] = value
    cfg = validate(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:12]}")
    tmp.write_text(dumps(cfg))
    os.replace(tmp, path)
    return cfg


def get(cfg: dict, key: str):
    sec, _, name = key.partition(".")
    if sec not in cfg or (name and name not in cfg[sec]):
        raise ConfigError(f"unknown key {key}")
    return cfg[sec][name] if name else cfg[sec]


def clock_policy(cfg: dict) -> str:
    """The clock-lock value winners are keyed on: the daemon's measurement policy, not the clock a program runs at."""
    return str(cfg["clock"]["lock_mhz"]) if cfg["clock"]["lock_mhz"] else "unlocked"


def env_extra(cfg: dict, models: dict | None = None) -> dict:
    """Fields lscli adds to `tune.winner_cache.environment()` (spec 3.6). `models` is None until a model set is active."""
    return {"clock_lock": clock_policy(cfg), **({"models": models} if models is not None else {})}


def client_document(cfg: dict, models: dict | None = None, gpu_uuid: str | None = None,
                    model_stems: dict | None = None) -> dict:
    """`client.json`: what the client needs without parsing TOML or importing the daemon: its settings, the search
    widths for an inline search, the environment fields winners are keyed on, the daemon's GPU, and where the active
    models live (for an inline search, when lightgbm is available in the user's env)."""
    return {"client": cfg["client"], "search": {k: cfg["search"][k] for k in ("k", "k_frequent", "frequent_calls", "lt_n",
                                                                                "lt_rank", "half_life_days")},
            "env_extra": env_extra(cfg, models), "gpu_uuid": gpu_uuid, "models": model_stems or {}}


def write_client_json(paths, cfg: dict, **kw) -> None:
    from lscli.files import write_json
    write_json(paths.client, client_document(cfg, **kw))
