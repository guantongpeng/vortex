#!/usr/bin/env python3
"""Validate and render the versioned Vortex deep-learning config matrix."""

import argparse
import json
import pathlib
import shlex
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = ROOT / "ci" / "dl_config_matrix.json"
SHAPE_KEYS = ("clusters", "cores", "warps", "threads")
BOOL_KEYS = ("l2cache", "l3cache")


def known_config_names():
    """Return VX_CFG names declared by the repository config sources."""
    names = set()
    for path in (ROOT / "VX_config.toml", ROOT / "VX_types.toml"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("VX_CFG_"):
                names.add(stripped.split("=", 1)[0].strip())
    return names


def load_matrix(path):
    with path.open(encoding="utf-8") as stream:
        matrix = json.load(stream)
    if matrix.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    profiles = matrix.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles must be a non-empty list")

    names = set()
    config_names = known_config_names()
    for profile in profiles:
        name = profile.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("profile names must be non-empty and unique")
        names.add(name)
        xlens = profile.get("xlen")
        if not isinstance(xlens, list) or not xlens:
            raise ValueError("%s: xlen must be a non-empty list" % name)
        if any(x not in (32, 64) for x in xlens):
            raise ValueError("%s: xlen must contain only 32 or 64" % name)
        shape = profile.get("shape")
        if not isinstance(shape, dict):
            raise ValueError("%s: shape must be an object" % name)
        for key in SHAPE_KEYS:
            value = shape.get(key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError("%s: %s must be a positive integer" % (name, key))
        for key in BOOL_KEYS:
            if not isinstance(shape.get(key), bool):
                raise ValueError("%s: %s must be boolean" % (name, key))
        configs = profile.get("configs", [])
        if not isinstance(configs, list) or any(
            not isinstance(item, str) or not item.startswith("-D") for item in configs
        ):
            raise ValueError("%s: configs must contain -D strings" % name)
        for item in configs:
            config_name = item[2:].split("=", 1)[0]
            if config_name.startswith("VX_CFG_") and config_name not in config_names:
                raise ValueError("%s: unknown config name %s" % (name, config_name))
    return matrix


def profile(matrix, name):
    for item in matrix["profiles"]:
        if item["name"] == name:
            return item
    raise ValueError("unknown profile: %s" % name)


def render(item, xlen=None):
    shape = item["shape"]
    flags = ["--%s=%s" % (key, shape[key]) for key in SHAPE_KEYS]
    flags.extend("--%s" % key for key in BOOL_KEYS if shape[key])
    configs = list(item["configs"])
    return flags, configs


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=pathlib.Path, default=DEFAULT_MATRIX)
    parser.add_argument("--list", action="store_true", help="list profile names")
    parser.add_argument("--name", help="profile to render")
    parser.add_argument("--xlen", type=int, choices=(32, 64))
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    args = parser.parse_args(argv)

    try:
        matrix = load_matrix(args.matrix)
        if args.list:
            for item in matrix["profiles"]:
                print(item["name"])
            return 0
        if not args.name:
            parser.error("--name is required unless --list is used")
        item = profile(matrix, args.name)
        if args.xlen is not None and args.xlen not in item["xlen"]:
            raise ValueError("%s does not define xlen=%d" % (args.name, args.xlen))
        flags, configs = render(item, args.xlen)
        if args.format == "json":
            print(json.dumps({"name": args.name, "xlen": args.xlen,
                              "shape": item["shape"], "configs": configs},
                             sort_keys=True))
        else:
            if args.xlen is not None:
                print("XLEN=%d" % args.xlen)
            print("BLACKBOX_FLAGS=" + " ".join(shlex.quote(x) for x in flags))
            print("CONFIGS=" + " ".join(shlex.quote(x) for x in configs))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("dl_config: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
