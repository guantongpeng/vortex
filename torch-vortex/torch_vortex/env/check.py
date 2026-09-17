# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check the current environment against the supported-version table.

    python -m torch_vortex.env.check [--manifest PATH] [--allow-untested]

Exit codes: 0 supported, 1 unsupported, 2 untested (unless --allow-untested).
Turns "these are environment observations" into a machine-checked statement,
which is what W0.1 of docs/mydocs/pytorch_plan.md asks for. Deliberately has
no third-party dependency so it can run before torch is even importable.
"""

import argparse
import json
import os
import re
import sys

from . import manifest as manifest_mod

_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supported.json")
_FIELDS = ("torch", "python", "triton", "xlen", "driver")

EXIT_OK = 0
EXIT_UNSUPPORTED = 1
EXIT_UNTESTED = 2


def _version_tuple(text):
    """'2.14.0+cpu' -> (2, 14, 0). Non-numeric tails are dropped."""
    nums = re.findall(r"\d+", str(text).split("+")[0].split("-")[0])
    return tuple(int(n) for n in nums)


def _match_one(spec, value):
    if spec is None:
        return True
    for term in str(spec).split(","):
        term = term.strip()
        if not term:
            continue
        m = re.match(r"^(==|>=|<=|>|<)?\s*(.+)$", term)
        op, want = m.group(1) or "==", m.group(2).strip()
        if op == "==" and not re.match(r"^[\d.]+$", want):
            # Not a version at all (e.g. a driver name): compare as a string.
            if str(value) != want:
                return False
            continue
        got_t, want_t = _version_tuple(value), _version_tuple(want)
        n = max(len(got_t), len(want_t))
        got_t += (0,) * (n - len(got_t))
        want_t += (0,) * (n - len(want_t))
        ok = {"==": got_t == want_t, ">=": got_t >= want_t,
              "<=": got_t <= want_t, ">": got_t > want_t,
              "<": got_t < want_t}[op]
        if not ok:
            return False
    return True


def _matches(entry, observed):
    """An entry matches when every constraint it states matches.

    An entry with no real constraint is not allowed to match anything --
    otherwise a note-only row would classify every environment.
    """
    used = 0
    for field in _FIELDS:
        if field not in entry:
            continue
        used += 1
        if not _match_one(entry[field], observed.get(field)):
            return False
    return used > 0


def _observed(man):
    o = dict(man.get("observed", {}))
    b = man.get("build", {})
    o["xlen"] = b.get("xlen")
    o["driver"] = b.get("driver")
    return o


def evaluate(man, table=None):
    """Return (status, entry, detail) for a manifest."""
    if table is None:
        with open(_TABLE) as f:
            table = json.load(f)
    observed = _observed(man)

    for status in ("unsupported", "untested", "supported"):
        for entry in table.get(status, []):
            if _matches(entry, observed):
                reason = entry.get("reason") or entry.get("note")
                return status, entry, reason
    return "untested", {}, "no entry in the supported-version table matches"


def _load_manifest(args):
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest) as f:
            return json.load(f), args.manifest
    man, path = manifest_mod.write(path=args.manifest)
    return man, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=None,
                        help="manifest to read; written if it does not exist")
    parser.add_argument("--allow-untested", action="store_true",
                        help="treat an untested combination as a warning")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    man, path = _load_manifest(args)
    status, entry, detail = evaluate(man)

    if args.json:
        json.dump({"status": status, "entry": entry, "detail": detail,
                   "manifest": man, "manifest_path": path},
                  sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print("env manifest: %s" % path)
        print("  " + manifest_mod.summarize(man))
        print("status: %s" % status)
        if detail:
            print("  %s" % detail)

    if status == "unsupported":
        return EXIT_UNSUPPORTED
    if status == "untested" and not args.allow_untested:
        return EXIT_UNTESTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
