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

"""Session setup for the torch-vortex suite.

Every session records an environment manifest (W0.1) and checks it against
the supported-version table, so a result can always be traced to the
interpreter, torch build, build tree and kernel image it came from.

Tiers: the default (`smoke`) excludes tests marked `slow` — the model-level
parity runs. Use `--tier=full` to include them.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_addoption(parser):
    parser.addoption("--tier", default="smoke", choices=("smoke", "full"),
                     help="smoke: skip slow model tests (default); full: run all")
    parser.addoption("--allow-untested", action="store_true",
                     help="do not fail the session on an untested environment")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs --tier=full")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--tier") == "full":
        return
    skip = pytest.mark.skip(reason="needs --tier=full")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def backend():
    """Import the backend, record the manifest, check the environment."""
    import torch_vortex
    from torch_vortex.env import check, manifest

    man, path = manifest.write(ext_module=torch_vortex._ext)
    status, _entry, detail = check.evaluate(man)

    print("\n[torch-vortex] manifest: %s" % path)
    print("[torch-vortex] %s" % manifest.summarize(man))
    print("[torch-vortex] environment: %s%s"
          % (status, (" (%s)" % detail) if detail else ""))

    if status == "unsupported":
        pytest.fail("unsupported environment (%s); see %s" % (detail, path),
                    pytrace=False)
    return torch_vortex


@pytest.fixture()
def stats(backend):
    """Zeroed device counters, returned after the test body finishes."""
    backend.reset_stats()
    yield backend.stats
