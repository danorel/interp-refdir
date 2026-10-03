import subprocess
import sys

# Runs in a subprocess: nnsight's `save` mount is process-wide and irreversible.
SCRIPT = """
import interptemp.models.nnterp_model  # pre-imports anyio's TypedAttributeSet subclasses
from nnsight.intervention.tracing.globals import _ensure_mounted
_ensure_mounted()  # what the first `.save()` in a trace does
assert "save" in dir(object), "mount did not happen; test no longer exercises the bug"

import anyio._backends._asyncio  # loaded lazily by the first async HTTP request
from openai import AsyncOpenAI
AsyncOpenAI(base_url="http://localhost", api_key="x")  # builds the httpx transport
"""


def test_http_client_works_after_nnsight_save_mount():
    proc = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
