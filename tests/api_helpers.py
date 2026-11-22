import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fixture(name):
    return json.loads((ROOT / "src/predictive_maintenance/api/examples" / (name + ".json")).read_text(encoding="utf-8"))


class FakeRegistry:
    release_version = "test-release"

    def __init__(self, root=None):
        pass

    def model_version(self, capability):
        return "test-model"

    def describe(self):
        return {"dataset": "FD001"}

    def predict(self, raw):
        return {"rul_cycles": 62.5, "rul_cap_cycles": 125.0, "warnings": []}
