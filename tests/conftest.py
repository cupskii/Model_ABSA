import os
import sys

# Pastikan root repo ada di sys.path agar import workflow.*, pipeline.*, model.* berjalan
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
