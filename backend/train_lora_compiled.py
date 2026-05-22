"""Wrapper around SA3's train_lora.py that applies torch.compile to the DIT.

Imports SA3's training code directly and patches the training wrapper's __init__
to compile the DIT before PL starts training.
"""
import os
import sys

sa3_root = os.environ.get("SA3_ROOT", os.path.expanduser("~/projects/stable-audio-3"))
if sa3_root not in sys.path:
    sys.path.insert(0, sa3_root)

import torch
from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

_orig_init = DiffusionCondTrainingWrapper.__init__

def _compiled_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    if hasattr(self, 'model') and hasattr(self.model, 'model'):
        print("[compile] Compiling DIT with torch.compile(mode='default')...")
        self.model.model = torch.compile(self.model.model, mode="default")
        print("[compile] DIT compiled")

DiffusionCondTrainingWrapper.__init__ = _compiled_init

exec(open(os.path.join(sa3_root, "scripts", "train_lora.py")).read())
