"""Noiseprint++ backbone from TruFor (Guillaro et al., CVPR 2023).
Model and licence: https://github.com/grip-unina/TruFor
Only generic forensic weights are downloaded, never reference stamp images.
"""
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import torch
from torch import nn


def load_noiseprint(cache, fused=False):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'noiseprint_weights.pt'
    if not path.exists():
        archive = cache / 'TruFor_weights.zip'
        if not archive.exists():
            urllib.request.urlretrieve('https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip', archive)
        checkpoint_path = cache / 'trufor.pth.tar'
        if not checkpoint_path.exists():
            with zipfile.ZipFile(archive) as z, z.open('weights/trufor.pth.tar') as source, checkpoint_path.open('wb') as target:
                import shutil
                shutil.copyfileobj(source, target)
        torch.serialization.add_safe_globals([
            (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
            np.dtype, type(np.dtype('float64'))])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['state_dict']
        state = {k[6:]: v for k, v in checkpoint.items() if k.startswith('dncnn.')}
        torch.save(state, path)
    layers = []
    for i in range(17):
        batchnorm = 0 < i < 16
        layers.append(nn.Conv2d(3 if i == 0 else 64, 1 if i == 16 else 64,
            3, padding=1, bias=not batchnorm))
        if batchnorm:
            layers.append(nn.BatchNorm2d(64, momentum=.1))
        if i < 16:
            layers.append(nn.ReLU(inplace=True))
    model = nn.Sequential(*layers)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    if fused:
        model = torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.script(model)))
    return model


def extract_noise(model, image):
    a = image[:, :, ::-1].copy().astype(np.float32).transpose(2, 0, 1)[None] / 256
    with torch.inference_mode():
        return model(torch.from_numpy(a))[0, 0].numpy()
