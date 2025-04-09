from glob import glob
import os
import numpy as np
from pathlib import Path
from numba import cuda
from simulib.mesh_objects import Scene, Mesh
from simulib.simulation_functions import db, azelToVec, genChirp, getElevation, enu2llh
from simulib.mesh_functions import readCombineMeshFile, getRangeProfileFromScene, _float
from simulib.platform_helper import SDRPlatform
from scipy.signal.windows import taylor
import matplotlib.pyplot as plt
import plotly.io as pio
from tqdm import tqdm
import yaml
from models import TargetEmbedding, load
from config import get_config
from sdrparse.SDRParsing import load, loadXMLFile
import torch

from utils import scale_normalize

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
TARGET_PROFILE_MIN_BEAMWIDTH = 0.19634954
fs = 2e9


if __name__ == '__main__':
    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    target_config = get_config('target_exp', './vae_config.yaml')

    root_path = f"{config['target_exp_params']['dataset_params']['data_path']}"
    root_dirs = os.listdir(root_path)
    tmeans = [np.zeros(target_config.latent_dim) for r in root_dirs if 'target_' in r]
    embedding = TargetEmbedding.load_from_checkpoint(f'{target_config.weights_path}/{target_config.model_name}.ckpt',
                                                     config=target_config, strict=False)
    embedding.eval()
    embedding.to('cuda')

    for t_dir in os.listdir(root_path):
        if 'target' in t_dir:
            files = glob(f'{root_path}/{t_dir}/*.pt')
            for f in tqdm(files):
                sample, label = torch.load(f)
                tmeans[label] += embedding.encode(sample.unsqueeze(0).to(embedding.device)).squeeze(0).cpu().data.numpy() / len(files)

    torch.save(tmeans, f'{root_path}/target_embedding_means.pt')