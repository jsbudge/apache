from target_train import loadModel
from glob import glob
import os
import numpy as np
from pathlib import Path
from simulib.simulation_functions import db, azelToVec, genChirp, getElevation
from simulib.cuda_mesh_kernels import readCombineMeshFile, getBoxesSamplesFromMesh, getRangeProfileFromMesh
from simulib.platform_helper import SDRPlatform
from simulib.cuda_kernels import cpudiff, getMaxThreads
from generate_trainingdata import formatTargetClutterData
from scipy.signal.windows import taylor
import matplotlib.pyplot as plt
import plotly.io as pio
from tqdm import tqdm
import yaml
from data_converter.SDRParsing import load
import torch
from pytorch_lightning import seed_everything
import cupy

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    gpu_num = 1
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(43, workers=True)

    with open('./vae_config.yaml') as y:
        param_dict = yaml.safe_load(y.read())

    # First, get a representation of the targets from the embedding model
    exp_params = param_dict['target_exp_params']

    trainer, model, data, task = loadModel(exp_params, gpu_num, param_dict['settings']['fft_len'], True, 'target', param_dict['train_params']['log_dir'])
    model.to(device)
    model.eval()

    ntargets = exp_params['dataset_params']['train_batch_size'] // 2

    embeddings = np.zeros((ntargets, exp_params['model_params']['latent_dim']))
    print('Iterating train dataloader...')
    val_gen = iter(data.train_dataloader())
    for i, sam in tqdm(enumerate(val_gen)):
        single_emb = model(sam[0].to(device)).cpu().data.numpy()
        embeddings += single_emb[:ntargets]
        embeddings += single_emb[ntargets:]

    print('Iterating val dataloader...')
    val_gen = iter(data.val_dataloader())
    nemb = i + 1
    for i, sam in tqdm(enumerate(val_gen)):
        single_emb = model(sam[0].to(device)).cpu().data.numpy()
        embeddings += single_emb[:ntargets]
        embeddings += single_emb[ntargets:]

    embeddings /= (nemb + i + 1) * 2

    model.to('cpu')

    save_path = param_dict['generate_data_settings']['local_path'] if (
        param_dict)['generate_data_settings']['use_local_storage'] else param_dict['dataset_params']['data_path']
    torch.save([torch.tensor(embeddings, dtype=torch.float32)],
               f"{save_path}/target_embedding_means.pt")


    # Now, load the clutter and target spec files for embedding
    glob(f'{save_path}/clutter_*.spec')

