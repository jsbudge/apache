from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
from tqdm import tqdm
from models import TargetEmbedding
from config import get_config
import torch
import os
import matplotlib as mplib
mplib.use('TkAgg')
import yaml

pio.renderers.default = 'browser'


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

    plt.figure()
    for t_dir in os.listdir(root_path):
        if 'target' in t_dir:
            files = glob(f'{root_path}/{t_dir}/*.pt')
            for f in tqdm(files):
                sample, label = torch.load(f)
                emb = embedding.encode(sample.unsqueeze(0).to(embedding.device)).squeeze(0).cpu().data.numpy()
                tmeans[label] += emb / len(files)
                plt.plot(emb)

    torch.save(tmeans, f'{root_path}/target_embedding_means.pt')

    plt.figure()
    for m in tmeans[:23]:
        plt.plot(m)
    plt.show()