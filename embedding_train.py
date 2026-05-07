from config import get_config
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataloaders import TargetEncoderModule
from models import TargetEmbedding
from sklearn.decomposition import PCA
import pickle


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))



if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()

    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(107, workers=True)

    config = get_config('embedding_exp', './vae_config.yaml')

    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)
    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config.dataset_params['max_pulse_length'] = nr
    config.dataset_params['min_pulse_length'] = 1000

    data = TargetEncoderModule(device=device, **config.dataset_params)
    data.setup()

    print('Initializing embedding...')
    if config.warm_start:
        embedding = TargetEmbedding.load_from_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt', config=config, strict=False)
    else:
        embedding = TargetEmbedding(**config.model_params, **config.training_params)
    logger = loggers.TensorBoardLogger(config.log_dir,
                                       name=config.model_name, log_graph=False)
    expected_lr = 1e-6
    if config.distributed:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, num_sanity_val_steps=0, default_root_dir=config.weights_path,
                          log_every_n_steps=config.log_epoch, check_val_every_n_epoch=1000, devices=[0, 1], strategy='ddp', callbacks=
                          [EarlyStopping(monitor='train_rec_loss', patience=config.patience, check_finite=True),
                           ModelCheckpoint(monitor='train_rec_loss_epoch')])
    else:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, num_sanity_val_steps=0,
                          default_root_dir=config.weights_path, check_val_every_n_epoch=1000,
                          log_every_n_steps=config.log_epoch, devices=[0], callbacks=
                          [EarlyStopping(monitor='train_rec_loss', patience=config.patience, check_finite=True),
                           ModelCheckpoint(monitor='train_rec_loss_epoch')])
    print("======= Training =======")
    try:
        trainer.fit(embedding, datamodule=data)
    except KeyboardInterrupt:
        if trainer.is_global_zero:
            print('Training interrupted.')
        else:
            print('adios!')
            exit(0)

    if trainer.global_rank == 0:
        if config.save_model:
            trainer.save_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt')
            print('Checkpoint saved.')

        with torch.no_grad():
            embedding.to(device)
            embedding.eval()
            data_iter = iter(data.train_dataloader())

            # plt.figure()
            colors = ['r', 'b', 'g', 'c', 'm', 'y']
            anch = []
            poses = []
            idxes = []
            for anchor, pos, neg, tidx in data_iter:
                anchor = anchor.to(device)
                pos = pos.to(device)
                # neg = neg.to(device)

                anch.append(embedding.embed(anchor).data.cpu().numpy())
                poses.append(embedding.embed(pos).data.cpu().numpy())
                idxes.append([np.where(tidx[0].data.cpu().numpy())[0][0]])
                # n_emb = embedding(neg).data.cpu().numpy()


                # plt.scatter(a_emb[:, 0], a_emb[:, 1], c=colors[tidx])

            pca = PCA(n_components=3).fit_transform(np.concatenate(anch + poses))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=np.array(idxes + idxes)[:, 0].astype(int))

            plt.figure('Anchor vs. Pos')
            plt.subplot(2, 1, 1)
            plt.title('Anchor')
            plt.imshow(abs(anchor.data.cpu().numpy()[0, 0]))
            plt.axis('tight')
            plt.subplot(2, 1, 2)
            plt.title('Pos')
            plt.imshow(abs(pos.data.cpu().numpy()[0, 0]))
            plt.axis('tight')

            groups = {}
            for key, val in zip(np.array(idxes)[:, 0], anch):
                groups.setdefault(key, []).append(val)

            for key, val in groups.items():
                groups[key] = np.concatenate(val, axis=0).mean(axis=0)

            with open('/home/jeff/repo/apache/data/target_new/embeddings.pic', 'wb') as f:
                pickle.dump(groups, f)
                print('Embeddings saved.')

