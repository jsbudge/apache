import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from simulib.simulation_functions import db
import yaml
from dataloaders import TargetEncoderModule
from models import TargetEmbedding, PulseClassifier, load
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    gpu_num = 1
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(43, workers=True)

    with open('./vae_config.yaml') as y:
        param_dict = yaml.safe_load(y.read())

    target_exp_params = param_dict['target_exp_params']
    pulse_exp_params = param_dict['pulse_exp_params']
    fft_len = param_dict['settings']['fft_len']

    data = TargetEncoderModule(**target_exp_params["dataset_params"])
    data.setup()

    # Get the model, experiment, logger set up
    if target_exp_params['warm_start']:
        model = load(TargetEmbedding, './model/current_te_params.pic')
    else:
        model = TargetEmbedding(**target_exp_params['model_params'], fft_len=fft_len, params=target_exp_params)
    name = 'TargetEncoder'

    if target_exp_params['is_training']:
        logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'], name=name)
        expected_lr = max((target_exp_params['LR'] *
                           target_exp_params['scheduler_gamma'] ** (target_exp_params['max_epochs'] *
                                                             target_exp_params['swa_start'])), 1e-9)
        trainer = Trainer(logger=logger, max_epochs=target_exp_params['max_epochs'],
                          log_every_n_steps=target_exp_params['log_epoch'], devices=[gpu_num], callbacks=
                          [EarlyStopping(monitor='train_loss', patience=target_exp_params['patience'],
                                         check_finite=True),
                           StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=target_exp_params['swa_start'])])
        print("======= Training =======")
        try:
            trainer.fit(model, datamodule=data)
        except KeyboardInterrupt:
            if trainer.is_global_zero:
                print('Training interrupted.')
            else:
                print('adios!')
                exit(0)

    if pulse_exp_params['warm_start']:
        pmodel = load(PulseClassifier, './model/current_pc_params.pic')
    else:
        pmodel = PulseClassifier(pulse_exp_params['model_params']['latent_dim'], pulse_exp_params['model_params']['label_sz'],
                        params=pulse_exp_params, embedding_model=model)
    name = 'PulseClassifier'
    if param_dict['pulse_exp_params']['is_training']:
        logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'], name=name)
        expected_lr = max((pulse_exp_params['LR'] *
                           pulse_exp_params['scheduler_gamma'] ** (pulse_exp_params['max_epochs'] *
                                                                    pulse_exp_params['swa_start'])), 1e-9)
        ptrainer = Trainer(logger=logger, max_epochs=pulse_exp_params['max_epochs'],
                          log_every_n_steps=pulse_exp_params['log_epoch'], devices=[gpu_num], callbacks=
                          [EarlyStopping(monitor='train_loss', patience=pulse_exp_params['patience'],
                                         check_finite=True),
                           StochasticWeightAveraging(swa_lrs=expected_lr,
                                                     swa_epoch_start=pulse_exp_params['swa_start'])])
        print("======= Training Pulse Classifier =======")
        try:
            ptrainer.fit(pmodel, datamodule=data)
        except KeyboardInterrupt:
            if ptrainer.is_global_zero:
                print('Training interrupted.')
            else:
                print('adios!')
                exit(0)

    if ptrainer.is_global_zero:
        model.to(device)
        model.eval()
        sample = next(iter(data.val_dataloader()))
        embedding = model(sample[0].to(device))[0].cpu().data.numpy()
        sample = sample[0][0].data.numpy()
        print('Plotting outputs...')
        plt.figure('Samples')
        plt.subplot(2, 1, 1)
        plt.title('Sample')
        plt.plot(np.fft.fftshift(np.fft.fftfreq(8192, 1 / 2e9) / 1e6), db(sample[0] + 1j * sample[1]))
        plt.xlabel('Freq (MHz)')
        plt.ylabel('Power (dB)')
        plt.subplot(2, 1, 2)
        plt.title('Vector')
        plt.xlabel('Element')
        plt.plot(embedding)

        # Snag the first 50 batches
        batch_sz = target_exp_params['dataset_params']['val_batch_size']
        embeddings = np.zeros((min(50, len(data.val_dataloader())) * batch_sz, target_exp_params['model_params']['latent_dim']))
        samples = np.zeros((min(50, len(data.val_dataloader())) * batch_sz, 2, 8192))
        file_idx = np.zeros(min(50, len(data.val_dataloader())) * batch_sz)
        val_gen = iter(data.val_dataloader())
        for i, sam in tqdm(enumerate(val_gen)):
            samples[i * batch_sz:(i + 1) * batch_sz, ...] = sam[0]
            embeddings[i * batch_sz:(i + 1) * batch_sz, :] = model(sam[0].to(device)).cpu().data.numpy()
            file_idx[i * batch_sz:(i + 1) * batch_sz] = sam[1]
            if i >= 49:
                break
        svd_t = KernelPCA(kernel='rbf', n_components=3).fit_transform(embeddings)

        ax = plt.figure('Embedding Distances').add_subplot(projection='3d')
        ax.scatter(svd_t[:, 0], svd_t[:, 1], svd_t[:, 2], c=file_idx / file_idx.max())
        model.to('cpu')

        shuffle_idxes = np.random.permutation(np.arange(len(file_idx)))
        pmodel.to(device)

        classifications = np.zeros(samples.shape[0])
        for block in tqdm(range(0, samples.shape[0], 256)):
            classifications[block:block + 256] = np.argmax(pmodel(torch.Tensor(samples[shuffle_idxes[block:block + 256]]).to(pmodel.device)).cpu().data.numpy(), axis=1)
        conf_sz = np.zeros((batch_sz // 2, batch_sz // 2))
        for n, m in itertools.product(range(batch_sz // 2), range(batch_sz // 2)):
            conf_sz[n, m] = sum(np.logical_and(classifications == n, file_idx[shuffle_idxes] == m))
        plt.figure('Pulse Confusion Matrix')
        plt.imshow(conf_sz)
        plt.colorbar()

        plt.show()
