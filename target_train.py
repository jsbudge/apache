from clearml import Task
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from simulib.simulation_functions import db
import yaml
from dataloaders import TargetEncoderModule
from models import init_weights, TargetEmbedding, PulseClassifier
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools


def loadModel(exp_params, gpu_num=1, fft_len=8192, load_data=True, model_type='target', log_dir=None, input_model=None):
    if load_data:
        data = TargetEncoderModule(**exp_params["dataset_params"])
        data.setup()
    else:
        data = None

    # Get the model, experiment, logger set up
    if model_type == 'target':
        model = TargetEmbedding(**exp_params['model_params'], fft_len=fft_len, params=exp_params)
        name = 'TargetEncoder'
    elif model_type == 'pulse':
        model = PulseClassifier(exp_params['model_params']['latent_dim'], exp_params['model_params']['label_sz'],
                                params=exp_params, embedding_model=input_model)
        name = 'PulseClassifier'
    print('Setting up model...')
    # Make sure that if we're just transforming data we don't accidentally transform using noise
    if exp_params['warm_start'] or (exp_params['transform_data'] and not exp_params['is_training']):
        print('Model loaded from save state.')
        try:
            model.load_state_dict(torch.load('./model/target_model.state'))
        except RuntimeError:
            print('Model save file does not match current structure. Re-running with new structure.')
            model.apply(init_weights)
    else:
        print('Initializing new model...')
        model.apply(init_weights)
    if exp_params['init_task'] and exp_params['is_training']:
        task = Task.init(project_name=name, task_name=exp_params['exp_name'])
    else:
        task = None

    if exp_params['is_training']:
        logger = loggers.TensorBoardLogger(log_dir, name=name)
        expected_lr = max((exp_params['LR'] *
                           exp_params['scheduler_gamma'] ** (exp_params['max_epochs'] *
                                                             exp_params['swa_start'])), 1e-9)
        trainer = Trainer(logger=logger, max_epochs=exp_params['max_epochs'],
                          log_every_n_steps=exp_params['log_epoch'], devices=[gpu_num], callbacks=
                          [EarlyStopping(monitor='train_loss', patience=exp_params['patience'],
                                         check_finite=True),
                           StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=exp_params['swa_start'])])
    else:
        trainer = None
    return trainer, model, data, task


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    gpu_num = 1
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(43, workers=True)

    with open('./vae_config.yaml') as y:
        param_dict = yaml.safe_load(y.read())

    exp_params = param_dict['target_exp_params']

    trainer, model, data, task = loadModel(exp_params, gpu_num, param_dict['settings']['fft_len'], True,'target', param_dict['train_params']['log_dir'])
    ptrainer, pmodel, _, _ = loadModel(param_dict['pulse_exp_params'], gpu_num, param_dict['settings']['fft_len'], False, 'pulse', param_dict['train_params']['log_dir'], input_model=model)

    if exp_params['is_training']:
        print("======= Training =======")
        try:
            trainer.fit(model, datamodule=data)
        except KeyboardInterrupt:
            if trainer.is_global_zero:
                print('Training interrupted.')
            else:
                print('adios!')
                exit(0)

    if param_dict['pulse_exp_params']['is_training']:
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
        if exp_params['save_model']:
            try:
                torch.save(model.state_dict(), './model/target_model.state')
                print('Model saved to disk.')
            except Exception as e:
                print(f'Model not saved: {e}')
        print('Plotting outputs...')
        plt.figure('Samples')
        plt.subplot(2, 1, 1)
        plt.title('Sample')
        plt.plot(db(sample[0] + 1j * sample[1]))
        plt.subplot(2, 1, 2)
        plt.title('Vector')
        plt.plot(embedding)

        # Snag the first 50 batches
        batch_sz = exp_params['dataset_params']['val_batch_size']
        embeddings = np.zeros((min(50, len(data.val_dataloader())) * batch_sz, exp_params['model_params']['latent_dim']))
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

        classified_pulses = pmodel(torch.Tensor(samples[shuffle_idxes]).to(pmodel.device)).cpu().data.numpy()
        file_pred = np.argmax(classified_pulses, axis=1)
        conf_sz = np.zeros((24, 24))
        for n, m in itertools.product(range(24), range(24)):
            conf_sz[n, m] = sum(np.logical_and(file_pred == n, file_idx[shuffle_idxes] == m))
        plt.figure('Pulse Confusion Matrix')
        plt.imshow(conf_sz)
        plt.colorbar()

        plt.show()

    if exp_params['transform_data']:
        print('Running data transformation of files...')
        save_path = param_dict['generate_data_settings']['local_path'] if (
            param_dict)['generate_data_settings']['use_local_storage'] else exp_params['dataset_params']['data_path']
        target_data = np.fromfile(f'{save_path}/targetpatterns.dat', np.float32).reshape((-1, 2, 256, 256))
        with open(
                f'{save_path}/targets.enc', 'wb') as writer:
            out_data = model.encode(torch.tensor(target_data)).data.numpy()
            out_data.tofile(writer)
    if exp_params['init_task'] and exp_params['is_training']:
        task.close()
