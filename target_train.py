from clearml import Task
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from simulib.simulation_functions import db
import yaml
from dataloaders import TargetEncoderModule
from models import init_weights, TargetEmbedding
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(43, workers=True)
    
    with open('./vae_config.yaml') as y:
        param_dict = yaml.safe_load(y.read())
    
    exp_params = param_dict['target_exp_params']
    
    data = TargetEncoderModule(**exp_params["dataset_params"])
    data.setup()
    
    # Get the model, experiment, logger set up
    model = TargetEmbedding(**exp_params['model_params'], fft_len=param_dict['settings']['fft_len'], params=exp_params)
    print('Setting up model...')
    tag_warm = 'new_model'
    # Make sure that if we're just transforming data we don't accidentally transform using noise
    if exp_params['warm_start'] or (exp_params['transform_data'] and not exp_params['is_training']):
        print('Model loaded from save state.')
        try:
            model.load_state_dict(torch.load('./model/target_model.state'))
            tag_warm = 'warm_start'
        except RuntimeError:
            print('Model save file does not match current structure. Re-running with new structure.')
            model.apply(init_weights)
    else:
        print('Initializing new model...')
        model.apply(init_weights)
    if exp_params['init_task'] and exp_params['is_training']:
        task = Task.init(project_name='TargetEncoder', task_name=exp_params['exp_name'])

    if exp_params['is_training']:
        logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'], name="TargetEncoder")
        expected_lr = max((exp_params['LR'] *
                           exp_params['scheduler_gamma'] ** (exp_params['max_epochs'] *
                                                             exp_params['swa_start'])), 1e-9)
        trainer = Trainer(logger=logger, max_epochs=exp_params['max_epochs'],
                          log_every_n_steps=exp_params['log_epoch'], devices=[1], callbacks=
                          [EarlyStopping(monitor='train_loss', patience=exp_params['patience'],
                                         check_finite=True),
                           StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=exp_params['swa_start'])])
        # trainer.test(model, train_loader, verbose=True)

        print("======= Training =======")
        try:
            trainer.fit(model, datamodule=data)
        except KeyboardInterrupt:
            if trainer.is_global_zero:
                print('Training interrupted.')
            else:
                print('adios!')
                exit(0)

        model.to(device)
        model.eval()
        sample = next(iter(data.val_dataloader()))
        embedding = model(sample[0].to(device))[0].cpu().data.numpy()
        sample = sample[0][0].data.numpy()

        if trainer.is_global_zero:
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
            file_idx = np.zeros(min(50, len(data.val_dataloader())) * batch_sz)
            val_gen = iter(data.val_dataloader())
            for i, sam in tqdm(enumerate(val_gen)):
                embeddings[i * batch_sz:(i + 1) * batch_sz, :] = model(sam[0].to(device)).cpu().data.numpy()
                file_idx[i * batch_sz:(i + 1) * batch_sz] = sam[1]
                if i >= 49:
                    break
            svd_t = KernelPCA(kernel='rbf', n_components=3).fit_transform(embeddings)

            ax = plt.figure('Embedding Distances').add_subplot(projection='3d')
            ax.scatter(svd_t[:, 0], svd_t[:, 1], svd_t[:, 2], c=file_idx / file_idx.max())
            model.to('cpu')

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
