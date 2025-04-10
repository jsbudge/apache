import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from simulib.simulation_functions import db

from config import get_config
from dataloaders import TargetEncoderModule
from models import TargetEmbedding, PulseClassifier
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def setupTrainer(a_gpu_num, tconf, do_logs=True, **trainer_args):


    enc_data_module = TargetEncoderModule(**tconf.dataset_params)
    enc_data_module.setup()

    # Get the model, experiment, logger set up
    if tconf.warm_start:
        mdl = TargetEmbedding.load_from_checkpoint(
            f'{tconf.weights_path}/{tconf.model_name}.ckpt', strict=False)
    else:
        mdl = TargetEmbedding(tconf)
    if do_logs:
        log_mod = loggers.TensorBoardLogger(tconf.log_dir, name=tconf.model_name)
    else:
        log_mod = None
    ret_train = Trainer(logger=log_mod, max_epochs=tconf.max_epochs,
                        default_root_dir=tconf.weights_path,
                        log_every_n_steps=tconf.log_epoch, detect_anomaly=False, devices=[a_gpu_num], **trainer_args)
    return ret_train, mdl, enc_data_module



if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    gpu_num = 0
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(43, workers=True)

    target_config = get_config('target_exp', './vae_config.yaml')
    classifier_config = get_config('pulse_exp', './vae_config.yaml')
    trainer, model, data = setupTrainer(gpu_num, target_config)

    # Get the model, experiment, logger set up
    if target_config.is_training:
        print("======= Training =======")
        try:
            trainer.fit(model, datamodule=data)
        except KeyboardInterrupt:
            if trainer.is_global_zero:
                print('Training interrupted.')
            else:
                print('adios!')
                exit(0)
        if target_config.save_model:
            trainer.save_checkpoint(f'{target_config.weights_path}/{target_config.model_name}.ckpt')

    if trainer.is_global_zero:
        import matplotlib as mplib
        mplib.use('TkAgg')
        model.to(device)
        model.eval()
        '''sample = next(iter(data.val_dataloader()))
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
        plt.plot(embedding)'''

        # Snag the first 50 batches
        batch_sz = target_config.val_batch_size
        embeddings = []
        samples = []
        file_idx = []
        val_gen = iter(data.val_dataloader())
        for i, sam in tqdm(enumerate(val_gen)):
            samples.append(sam[0])
            embeddings.append(model.encode(sam[0].to(model.device)).cpu().data.numpy())
            file_idx.append(sam[1])
            if i >= batch_sz - 1:
                break
        embeddings = np.concatenate(embeddings, axis=0)
        file_idx = np.concatenate(file_idx)
        svd_t = KernelPCA(kernel='rbf', n_components=3).fit_transform(embeddings)

        ax = plt.figure('Embedding Distances').add_subplot(projection='3d')
        ax.scatter(svd_t[:, 0], svd_t[:, 1], svd_t[:, 2], c=file_idx / file_idx.max())
        model.to('cpu')

        plt.figure()
        plt.plot(embeddings[::batch_sz].T)

        example = torch.load('/home/jeff/repo/apache/data/target_tensors/target_2/target_2_24.pt', weights_only=True)
        example_data = example[0].cpu().data.numpy()
        plt.figure()
        plt.imshow(db(example_data[0] + 1j * example_data[1]))
        plt.axis('tight')
        plt.clim(-10, 10)
        plt.xlabel('Range Bin')
        plt.yticks([])

        plt.show()


    '''if classifier_config.is_training:
        pmodel = PulseClassifier(config=classifier_config, embedding_model=model)
        logger = loggers.TensorBoardLogger(classifier_config.log_dir, name=classifier_config.model_name)
        expected_lr = max((classifier_config.lr * classifier_config.scheduler_gamma ** (classifier_config.max_epochs *
                                                                                classifier_config.swa_start)), 1e-9)
        ptrainer = Trainer(logger=logger, max_epochs=classifier_config.max_epochs, default_root_dir=classifier_config.weights_path,
                          log_every_n_steps=classifier_config.log_epoch, devices=[gpu_num], callbacks=
                          [EarlyStopping(monitor='train_loss', patience=classifier_config.patience,
                                         check_finite=True),
                           StochasticWeightAveraging(swa_lrs=expected_lr,
                                                     swa_epoch_start=classifier_config.swa_start),
                           ModelCheckpoint(monitor='train_loss')])
        print("======= Training Pulse Classifier =======")
        try:
            if classifier_config.warm_start:
                ptrainer.fit(pmodel, ckpt_path=f'{classifier_config.weights_path}/{classifier_config.model_name}.ckpt',
                            datamodule=data)
            else:
                ptrainer.fit(pmodel, datamodule=data)
        except KeyboardInterrupt:
            if ptrainer.is_global_zero:
                print('Training interrupted.')
            else:
                print('adios!')
                exit(0)
        if classifier_config.save_model:
            ptrainer.save_checkpoint(f'{classifier_config.weights_path}/{classifier_config.model_name}.ckpt')
    else:
        pmodel = PulseClassifier.load_from_checkpoint(f'{classifier_config.weights_path}/{classifier_config.model_name}.ckpt')

    if ptrainer.is_global_zero or not target_config.is_training:

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
        batch_sz = target_config.val_batch_size
        embeddings = np.zeros((min(50, len(data.val_dataloader())) * batch_sz, target_config.latent_dim))
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

        plt.show()'''
