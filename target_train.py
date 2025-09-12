import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from simulib.simulation_functions import db
from sklearn.decomposition import KernelPCA

from config import get_config
from dataloaders import TargetEncoderModule
from models import TargetEmbedding
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


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
    gpu_num = 1
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(43, workers=True)

    target_config = get_config('target_exp', './vae_config.yaml')
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

        # Get one example from each target
        examples = []
        example_files = glob('/home/jeff/repo/apache/data/target_tensors/target_*')
        pcas = []
        for ex in example_files:
            pc0 = []
            if fls := glob(f'{ex}/target_*_*.pt'):
                pc0.extend(torch.load(f, weights_only=True) for f in fls)
                examples.append(torch.load(fls[0], weights_only=True))
            pcas.append(pc0)

        model.to('cuda:0')

        # A look at how the different targets are located in latent space
        enc_vecs = []
        for ptarg in pcas:
            enc0 = []
            enc0.extend(
                model.encode(p[0].unsqueeze(0).to(model.device))
                .squeeze(0)
                .cpu()
                .data.numpy()
                for p in ptarg
            )
            enc_vecs.append(enc0)
        # Stack everything together
        pca_stack = np.concatenate([e for e in enc_vecs if len(e) > 0])
        pca = KernelPCA(n_components=3, kernel='rbf')
        pca = pca.fit(pca_stack)

        ax = plt.subplot(111, projection='3d')
        for enc in enc_vecs:
            if len(enc) > 0:
                pca_vecs = pca.transform(enc)
                ax.scatter(pca_vecs[:, 0], pca_vecs[:, 1], pca_vecs[:, 2])

        # Reconstruction, less important but contains salient information
        for idx, example in enumerate(examples):
            reconstruction = model(example[0].unsqueeze(0).to(model.device)).squeeze(0).cpu().data.numpy()
            rec = (reconstruction[0] * target_config.var[0] + target_config.mu[0] + 1j * reconstruction[1] * target_config.var[1] + target_config.mu[1])
            example_data = example[0].cpu().data.numpy()
            ex = example_data[0] + 1j * example_data[1]
            plt.figure(f'Target {idx}')
            plt.subplot(221)
            plt.title('Original')
            plt.imshow(db(ex))
            plt.axis('tight')
            # plt.clim(-10, 10)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(222)
            plt.title('Reconstruction')
            plt.imshow(db(rec))
            plt.axis('tight')
            # plt.clim(-10, 10)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(223)
            plt.title('Phase Error')
            plt.imshow(np.angle(ex - rec))
            plt.axis('tight')
            # plt.clim(-10, 10)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(224)
            plt.title('Mag Error')
            plt.imshow(abs(ex - rec))
            plt.axis('tight')
            # plt.clim(-10, 10)
            plt.xticks([])
            plt.yticks([])

            plt.show()
