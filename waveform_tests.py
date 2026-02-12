import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from simulib.simulation_functions import db
from sklearn.decomposition import KernelPCA
from config import get_config
from dataloaders import TargetEncoderModule, ClutterEncoderModule
from models import TargetEmbedding, ClutterTransformer
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


def setupTrainer(a_gpu_num, tconf, do_logs=True, **trainer_args):

    enc_data_module = ClutterEncoderModule(**tconf.dataset_params)
    enc_data_module.setup()

    # Get the model, experiment, logger set up
    if tconf.warm_start:
        mdl = ClutterTransformer.load_from_checkpoint(
            f'{tconf.weights_path}/{tconf.model_name}.ckpt', strict=False)
    else:
        mdl = ClutterTransformer(**tconf.model_params, **tconf.training_params)  # input_dim=8192, model_dim=1256, num_layers=20, lr=1e-0, warmup=100, max_iters=20000)
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

    target_config = get_config('transformer_exp', './vae_config.yaml')
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

    if trainer.is_global_zero:
        if target_config.save_model:
            trainer.save_checkpoint(f'{target_config.weights_path}/{target_config.model_name}.ckpt')
            print('Checkpoint saved.')
        # import matplotlib as mplib
        # mplib.use('TkAgg')
        model.to(device)
        model.eval()

        data_iter = iter(data.val_dataloader())
        clutter = next(data_iter)
        clutter_numpy = clutter.cpu().numpy()

        clutter = clutter.to(device)
        check = model(clutter).detach().cpu().numpy()

        plt.figure()
        plt.plot(clutter_numpy[0, -1, 0])
        plt.plot(check[0, -2, 0])
        plt.show()


