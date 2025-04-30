import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from simulib.simulation_functions import db
from config import get_config
from dataloaders import TargetEncoderModule
from models import TargetEmbedding
import matplotlib.pyplot as plt
import numpy as np


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

        # Snag the first 50 batches
        batch_sz = target_config.val_batch_size

        example = torch.load('/home/jeff/repo/apache/data/target_tensors/target_3/target_3_34.pt', weights_only=True)
        model.to('cuda:0')
        reconstruction = model(example[0].unsqueeze(0).to(model.device)).squeeze(0).cpu().data.numpy()
        example_data = example[0].cpu().data.numpy()
        plt.figure()
        plt.subplot(121)
        plt.title('Original')
        plt.imshow(db(example_data[0] + 1j * example_data[1]))
        plt.axis('tight')
        # plt.clim(-10, 10)
        plt.xlabel('Range Bin')
        plt.yticks([])
        plt.subplot(122)
        plt.title('Reconstruction')
        plt.imshow(db(reconstruction[0] + 1j * reconstruction[1]))
        plt.axis('tight')
        # plt.clim(-10, 10)
        plt.xlabel('Range Bin')
        plt.yticks([])

        plt.show()
