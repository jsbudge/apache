import yaml
from torchnets import BetaVAE
from torchdata.datapipes.iter import FileLister, FileOpener
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import torch
from tqdm import tqdm
import numpy as np

yaml_fnme = './vae_config.yaml'

with open(yaml_fnme, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Get data
dpipe1 = FileLister('./data', '*.tfrecords')
dpipe2 = FileOpener(dpipe1, mode='b')
tfrecord_loader = dpipe2.load_from_tfrecord()
dl = DataLoader(dataset=tfrecord_loader)

mdl = BetaVAE(2, 32)

loss_fn = nn.MSELoss()
optimizer = Adam(mdl.parameters())


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    progress = tqdm(np.arange(0, len(dl.dataset), config['settings']['batch_sz']), unit='batch', miniterval=0)
    for i, data in enumerate(dl):
        progress.update(1)
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = mdl(inputs)

        # Compute the loss and its gradients
        loss = mdl.loss_function(*outputs, M_N=1.0)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        progress.set_postfix(loss=running_loss / config['settings']['batch_sz'])
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


epoch_number = 0
EPOCHS = 5
best_vloss = 1000000.
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    mdl.train(True)
    avg_loss = train_one_epoch(epoch_number)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    mdl.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(dl):
            vinputs, vlabels = vdata
            voutputs = mdl(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss

    epoch_number += 1
