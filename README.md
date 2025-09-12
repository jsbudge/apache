# Apache
There are several necessary steps to get data for training the wavemodel and the target autoencoder.
Simulation of training data also expects specific mesh model files that are used to generate data.

First, to create mesh models:
1) Select a .obj or .gbl triangle mesh model file.
2) Add the file path to create_model_file.py and run it.
3) This should place a .model file into the folder specified in the script.

Model files are used inside the simulation interface, which is explained in the final report for this project.
To train the wavemodel, you will need data files from the target autoencoder. To train the autoencoder, you will need
data files from the target_data_generator.py script.

To update data:

1) Run target_data_generator.py - make sure in vae_config.yaml you have
```commandline
generate_data_settings:
  save_files: True
  run_target: True
  run_clutter: True
```
all of these things True, otherwise it will not run everything or save anything.
2) Train the target embedding model with target_train.py. Make sure to save the model.
3) Run target_means_generator.py

You can now run wave_train.py to train the wavemodel.
