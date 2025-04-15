# Apache

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

This should get all the data needed to train the wavemodel.
