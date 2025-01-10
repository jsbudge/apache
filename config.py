import yaml


class Config:
    pass

def get_config(model: str = None, param_filepath: str = None, param_dict: dict = None):
    c = Config()
    if param_filepath is not None:
        with open(param_filepath, 'r') as file:
            try:
                config_params = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        for key, val in config_params[f'{model}_params'].items():
            c.__dict__[key] = val
            for ik, iv in val.items():
                c.__dict__[ik] = iv
    elif param_dict is not None:
        for key, val in param_dict.items():
            c.__dict__[key] = val
            for ik, iv in val.items():
                c.__dict__[ik] = iv
    else:
        print('Neither param_filepath nor param_dict has been populated.')
        return None
    return c

if __name__ == '__main__':
    test = get_config('wave_exp', './vae_config.yaml')
