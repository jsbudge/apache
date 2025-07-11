import yaml


def parse_yaml(yaml_dict, param=None):
    # Loop through the yaml dictionary and assign everything to member variables
    if param is None:
        param = Config()
    for key, val in yaml_dict.items():
        if isinstance(val, dict):
            param.__dict__[key] = parse_yaml(val, Config())
        else:
            param.__dict__[key] = val
    return param

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


def load_yaml_config(filepath: str):
    with open(filepath, 'r') as file:
        try:
            config_params = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return parse_yaml(config_params)


if __name__ == '__main__':
    # test = get_config('wave_exp', './vae_config.yaml')
    test = load_yaml_config('./wave_simulator.yaml')
