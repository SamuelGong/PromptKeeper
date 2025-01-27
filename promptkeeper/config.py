import os
import yaml
from munch import DefaultMunch  # nested dict to object
config_rel = "config.yml"


class MyConfig(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def load_configurations(working_dir):
    config_path = os.path.join(working_dir, config_rel)

    assert os.path.isfile(config_path)
    with open(config_path, 'rb') as fin:
        config_dict = yaml.load(fin, Loader=yaml.FullLoader)

    result = DefaultMunch.fromDict(config_dict)
    return result
