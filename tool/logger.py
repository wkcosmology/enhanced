import logging
import os
import yaml
import logging.config
import enhanced

def get_logger(name, filename=""):
    """get user defined logger

    Parameters
    ----------
    name : str
        name of the logger defined in my yaml config file
    filename : str, optional
        name of log file

    Returns
    -------
    Logger

    """
    config_file = os.path.join(enhanced.__path__[0], "tool/logger_config.yaml")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if filename != "":
        filename = filename + "_"
    config['handlers']['file']['filename'] = filename + "logger.log"
    config['handlers']['complete']['filename'] = filename + "debug.log"
    # This is to avoid creat debugging log file when I do not want to
    # Should modify the code when you modify the yaml file
    useless_handlers = [h
                        for h in config['handlers'].keys()
                        if h not in config["loggers"][name]['handlers']]
    useless_loggers = [logger
                       for logger in config['loggers'].keys()
                       if logger != name]
    for h in useless_handlers:
        del config['handlers'][h]
    for l in useless_loggers:
        del config['loggers'][l]

    logging.config.dictConfig(config)
    return logging.getLogger(name)
