import logging
from pathlib import Path
from typing import Union


def set_logging(console_level: str = 'WARNING', logfiles_dir: Union[str, Path] = None, logfiles_prefix: str = 'log',
                conf_path: Union[str, Path] = 'logging.conf'):
    """
    Configures logging with provided ".conf" file, console level, output paths.
    @param conf_path: Path to ".conf" file with loggers, handlers, formatters, etc.
    @param console_level: Level of logging to output to console. Defaults to "WARNING"
    @param logfiles_dir: path where output logs will be written
    @return:
    """
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    if not logfiles_dir:
        logging.basicConfig(format='%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.INFO)
        return
    conf_path = Path(conf_path).absolute()
    if not conf_path.is_file():
        raise FileNotFoundError(f'Invalid logging configuration file')
    log_config_path = Path(conf_path).absolute()
    out = Path(logfiles_dir) / logfiles_prefix
    logging.config.fileConfig(log_config_path, defaults={'logfilename': f'{out}.log',
                                                         'logfilename_error': f'{out}_error.log',
                                                         'logfilename_debug': f'{out}_debug.log',
                                                         'console_level': console_level})