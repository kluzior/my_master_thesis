import logging
import pathlib

def configure_logger(path='debug' ):
    log_file = pathlib.Path(f"{path}.log")

    if log_file.is_file():
        log_file.unlink()

    main_logger = logging.getLogger(__name__.split('.')[0])
    main_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    CONSOLE_LOG_FORMAT = '[%(levelname)-8s] [%(module)-15s] %(message)s'
    console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    main_logger.addHandler(console_handler)


    debug_handler = logging.FileHandler(filename=log_file)
    debug_handler.setLevel(logging.DEBUG)
    DEBUG_LOG_FORMAT = '[%(asctime)-15s] [%(levelname)-8s] [%(name)-50s] [%(module)-15s] %(message)s (%(funcName)s:%(lineno)d)'
    debug_formatter = logging.Formatter(DEBUG_LOG_FORMAT)
    debug_handler.setFormatter(debug_formatter)
    main_logger.addHandler(debug_handler)
