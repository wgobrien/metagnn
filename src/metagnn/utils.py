# metagnn.utils.py

import os

METAGNN_GLOBALS = dict()
METAGNN_GLOBALS["save_folder"] = "./_metagnn/"

def get_logger(name):
    import logging
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s : %(message)s",
        level=logging.INFO,
    )
    return logger

def is_notebook():
    """
    Check if the script is running in a Jupyter notebook.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False
    except ImportError:
        return False