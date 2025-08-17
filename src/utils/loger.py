import os
import time
import datetime
import logging
from pathlib import Path

# Suppress matplotlib font warnings early
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Glyph.*missing from current font')

# Also suppress matplotlib debug logging globally
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# def set_up_logger(args, sys_argv):
#     # Process the number of degrees and layers
#     n_degree, n_layer = process_sampling_numbers(args.n_degree, args.n_layer)
#     n_degree = [str(n) for n in n_degree]

#     # Get the start time and format it
#     start_time = time.time()
#     time_value = datetime.datetime.fromtimestamp(start_time)
#     runtime_id = f'{time_value.strftime("%Y_%m_%d_%H_%M_%S")}-{args.data}-{args.mode[0]}-{args.agg}-{n_layer}-{"k".join(n_degree)}-{args.pos_dim}'

#     # Set up the logging configuration
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
#     file_path = f'log/{runtime_id}.log'
#     fh = logging.FileHandler(file_path)
#     fh.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.WARN)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)
#     ch.setFormatter(formatter)
#     logger.addHandler(fh)
#     logger.addHandler(ch)
#     logger.info(f'Create log file at {file_path}')
#     logger.info(f'Command line executed: python ' + ' '.join(sys_argv))
#     logger.info('Full args parsed:')
#     logger.info(args)

#     # Set up the directories for storing the checkpoints and best model
#     checkpoint_root = './saved_checkpoints/'
#     checkpoint_dir = checkpoint_root + runtime_id + '/'
#     best_model_root = './best_models/'
#     best_model_dir = best_model_root + runtime_id + '/'

#     # Create the directories if they do not exist
#     Path(checkpoint_root).mkdir(parents=True, exist_ok=True)
#     Path(best_model_root).mkdir(parents=True, exist_ok=True)
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     os.makedirs(best_model_dir, exist_ok=True)



def setup_logger(results_save_path, script_time, setup=None):
    # Suppress matplotlib warnings globally
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("../log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f'{results_save_path}/{script_time}.log')
    fh.setLevel(logging.DEBUG)
    
    # Enhanced filter to ignore various debug messages
    class IgnoreDebugFilter(logging.Filter):
        def filter(self, record):
            # Filter out matplotlib debug messages
            if 'matplotlib' in record.name and record.levelno == logging.DEBUG:
                return False
            # Filter out font manager messages
            if 'font_manager' in record.name:
                return False
            # Filter out PIL debug messages
            if 'PIL' in record.name and record.levelno == logging.DEBUG:
                return False
            return True
    
    fh.addFilter(IgnoreDebugFilter())
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    if setup:
        logger.info(setup)
        
    return logger




def setup_standard_logger(results_save_path, script_time, level=logging.INFO, setup=None):
    """Function to setup a standard logger"""
    
    # Create a logger object
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create a file handler to write log messages to a file
    file_handler = logging.FileHandler(f'{results_save_path}/{script_time}.log')
    file_handler.setLevel(level)
    
    # Create a stream handler to output log messages to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    
    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add the formatter to the handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    if setup:
        logger.info(setup)
    
    return logger







