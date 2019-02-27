import os
import torch

from common.err import StorePathFileExist
from common.logger import logger

def check_file_exists(fp, overwrite):
    if not overwrite and os.path.exists(fp):
        raise StorePathFileExist(fp)

def default_data_saver(obj, fp):
    torch.save(obj, fp)
    logger.info("process saved")