import wget
import os
import config as cfg

wget.download(cfg.URL, cfg.DATA_PATH)