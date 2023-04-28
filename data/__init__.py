from .data_mfm import build_loader_mfm
from .data_finetune import build_loader_finetune

def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_mfm(config, logger)
    else:
        return build_loader_finetune(config, logger)
