# metagnn.utils.py

import torch
import pyro 

from datetime import datetime
import logging
import os, gc, json, re, shutil

from metagnn.utils import METAGNN_GLOBALS, get_logger
from metagnn.tools.common import MetaGNNConfig
from metagnn.tools.model.vae import MixtureVAE

logger = get_logger(__name__)

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the subdirectory
        except Exception as e:
            raise RuntimeError(f"Failed to delete {file_path}. Reason: {e}")

def get_most_recent_run_id(save_folder):
    run_ids = [
        d for d in os.listdir(save_folder)
        if os.path.isdir(os.path.join(save_folder, d)) and re.match(r"^\d{8}-\d{6}$", d)
    ]
    if not run_ids:
        return None
    run_ids.sort(reverse=True)
    return run_ids[0]
    
def save_model(model, overwrite=False, verbose=True):
    save_folder = METAGNN_GLOBALS["save_folder"]
    os.makedirs(save_folder, exist_ok=True)

    if overwrite is True:
        run_id = model.run_id
        if verbose is True:
            logger.info(f"Overwriting existing model.")
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        model.run_id = run_id
        if verbose is True:
            logger.info(f"Saving model with run_id {run_id}.")
    
    full_path = os.path.join(save_folder, run_id)
    os.makedirs(full_path, exist_ok=True)
    clear_directory(full_path)
    
    model_config = model.config if hasattr(model, "config") else None
    encoder = model.encoder
    decoder = model.decoder
    
    torch.save(model.encoder, os.path.join(full_path, "metagnn_encoder.pt"))
    torch.save(model.decoder, os.path.join(full_path, "metagnn_decoder.pt"))
    with open(os.path.join(full_path, "config.json"), "w") as f:
        json.dump(model_config.to_dict(), f)

    param_store = pyro.get_param_store()
    param_dict = {name: param_store[name].detach().cpu() for name in param_store.keys()}
    torch.save(param_dict, os.path.join(full_path, "metagnn_param_store.pt"))
    
def load_model(run_id=None):
    pyro.clear_param_store()
    gc.collect()
    torch.cuda.empty_cache()

    save_folder = METAGNN_GLOBALS["save_folder"]
    if run_id is None:
        run_id = get_most_recent_run_id(save_folder)
    full_path = os.path.join(save_folder, run_id)

    with open(os.path.join(full_path, "config.json"), "r") as f:
        config_dict = json.load(f)
        config = MetaGNNConfig.from_dict(config_dict)
    
    decoder = torch.load(
        os.path.join(full_path, "metagnn_decoder.pt"), weights_only=False
    )
    model = MixtureVAE(decoder.kmer_map, config)
    model.encoder = torch.load(os.path.join(full_path, "metagnn_encoder.pt"), weights_only=False)
    model.decoder = decoder
    
    param_store = pyro.get_param_store()
    param_dict = torch.load(os.path.join(full_path, "metagnn_param_store.pt"), weights_only=False)
    for name, param in param_dict.items():
        if name in param_store:
            param_store[name] = param.to(model.config.device)
        else:
            pyro.param(name, param.to(model.config.device))

    torch.set_default_dtype(torch.float32)
    torch.set_default_device(model.config.device)

    return model