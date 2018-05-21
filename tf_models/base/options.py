import json
from bunch import Bunch
import os

def get_options(jsonfile):
    with open(jsonfile, 'r') as f:
        option_dict = json.load(f)
    option = Bunch(option_dict)
    option.checkpoint_dir = option.checkpoint_dir or "checkpoint"
    option.summary_dir = option.summary_dir or "summary"
    option.max_to_keep = option.max_to_keep or 5
    option.num_epochs = option.num_epochs or 1
    option.num_iter_per_epoch = option.num_iter_per_epoch or 1
    option.batch_size = option.batch_size or 1
    return option
