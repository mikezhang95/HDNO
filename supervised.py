"""
    This is the main entry of multiwoz experiments(supervised learning)
    Usage:
        - train and evaluate
            python supervised.py --config_name sl_cat
        - evaluate only
            python supervsied.py --config_name sl_cat --forward_only
"""

import time
import os,sys
import json
import logging
import torch as th

from latent_dialog.utils import Pack, prepare_dirs_loggers, set_seed
from latent_dialog.data_loaders import MultiWozCorpus, MultiWozDataLoader
from latent_dialog.evaluators import MultiWozEvaluator
import latent_dialog.models as models
from latent_dialog.main import  train, validate, generate

# load config
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default="sl_cat")
parser.add_argument('--forward_only', action='store_true')
parser.add_argument('--gen_type', type=str, default='greedy')
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--alias', type=str, default="")
args = parser.parse_args()
config_path = "./configs/" + args.config_name + ".conf"
config = Pack(json.load(open(config_path)))
config["forward_only"] = args.forward_only
config["gen_type"] = args.gen_type
config["beam_size"] = args.beam_size

# set random_seed/logger/save_path
set_seed(config.random_seed)

alias = args.alias if args.alias == "" else '-' + args.alias
saved_path = os.path.join("./outputs", args.config_name + alias)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
config.saved_path = saved_path

# prepare logs
prepare_dirs_loggers(config)
logger = logging.getLogger()
start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[START]\n{}\n{}'.format(start_time, '=' * 30))

# load dataset dependent (corpus, context-to-response, evaluator)
if config.data_name.startswith("camrest"):
    corpus = CamRestCorpus(config)
    train_dial, val_dial,  test_dial = corpus.get_corpus()
    train_data = CamRestDataLoader('Train', train_dial, config)
    val_data = CamRestDataLoader('Val', val_dial, config)
    test_data = CamRestDataLoader('Test', test_dial, config)
    evaluator = CamRestEvaluator(config.data_name)
else: # multiwoz
    corpus = MultiWozCorpus(config)
    train_dial, val_dial,  test_dial = corpus.get_corpus()
    train_data = MultiWozDataLoader('Train', train_dial, config)
    val_data = MultiWozDataLoader('Val', val_dial, config)
    test_data = MultiWozDataLoader('Test', test_dial, config)
    evaluator = MultiWozEvaluator(config.data_name)
    
# create system model
model_class = getattr(models, config.model_name)
model = model_class(corpus, config)
if config.use_gpu:
    model.cuda()
model.print_summary()


##################### Training #####################
best_epoch = None
if not config.forward_only:
    # save config
    with open(os.path.join(saved_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)  # sort_keys=True
    try:
        best_epoch = train(model, train_data, val_data, config, evaluator)
    except KeyboardInterrupt:
        logger.error('Training stopped by keyboard.')
if best_epoch is None:
    model_ids = sorted([int(p.replace('-model', '')) for p in os.listdir(saved_path) if 'model' in p and 'rl' not in p])
    best_epoch = model_ids[-1]
model.load(saved_path, best_epoch)

##################### Validation #####################
logger.info("\n***** Forward Only Evaluation on val/test *****")
logger.info("$$$ Load {}-model".format(best_epoch))
validate(model, val_data, config)
validate(model, test_data, config)

##################### Generation #####################
with open(os.path.join(saved_path, '{}_valid_file.txt'.format(best_epoch)), 'w') as f:
    generate(model, val_data, config, evaluator, dest_f=f)

with open(os.path.join(saved_path, '{}_test_file.txt'.format(best_epoch)), 'w') as f:
    generate(model, test_data, config, evaluator, dest_f=f)


end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[END]' + end_time)
