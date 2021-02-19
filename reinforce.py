"""
    This is the main entry of multiwoz experiments(supervised learning)
    Usage:
        - train and evaluate
            python reinforce.py --config_name sl_cat
        - evaluate only
            python reinforce.py --config_name sl_cat --forward_only
    Details:

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
from latent_dialog.main import  reinforce, validate, generate
import  latent_dialog.agents as agents 


# load config
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default="rl_cat")
parser.add_argument('--forward_only', action='store_true')
parser.add_argument('--gen_type', type=str, default='greedy')
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--alias', type=str, default="")
args = parser.parse_args()
rl_config_path = "./configs/" + args.config_name + ".conf"
rl_config = Pack(json.load(open(rl_config_path)))
rl_config["forward_only"] = args.forward_only


# set random_seed/logger/save_path
set_seed(rl_config.random_seed)

alias = args.alias if args.alias == "" else '-' + args.alias
saved_path = os.path.join('./outputs/', args.config_name + alias) # path for rl
if not os.path.exists(saved_path):
    os.makedirs(saved_path)


pretrain_path = './outputs/' + rl_config.pretrain_folder + '/'   # path for sl
if os.path.exists(os.path.join(pretrain_path, 'config.json')):
    sl_config = Pack(json.load(open(os.path.join(pretrain_path, 'config.json'))))
else:
    sl_config = Pack(json.load(open(rl_config_path.replace("rl", "sl"))))
sl_config['dropout'] = 0.0
sl_config['use_gpu'] = rl_config.use_gpu
sl_config['gen_type'] = args.gen_type
sl_config['beam_size'] = args.beam_size
rl_config.saved_path = saved_path

prepare_dirs_loggers(rl_config)
logger = logging.getLogger()
start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[START]\n{}\n{}'.format(start_time, '=' * 30))


# load dataset dependent (corpus, context-to-response, evaluator)
if rl_config.data_name.startswith("camrest"):
    corpus = CamRestCorpus(sl_config)
    train_dial, val_dial,  test_dial = corpus.get_corpus()
    train_data = CamRestDataLoader('Train', train_dial, sl_config)
    val_data = CamRestDataLoader('Val', val_dial, sl_config)
    test_data = CamRestDataLoader('Test', test_dial, sl_config)
    evaluator = CamRestEvaluator(sl_config.data_name)
else: # multiwoz
    corpus = MultiWozCorpus(sl_config)
    train_dial, val_dial,  test_dial = corpus.get_corpus()
    train_data = MultiWozDataLoader('Train', train_dial, sl_config)
    val_data = MultiWozDataLoader('Val', val_dial, sl_config)
    test_data = MultiWozDataLoader('Test', test_dial, sl_config)
    evaluator = MultiWozEvaluator(sl_config.data_name)
    
# load pretrained models
model_class = getattr(models, sl_config.model_name)
model = model_class(corpus, sl_config)

if sl_config.use_gpu:
    model.cuda()
model_ids = sorted([int(p.replace('-model', '')) for p in os.listdir(pretrain_path) if 'model' in p])
best_epoch = model_ids[-1]
model.load(pretrain_path, best_epoch)
model.print_summary()


# create rl agent
agent_class = getattr(agents, rl_config.agent_name)
agent = agent_class(model, corpus, rl_config, name='System', tune_pi_only=rl_config.tune_pi_only)


##################### Training #####################
best_episode = None
if not rl_config.forward_only:
    # save config
    with open(os.path.join(saved_path, 'config.json'), 'w') as f:
        json.dump(rl_config, f, indent=4)  # sort_keys=True
    try:
        best_episode = reinforce(agent, model, train_data, val_data, rl_config, sl_config, evaluator)
    except KeyboardInterrupt:
        logger.error('Training stopped by keyboard.')

if best_episode is None:
    model_ids = sorted([int(p.replace('-model', '')) for p in os.listdir(saved_path) if 'model' in p])
    best_episode= model_ids[-1]
model.load(saved_path, best_episode)

#################### Validation #####################
logger.info("\n***** Forward Only Evaluation on val/test *****")
logger.info("$$$ Load {}-model".format(best_episode))
validate(model, val_data, sl_config)
validate(model, test_data, sl_config)

##################### Generation #####################
with open(os.path.join(saved_path, '{}_valid_file.txt'.format(best_episode)), 'w') as f:
    generate(model, val_data, sl_config, evaluator, dest_f=f)

# Save latent action 
vec_f = open(os.path.join(saved_path, 'vec_file.tsv'), 'w')
label_f = open(os.path.join(saved_path, 'label_file.tsv'), 'w')
with open(os.path.join(saved_path, '{}_test_file.txt'.format(best_episode)), 'w') as f:
    generate(model, test_data, sl_config, evaluator, dest_f=f, vec_f=vec_f, label_f=label_f)


end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[END]' +  end_time)

