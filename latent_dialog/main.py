import os
import sys
import numpy as np
import torch as th
from torch import nn

from collections import defaultdict
from datetime import datetime

from latent_dialog.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from collections import defaultdict
from latent_dialog.utils import idx2word

# # rl
# from latent_dialog.record import record, record_task, UniquenessSentMetric, UniquenessWordMetric

import logging
from latent_dialog.utils import TBLogger



logger = logging.getLogger()

PREVIEW_NUM = 0


class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            # print('key = %s\nval = %s' % (key, val))
            if val is not None and type(val) is not bool:
                self.losses[key].append(val.item())

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            aver_loss = np.average(loss) if window is None else np.average(loss[-window:])
            if 'nll' in key:
                str_losses.append('{} PPL {:.3f}'.format(key, np.exp(aver_loss)))
            else:
                str_losses.append('{} {:.3f}'.format(key, aver_loss))

        if prefix:
            return '{}: {} {}'.format(prefix, name, ' '.join(str_losses))
        else:
            return '{} {}'.format(name, ' '.join(str_losses))

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def avg_loss(self):
        return np.mean(self.backward_losses)


def reinforce(agent, model, train_data, val_data, rl_config, sl_config, evaluator):

    # TODO: model can be saved in agent

    # clone trian data for supervised learning
    sl_train_data = train_data.clone()

    # tensorboard
    tb_path = os.path.join(rl_config.saved_path, "tensorboard/")
    tb_logger = TBLogger(tb_path)

    episode_cnt, best_episode = 0, 0
    best_valid_loss = np.inf
    best_rewards = -1 * np.inf

    # model
    model.train()
    saved_models = []
    last_n_model = rl_config.last_n_model 

    logger.info('***** Reinforce Begins at {} *****'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    for epoch_id in range(rl_config.num_epoch):

        train_data.epoch_init(sl_config, shuffle=True, verbose=epoch_id == 0, fix_batch=True) # fix_batch has to be true for offline reinforce. each batch is an episode
        while True:
            batch = train_data.next_batch() 
            if batch is None:
                break

            # reinforcement learning
            assert len(set(batch['keys'])) == 1 # make sure it's the same dialo
            report, success, match, bleu, nll_reward, nll_reward_np = agent.run(batch, evaluator, max_words=rl_config.max_words, temp=rl_config.temperature)
            # this is the reward function during training
            reward = float(success)
            reward_dict = {'success': float(success), 'match': float(match), 'bleu': float(bleu)}
            # 
            if rl_config.disc2reward:
                reward_dict['nll'] = nll_reward
                nll_reward_record = []
                for b_r in nll_reward_np:
                    nll_reward_record.extend(b_r)
                reward_dict['nll_np'] = nll_reward_record
                # reward_dict['nll_std'] = np.std(nll_reward_record)
                # print (nll_reward_np) 
                stats = {'train/match': match, 'train/success': success, 'train/bleu': bleu, 'train/nll': np.mean(nll_reward_record)}
            else:
                stats = {'train/match': match, 'train/success': success, 'train/bleu': bleu}
            agent.update(reward_dict, stats)
            tb_logger.add_scalar_summary(stats, episode_cnt)
            
            # supervised learning
            if rl_config.sl_train_frequency > 0 and episode_cnt % rl_config.sl_train_frequency == 0:
                train_single_batch(model, sl_train_data, sl_config)

            # print loss sometimes
            episode_cnt += 1
            if episode_cnt % rl_config.print_frequency == 0:
                # fit hierarchical rl, display success rate only
                if rl_config.disc2reward:
                    # print (agent.all_rewards['nll'])
                    logger.info("{}/{} episode: mean_reward {} , mean_nll {} and mean_bleu {} for last {} episodes".format(episode_cnt,
                                                train_data.num_batch*rl_config.num_epoch,
                                                np.mean(agent.all_rewards['success'][-rl_config.print_frequency:]),
                                                np.mean(agent.all_rewards['nll'][-rl_config.print_frequency:]),
                                                np.mean(agent.all_rewards['bleu'][-rl_config.print_frequency:]),
                                                rl_config.print_frequency)
                    )
                else:
                    logger.info("{}/{} episode: mean_reward {} and mean_bleu {} for last {} episodes".format(episode_cnt, 
                                                train_data.num_batch*rl_config.num_epoch,
                                                np.mean(agent.all_rewards['success'][-rl_config.print_frequency:]),
                                                np.mean(agent.all_rewards['bleu'][-rl_config.print_frequency:]),
                                                rl_config.print_frequency)
                    ) # episode = batch * epoch

            # record model performance in terms of several evaluation metrics
            if rl_config.record_frequency > 0 and episode_cnt % rl_config.record_frequency == 0:
                
                logger.info('Checkpoint step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
                logger.info('==== Evaluating Model ====')

                # TODO:  train reward
                agent.print_dialog(agent.dlg_history, reward, stats)

                # fit hierarchical rl, display success rate only
                logger.info('mean_reward {} for last {} episodes'.format(np.mean(agent.all_rewards['success'][-rl_config.record_frequency:]), rl_config.record_frequency))
                
                # validation 
                valid_loss = validate(model, val_data, sl_config)
                v_success, v_match, v_bleu = generate(model, val_data, sl_config, evaluator)

                # tensorboard
                stats = {'val/success': v_success, 'val/match': v_match, 'val/bleu': v_bleu, "val/loss": valid_loss}
                tb_logger.add_scalar_summary(stats, episode_cnt)

                # save model
                # consider bleu into the evaluation metric
                if (v_success+v_match)/2+v_bleu > best_rewards:
                    cur_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    logger.info('*** Model Saved with match={} success={} bleu={}, at {}. ***\n'.format(v_match, v_success, v_bleu, cur_time))

                    model.save(rl_config.saved_path, episode_cnt)
                    best_episode = episode_cnt
                    saved_models.append(episode_cnt)
                    if len(saved_models) > last_n_model:
                        remove_model = saved_models[0]
                        saved_models = saved_models[-last_n_model:]
                        os.remove(os.path.join(rl_config.saved_path, "{}-model".format(remove_model)))
                    # new evaluation metric
                    best_rewards = (v_success+v_match)/2+v_bleu

            model.train()
            sys.stdout.flush()

    return best_episode


def train_single_batch(model, train_data, config):

    batch_cnt = 0
    optimizer = model.get_optimizer(config, verbose=False)
    model.train()
    # decoding CE
    train_data.epoch_init(config, shuffle=True, verbose=False)
    # TODO: hard code here
    for i in range(16):
        batch = train_data.next_batch()
        if batch is None:
            train_data.epoch_init(config, shuffle=True, verbose=False)
            batch = train_data.next_batch()
        optimizer.zero_grad()
        loss = model(batch, mode=TEACH_FORCE)
        model.backward(loss, batch_cnt)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()


def train(model, train_data, val_data, config, evaluator):
    # tensorboard
    tb_path = os.path.join(config.saved_path, "tensorboard/")
    tb_logger = TBLogger(tb_path)

    # training parameters
    patience, batch_cnt, best_epoch = 10, 0, 0
    valid_loss_threshold, best_valid_loss = np.inf, np.inf
    
    train_loss = LossManager()
    best_rewards = 0

    # models
    model.train()
    optimizer = model.get_optimizer(config, verbose=False)
    saved_models = []
    last_n_model = config.last_n_model 


    logger.info('***** Training Begins at {} *****'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    logger.info('***** Epoch 0/{} *****'.format(config.num_epoch))
    for epoch in range(config.num_epoch):
        # EPOCH
        train_data.epoch_init(config, shuffle=True, verbose=epoch==0, fix_batch=config.fix_train_batch)
        num_batch = train_data.num_batch

        while True:
            # BATCH
            batch = train_data.next_batch()
            if batch is None:
                break

            optimizer.zero_grad()
            # TODO: TEACH_FORCE = decoding directly by groundtruth then adding attn
            loss = model(batch, mode=TEACH_FORCE, epoch=epoch)
            train_loss.add_loss(loss) 
            model.backward(loss, batch_cnt)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_cnt += 1

            # tensorboard save 
            data_dict = {}
            for key, val in loss.items():
                if val is not None and type(val) is not bool:
                    data_dict["train/%s"%key] = val.item()
            tb_logger.add_scalar_summary(data_dict, batch_cnt)

            # print training loss every print_frequency batch
            if batch_cnt % config.print_frequency == 0:
                # TODO: what is model.kl_w
                logger.info(train_loss.pprint('Train',
                                        window=config.print_frequency,
                                        prefix='{}/{}-({:.3f})'.format(batch_cnt%num_batch, num_batch, model.kl_w)))
                sys.stdout.flush()

        # Evaluate at the end of every epoch
        logger.info('Checkpoint step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
        logger.info('==== Evaluating Model ====')

        # Generation (bleu/success/match)
        success, match, bleu = generate(model, val_data, config, evaluator)

        # Validation (loss)
        logger.info(train_loss.pprint('Train'))
        valid_loss = validate(model, val_data, config, batch_cnt)

        stats = {'val/success': success, 'val/match': match, 'val/bleu': bleu, "val/loss": valid_loss}
        tb_logger.add_scalar_summary(stats, batch_cnt)

        if epoch >= config.warmup:
            # Save Models if valid loss decreases
            if valid_loss < best_valid_loss:
                if valid_loss <= valid_loss_threshold * config.improve_threshold:
                    patience = max(patience, epoch*config.patient_increase)
                    valid_loss_threshold = valid_loss
                    logger.info('Update patience to {}'.format(patience))

                if config.save_model:
                    cur_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    logger.info('*** Model Saved with valid_loss = {}, at {}. ***'.format(valid_loss, cur_time))
                    model.save(config.saved_path, epoch)
                    best_epoch = epoch
                    saved_models.append(epoch)
                    if len(saved_models) > last_n_model:
                        remove_model = saved_models[0]
                        saved_models = saved_models[-last_n_model:]
                        os.remove(os.path.join(config.saved_path, "{}-model".format(remove_model)))
                best_valid_loss = valid_loss

        # Early stop 
        if config.early_stop and patience <= epoch:
            logger.info('*** Early stop due to run out of patience ***')
            break

        # exit val mode
        model.train()
        train_loss.clear()
        logger.info('\n***** Epoch {}/{} *****'.format(epoch+1, config.num_epoch))
        sys.stdout.flush()
    
    logger.info('Training Ends. Best validation loss = %f' % (best_valid_loss, ))
    return best_epoch



def validate(model, data, config, batch_cnt=None):
    model.eval()
    data.epoch_init(config, shuffle=False, verbose=False, fix_batch=True)
    losses = LossManager()
    while True:
        batch = data.next_batch()
        if batch is None:
            break
        loss = model(batch, mode=TEACH_FORCE)
        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

    valid_loss = losses.avg_loss()
    logger.info(losses.pprint(data.name))
    logger.info('--- Total loss = {}'.format(valid_loss))
    sys.stdout.flush()
    return valid_loss


def generate(model, data, config, evaluator, verbose=True, dest_f=None, vec_f=None, label_f=None):
    """
        Args:
            - evalutor: this is used to calculate bleu/match/success
    """

    model.eval()
    batch_cnt = 0
    generated_dialogs = defaultdict(list)
    real_dialogs = defaultdict(list)

    data.epoch_init(config, shuffle=False, verbose=False, fix_batch=True)
    logger.debug('Generation: {} batches'.format(data.num_batch))
    while True:
        batch = data.next_batch()
        batch_cnt += 1
        if batch is None:
            break

        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int)  # (batch_size, max_dec_len)
        true_labels = labels.data.numpy()  # (batch_size, output_seq_len)

        # get context
        ctx = batch.get('contexts')  # (batch_size, max_ctx_len, max_utt_len)
        ctx_len = batch.get('context_lens')  # (batch_size, )
        keys = batch['keys']

        sample_z = outputs["sample_z"].cpu().data.numpy()

        batch_size = pred_labels.shape[0]
        for b_id in range(batch_size):
            pred_str = idx2word(model.vocab, pred_labels, b_id)
            true_str = idx2word(model.vocab, true_labels, b_id)
            prev_ctx = ''
            if ctx is not None:
                ctx_str = []
                for t_id in range(ctx_len[b_id]):
                    temp_str = idx2word(model.vocab, ctx[:, t_id, :], b_id, stop_eos=False)
                    ctx_str.append(temp_str)
                prev_ctx = 'Source context: {}'.format(ctx_str)

            generated_dialogs[keys[b_id]].append(pred_str)
            real_dialogs[keys[b_id]].append(true_str)

            if verbose and batch_cnt <= PREVIEW_NUM:
                logger.debug('%s-prev_ctx = %s' % (keys[b_id], prev_ctx,))
                logger.debug('True: {}'.format(true_str, ))
                logger.debug('Pred: {}'.format(pred_str, ))
                logger.debug('-' * 40)

            if dest_f is not None:
                dest_f.write('%s-prev_ctx = %s\n' % (keys[b_id], prev_ctx,))
                dest_f.write('True: {}\n'.format(true_str, ))
                dest_f.write('Pred: {}\n'.format(pred_str, ))
                dest_f.write('-' * 40+"\n")

            if  vec_f is not None:
                sample = sample_z[b_id]
                sample_str = "\t".join( str(x) for x in sample )
                vec_f.write(sample_str + "\n")

            if label_f is not None:
                label_f.write("%s\t%s\n"%(true_str, pred_str))


    task_report, success, match, bleu  = evaluator.evaluateModel(generated_dialogs, real_dialogues=real_dialogs, mode=data.name)

    logger.debug('Generation Done')
    logger.info(task_report)
    logger.debug('-' * 40)
    return success, match, bleu









