import torch.nn as nn
import torch.optim as optim
import numpy as np
from latent_dialog.utils import LONG, FLOAT, Pack, cast_type
from latent_dialog.criterions import NLLEntropy, CatKLLoss, Entropy, NormKLLoss
from latent_dialog.woz_util  import SYS, EOS, PAD, BOS
from torch.autograd import Variable
import torch as th
from latent_dialog.enc2dec.encoders import RnnUttEncoder



class RlAgent(object):
    def __init__(self, model, corpus, args, name, tune_pi_only):
        self.model = model
        self.corpus = corpus
        self.args = args
        self.name = name
        self.raw_goal = None
        self.vec_goals_list = None
        self.logprobs = None
        # print("Do we only tune the policy: {}".format(tune_pi_only))
        # for n, p in self.model.named_parameters():
        #     print (n)
        self.opt = optim.SGD(
            [p for n, p in self.model.named_parameters() if 'c2z' in n or not tune_pi_only],
            lr=self.args.rl_lr,
            momentum=self.args.momentum,
            nesterov=(self.args.nesterov and self.args.momentum > 0))
        # self.opt = optim.Adam(self.model.parameters(), lr=0.01)
        # self.opt = optim.RMSprop(self.model.parameters(), lr=0.0005)
        self.all_rewards = []
        self.all_grads = []
        self.model.train()

    def print_dialog(self, dialog, reward, stats):
        for t_id, turn in enumerate(dialog):
            if t_id % 2 == 0:
                print("Usr: {}".format(' '.join([t for t in turn if t != '<pad>'])))
            else:
                print("Sys: {}".format(' '.join(turn)))
        report = ['{}: {}'.format(k, v) for k, v in stats.items()]
        print("Reward {}. {}".format(reward, report))

    def run(self, batch, evaluator, max_words=None, temp=0.1):
        self.logprobs = []
        self.dlg_history =[]
        batch_size = len(batch['keys'])
        logprobs, outs = self.model.forward_rl(batch, max_words, temp)
        if batch_size == 1:
            logprobs = [logprobs]
            outs = [outs]

        key = batch['keys'][0]
        sys_turns = []
        sys_turns_gt = []
        # construct the dialog history for printing
        for turn_id, turn in enumerate(batch['contexts']):
            user_input = self.corpus.id2sent(turn[-1])
            self.dlg_history.append(user_input)
            # : collect the groudtruth respnse of system
            sys_output_gt = self.corpus.id2sent(turn[0])
            sys_output = self.corpus.id2sent(outs[turn_id])
            self.dlg_history.append(sys_output)
            sys_turns.append(' '.join(sys_output))
            # : gather the groundtruth response of system
            sys_turns_gt.append(' '.join(sys_output_gt))

        for log_prob in logprobs:
            self.logprobs.extend(log_prob)
        # compute reward here
        generated_dialog = {key: sys_turns}
        real_dialogues = {key: sys_turns_gt}
        # : add bleu to the reward during training
        if self.args.bleu2reward:
            return evaluator.evaluateModel(generated_dialog, real_dialogues=real_dialogues, mode="offline_rl")
        else:
            return evaluator.evaluateModel(generated_dialog, real_dialogues=False, mode="offline_rl")

    def update(self, reward, stats):
        self.all_rewards.append(reward)
        # standardize the reward
        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        # compute accumulated discounted reward
        g = self.model.np2var(np.array([r]), FLOAT).view(1, 1)
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.rl_clip)
        # for name, p in self.model.named_parameters():
        #    print(name)
        #    print(p.grad)
        self.opt.step()


        
class HierarchicalRlAgent(RlAgent):
    def __init__(self, model, corpus, args, name, tune_pi_only):
        super(HierarchicalRlAgent, self).__init__(model, corpus, args, name, tune_pi_only)
        # for n, p in self.model.named_parameters():
        #     if 'utt' in n or 'c2z' in n or 'inter_attention' in n:
        #         print ('This is the name {}'.format(n))
        if self.args.high_level:
            self.opt_high = optim.SGD(
                [p for n, p in self.model.named_parameters() if 'c2z' in n],
                lr=self.args.rl_lr_high,
                momentum=self.args.momentum,
                nesterov=(self.args.nesterov and self.args.momentum > 0),
                weight_decay=self.args.weight_decay
                )
        if self.args.low_level:
            if self.args.end2end:
                self.opt_low = optim.SGD(
                    [p for n, p in self.model.named_parameters() if 'c2z' in n or 'decoder_x2y' in n or 'z_embedding_x2y' in n],
                    lr=self.args.rl_lr_low,
                    momentum=self.args.momentum,
                    nesterov=(self.args.nesterov and self.args.momentum > 0),
                    weight_decay=self.args.weight_decay
                    )
            else:
                self.opt_low = optim.SGD(
                    [p for n, p in self.model.named_parameters() if 'decoder_x2y' in n or 'z_embedding_x2y' in n],
                    lr=self.args.rl_lr_low,
                    momentum=self.args.momentum,
                    nesterov=(self.args.nesterov and self.args.momentum > 0),
                    weight_decay=self.args.weight_decay
                    )
        self.update_n = 0
        self.all_rewards = {'success': [], 'match': [], 'bleu': [], 'nll': []}
        if self.args.kl:
            self.gauss_kl = NormKLLoss(unit_average=True)
        if self.args.low_level:
            self.curr_level = 'low'
        if self.args.high_level:
            self.curr_level = 'high'
        self.low_cnt = 0
        self.high_cnt = 0
        self.alpha = self.args.alpha
        self.rl_clip = self.args.rl_clip
        
    def construct_nll_reward(self, log_softmax, symbol):
        assert len(log_softmax)==len(symbol), "The length of log_softmax is {}, whereas the length of symbol is {}.".format(len(log_softmax), len(symbol))
        log_softmax_list = []
        for b_id in range(len(log_softmax)):
            b_log_softmax = []
            # to avoid generated sentence is longer than gt, and vice versa
            for t_id in range(min(len(log_softmax[b_id]), len(symbol[b_id]))):
                b_log_softmax.append( log_softmax[b_id][t_id].gather(0, symbol[b_id][t_id]) )
            log_softmax_list.append(b_log_softmax)
        return log_softmax_list

    def run(self, batch, evaluator, max_words=None, temp=0.1):
        self.logprobs = dict(high_level=[], low_level=[], low_level_org=[])
        self.dlg_history =[]
        batch_size = len(batch['keys'])
        logprobs, outs, logprob_z, sample_z, log_softmax, log_softmax_np = self.model.forward_rl(batch, max_words, temp, self.args)

        if self.args.kl:
            prior_mu = self.model.np2var(0.0*np.ones((1,1)), FLOAT)
            prior_logvar = self.model.np2var(-2.3*np.ones((1,1)), FLOAT)
            self.kl_loss = self.gauss_kl(self.model.q_mu, self.model.q_logvar, prior_mu, prior_logvar).view(1,1)
            # print ('This is the shape of kl: ', self.kl_loss)

        # for tackling some special case
        if batch_size == 1:
            logprobs = [logprobs]
            outs = [outs]

        # log likelihood from disc
        nll_reward = log_softmax 
        nll_reward_np = log_softmax_np

        key = batch['keys'][0]
        sys_turns = []
        sys_turns_gt = []
        # construct the dialog history for printing
        for turn_id, turn in enumerate(batch['contexts']):
            user_input = self.corpus.id2sent(turn[-1])
            self.dlg_history.append(user_input)
            # collect the groudtruth respnse of system
            sys_output_gt = self.corpus.id2sent(turn[0])
            sys_output = self.corpus.id2sent(outs[turn_id])
            self.dlg_history.append(sys_output)
            sys_turns.append(' '.join(sys_output))
            # gather the groundtruth response of system
            sys_turns_gt.append(' '.join(sys_output_gt))

        for b_id in range(batch_size):
            self.logprobs['high_level'].append(logprob_z[b_id])
        
        for log_prob in logprobs:
            self.logprobs['low_level'].extend(log_prob)

        self.logprobs['low_level_org'] = logprobs

        # compute reward here
        generated_dialog = {key: sys_turns}
        real_dialogues = {key: sys_turns_gt}

        # add bleu to the reward during training
        if self.args.bleu2reward:
            report, success, match, bleu = evaluator.evaluateModel(generated_dialog, real_dialogues=real_dialogues, mode="offline_rl")
        else:
            report, success, match, bleu = evaluator.evaluateModel(generated_dialog, real_dialogues=False, mode="offline_rl")
        return report, success, match, bleu, nll_reward, nll_reward_np

    def preprocess_common_reward(self, reward, all_rewards):
        all_rewards.append(reward)
        # standardize the reward
        r = (reward - np.mean(all_rewards)) / max(self.args.std_threshold, np.std(all_rewards))
        # compute accumulated discounted reward
        g = self.model.np2var(np.array([r]), FLOAT).view(1, 1)
        rewards_high_level = []
        rewards_low_level = []
        assert len(self.logprobs['low_level_org'])==len(self.logprobs['high_level'])
        n = len(self.logprobs['low_level_org'])
        for b_p in self.logprobs['low_level_org']:
            if not self.args.long_term:
                g = self.model.np2var(np.array([r]), FLOAT).view(1, 1) / n
            if self.args.low_level:
                for w_p in b_p:
                    rewards_low_level.insert(0, g)
                    g = g * self.args.gamma
                rewards_high_level.insert(0, g)
            else:
                rewards_high_level.insert(0, g)
                if self.args.long_term:
                    g = g * self.args.gamma
        return rewards_high_level, rewards_low_level

    def preprocess_nll_reward(self, nll_reward):
        # follow the SMDP version
        high_level_nll_rewards = []
        low_level_nll_rewards = []
        g = 0
        nll_reward.reverse()
        m = np.mean(self.all_rewards['nll'])
        s = np.std(self.all_rewards['nll'])
        if self.args.low_level:
            for b_r in nll_reward:
                if not self.args.long_term:
                    g = 0
                b_r.reverse()
                for w_r in b_r:
                    if self.args.nll_normalize:
                        w_r_ = (w_r.detach() - m) / max(self.args.std_threshold, s)
                    else:
                        w_r_ = w_r.detach()
                    g = self.args.gamma_nll * g + w_r_
                    low_level_nll_rewards.insert(0, g)
                high_level_nll_rewards.insert(0, g)
        else:
            for b_r in nll_reward:
                b_r.reverse()
                g_ = 0
                for w_r in b_r:
                    if self.args.nll_normalize:
                        w_r_ = (w_r.detach() - m) / max(self.args.std_threshold, s)
                    else:
                        w_r_ = w_r.detach()
                    g_ = g_ + w_r_
                if self.args.long_term:
                    g = self.args.gamma_nll * g + g_
                    high_level_nll_rewards.insert(0, g)
                else:
                    high_level_nll_rewards.insert(0, g_)
        return high_level_nll_rewards, low_level_nll_rewards

    def update(self, reward, stats):
        rewards = {'success': {'high': [], 'low': []}, 'match': {'high': [], 'low': []}, 'bleu': {'high': [], 'low': []}}
        if reward.get('nll', 0):
            self.all_rewards['nll'].extend(reward['nll_np'])
            high_level_nll_rewards, low_level_nll_rewards = self.preprocess_nll_reward(reward['nll'].copy())
        for k in rewards.keys():
            rewards_high_level, rewards_low_level = self.preprocess_common_reward(reward[k], self.all_rewards[k])
            rewards[k]['high'] = rewards_high_level.copy()
            rewards[k]['low'] = rewards_low_level.copy()

        self.loss = 0

        # estimate the loss using one MonteCarlo rollout
        if self.args.low_level and (self.curr_level=='low' or self.args.synchron) :
            if self.args.success2reward:
                for lp, r in zip(self.logprobs['low_level'], rewards['success']['low']):
                    self.loss -= lp * (1 - self.alpha) * r
            if self.args.bleu2reward:
                for lp, r in zip(self.logprobs['low_level'], rewards['bleu']['low']):
                    self.loss -= lp * self.alpha * r
            if self.args.disc2reward:
                for lp, r in zip(self.logprobs['low_level'], low_level_nll_rewards):
                    self.loss -= lp * self.alpha * r
            self.low_cnt += 1

        # add flag to control the freq of update on high level policy
        if self.args.high_level and (self.curr_level=='high' or self.args.synchron) :
            if self.args.success2reward:
                for lp, r in zip(self.logprobs['high_level'], rewards['success']['high']):
                    self.loss -= lp * (1 - self.alpha) * r
            if self.args.bleu2reward:
                for lp, r in zip(self.logprobs['high_level'], rewards['bleu']['high']):
                    self.loss -= lp * self.alpha * r
            if self.args.disc2reward:
                for lp, r in zip(self.logprobs['high_level'], high_level_nll_rewards):
                    self.loss -= lp * self.alpha * r
            self.high_cnt += 1

        if self.args.high_level and self.args.low_level and self.args.synchron :
           self.opt_high.zero_grad()
           self.opt_low.zero_grad()
           self.loss.backward()
           nn.utils.clip_grad_norm_(self.model.parameters(), self.rl_clip)
           self.opt_high.step()
           self.opt_low.step()
        else:
            if self.args.high_level and self.curr_level=='high' :
                self.opt_high.zero_grad()
                self.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.rl_clip)
                self.opt_high.step()
                if self.args.low_level:
                    if self.high_cnt == self.args.high_freq:
                        self.curr_level = 'low'
                        self.high_cnt = 0
            elif self.args.low_level and self.curr_level=='low' :
                self.opt_low.zero_grad()
                self.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.rl_clip)
                self.opt_low.step()
                if self.args.high_level:
                    if self.low_cnt == self.args.low_freq:
                        self.curr_level = 'high'
                        self.low_cnt = 0

        self.update_n += 1
        if self.args.rl_clip_scheduler and self.update_n%self.args.rl_clip_freq==0:
            self.rl_clip *= self.args.rl_clip_decay
