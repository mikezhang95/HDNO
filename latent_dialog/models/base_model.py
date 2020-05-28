import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from latent_dialog.utils import INT, FLOAT, LONG, cast_type
from latent_dialog.enc2dec.base_modules import summary # print summary of the model

import logging
logger = logging.getLogger()


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.config = config
        self.kl_w = 0.0

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(th.from_numpy(inputs)), 
                         dtype, 
                         self.use_gpu)

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, loss, batch_cnt):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = 0.0
        for k, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss

    def get_optimizer(self, config, verbose=True):
        if config.op == 'adam':
            if verbose:
                print('Use Adam')
            return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=config.init_lr,
                              weight_decay=config.l2_norm)
        elif config.op == 'sgd':
            print('Use SGD')
            return optim.SGD(self.parameters(), lr=config.init_lr, momentum=config.momentum)
        elif config.op == 'rmsprop':
            print('Use RMSProp')
            return optim.RMSprop(self.parameters(), lr=config.init_lr, momentum=config.momentum)

    def get_clf_optimizer(self, config):
        params = []
        params.extend(self.gru_attn_encoder.parameters())
        params.extend(self.feat_projecter.parameters())
        params.extend(self.sel_classifier.parameters())

        if config.fine_tune_op == 'adam':
            print('Use Adam')
            return optim.Adam(params, lr=config.fine_tune_lr)
        elif config.fine_tune_op == 'sgd':
            print('Use SGD')
            return optim.SGD(params, lr=config.fine_tune_lr, momentum=config.fine_tune_momentum)
        elif config.fine_tune_op == 'rmsprop':
            print('Use RMSProp')
            return optim.RMSprop(params, lr=config.fine_tune_lr, momentum=config.fine_tune_momentum)

        
    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)


    def extract_short_ctx(self, context, context_lens, backward_size=1):
        utts = []
        if self.config.context_lens == 'long':
            for b_id in range(context.shape[0]):
                utts.append(np.concatenate(context[b_id]))
        else:
            for b_id in range(context.shape[0]):
                # print ('This is the ctx len: ', context_lens[b_id])
                utts.append(context[b_id, context_lens[b_id]-1])
        return np.array(utts)

    def flatten_context(self, context, context_lens, align_right=False):
        utts = []
        temp_lens = []
        for b_id in range(context.shape[0]):
            temp = []
            for t_id in range(context_lens[b_id]):
                for token in context[b_id, t_id]:
                    if token != 0:
                        temp.append(token)
            temp_lens.append(len(temp))
            utts.append(temp)
        max_temp_len = np.max(temp_lens)
        results = np.zeros((context.shape[0], max_temp_len))
        for b_id in range(context.shape[0]):
            if align_right:
                results[b_id, -temp_lens[b_id]:] = utts[b_id]
            else:
                results[b_id, 0:temp_lens[b_id]] = utts[b_id]

        return results

    def print_summary(self):
        logger.info(summary(self, show_weights=False))

    def load(self, path, model_id):
        """
            load {model_id}-model from {path}
        """
        self.load_state_dict(th.load(os.path.join(path, '{}-model'.format(model_id))))

    def save(self, path, model_id):
        """
            save {model_id}-model in {path}
        """
        th.save(self.state_dict(), os.path.join(path, '{}-model'.format(model_id)))



