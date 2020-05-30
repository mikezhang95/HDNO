import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from latent_dialog.enc2dec.base_modules import BaseRNN


class EncoderRNN(BaseRNN):
    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional, variable_lengths):
        super(EncoderRNN, self).__init__(input_dropout_p=input_dropout_p, 
                                         rnn_cell=rnn_cell, 
                                         input_size=input_size, 
                                         hidden_size=hidden_size, 
                                         num_layers=num_layers, 
                                         output_dropout_p=output_dropout_p, 
                                         bidirectional=bidirectional)
        self.variable_lengths = variable_lengths
        self.output_size = hidden_size*2 if bidirectional else hidden_size

    def forward(self, input_var, init_state=None, input_lengths=None, goals=None):
        # add goals
        # if goals is not None:
        #     batch_size, max_ctx_len, ctx_nhid = input_var.size()
        #     goals = goals.view(goals.size(0), 1, goals.size(1))
        #     goals_rep = goals.repeat(1, max_ctx_len, 1).view(batch_size, max_ctx_len, -1) # (batch_size, max_ctx_len, goal_nhid)
        #     input_var = th.cat([input_var, goals_rep], dim=2)

        embedded = self.input_dropout(input_var)

        # if self.variable_lengths:
        #     embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths,
        #                                                  batch_first=True)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        # if self.variable_lengths:
        #     output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden


class RnnUttEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feat_size, goal_nhid, rnn_cell,
                 utt_cell_size, num_layers, input_dropout_p, output_dropout_p,
                 bidirectional, variable_lengths, use_attn, embedding=None):
        super(RnnUttEncoder, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p,
                              rnn_cell=rnn_cell, 
                              input_size=embedding_dim+feat_size+goal_nhid, 
                              hidden_size=utt_cell_size, 
                              num_layers=num_layers, 
                              output_dropout_p=output_dropout_p, 
                              bidirectional=bidirectional, 
                              variable_lengths=variable_lengths)

        self.utt_cell_size = utt_cell_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.utt_cell_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, feats=None, init_state=None, goals=None):
        batch_size, max_ctx_len, max_utt_len = utterances.size()
        # print ('This is the batch size: {}, max_ctx_len: {} and max_utt_len: {}.'.format(batch_size, max_ctx_len, max_utt_len))
        # get word embeddings
        flat_words = utterances.view(-1, max_utt_len) # (batch_size, max_utt_len)
        # print ('This is the flat words size: {}.'.format(flat_words.size()))
        word_embeddings = self.embedding(flat_words) # (batch_size, max_utt_len, embedding_dim)
        # print ('This is the word_embeddings size: {}.'.format(word_embeddings.size()))
        flat_mask = th.sign(flat_words).float()
        # # add features
        # if feats is not None:
        #     flat_feats = feats.view(-1, 1) # (batch_size*max_ctx_len, 1)
        #     flat_feats = flat_feats.unsqueeze(1).repeat(1, max_utt_len, 1) # (batch_size*max_ctx_len, max_utt_len, 1)
        #     word_embeddings = th.cat([word_embeddings, flat_feats], dim=2) # (batch_size*max_ctx_len, max_utt_len, embedding_dim+1)

        # # add goals
        # if goals is not None:
        #     goals = goals.view(goals.size(0), 1, 1, goals.size(1))
        #     goals_rep = goals.repeat(1, max_ctx_len, max_utt_len, 1).view(batch_size*max_ctx_len, max_utt_len, -1) # (batch_size*max_ctx_len, max_utt_len, goal_nhid)
        #     word_embeddings = th.cat([word_embeddings, goals_rep], dim=2)

        # enc_outs: (batch_size, max_utt_len, num_directions*utt_cell_size)
        # enc_last: (num_layers*num_directions, batch_size*max_ctx_len, utt_cell_size)
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)
        
        # print ('This is the size of enc_last: {}.'.format(enc_last.size()))
        # print ('This is the size of enc_outs: {}.'.format(enc_outs.size()))

        if self.use_attn:
            fc1 = th.tanh(self.key_w(enc_outs)) # (batch_size, max_utt_len, utt_cell_size)
            # print ('This is the size of fc1: {}.'.format(fc1.size()))
            attn = self.query(fc1).squeeze(2)  # (batch_size, max_utt_len)
            # print ('This is the size of attn: {}.'.format(attn.size()))
            attn = F.softmax(attn, attn.dim()-1) # (batch_size, max_utt_len, 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True)+1e-10)).unsqueeze(2)
            # print ('This is the unsqueezed attn size: {}.'.format(attn.size()))
            utt_embedded = attn * enc_outs # (batch_size*max_ctx_len, max_utt_len, num_directions*utt_cell_size)
            utt_embedded = th.sum(utt_embedded, dim=1) # (batch_size*max_ctx_len, num_directions*utt_cell_size)
        else:
            # FIXME bug for multi-layer
            attn = None
            utt_embedded = enc_last.transpose(0, 1).contiguous() # (batch_size*max_ctx_lens, num_layers*num_directions, utt_cell_size)
            utt_embedded = utt_embedded.view(-1, self.output_size) # (batch_size*max_ctx_len*num_layers, num_directions*utt_cell_size)

        utt_embedded = utt_embedded.view(batch_size, max_ctx_len, self.output_size)
        return utt_embedded, word_embeddings.contiguous().view(batch_size, max_ctx_len*max_utt_len, -1), \
               enc_outs.contiguous().view(batch_size, max_ctx_len*max_utt_len, -1)


class RnnUttMixedEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, bs_size, db_size, rnn_cell,
                 utt_cell_size, num_layers, input_dropout_p, output_dropout_p,
                 bidirectional, variable_lengths, use_attn, embedding=None):
        super(RnnUttMixedEncoder, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p,
                              rnn_cell=rnn_cell, 
                              input_size=embedding_dim*3, 
                              hidden_size=utt_cell_size, 
                              num_layers=num_layers, 
                              output_dropout_p=output_dropout_p, 
                              bidirectional=bidirectional, 
                              variable_lengths=variable_lengths)

        self.utt_cell_size = utt_cell_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.utt_cell_size
        self.use_attn = use_attn
        if self.use_attn:
            self.self_attention = SelfAttn(utt_cell_size, utt_cell_size)
            self.inter_attention_bs = InterAttn(utt_cell_size, bs_size, embedding_dim)
            self.inter_attention_db = InterAttn(utt_cell_size, db_size, embedding_dim)
            self.key_w = nn.Linear(self.output_size, self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    @staticmethod
    def get_index(one_hot):
        one_hot_indice = []
        for b_one_hot in one_hot:
            index = b_one_hot.nonzero().squeeze(-1)
            pad = th.zeros(one_hot.size(1)-index.size(0)).long().cuda()
            one_hot_indice.append(th.cat([index, pad], dim=0))
        one_hot_indice = th.stack(one_hot_indice)
        return one_hot_indice

    def forward(self, utterances, bs=None, db=None, init_state=None):
        batch_size, max_ctx_len, max_utt_len = utterances.size()
        # print ('This is the batch size: {}, max_ctx_len: {} and max_utt_len: {}.'.format(batch_size, max_ctx_len, max_utt_len))
        # get word embeddings
        flat_words = utterances.view(-1, max_utt_len) # (batch_size, max_utt_len)
        # print ('This is the flat words size: {}.'.format(flat_words.size()))
        word_embeddings = self.embedding(flat_words) # (batch_size, max_utt_len, embedding_dim)
        # print ('This is the word_embeddings size: {}.'.format(word_embeddings.size()))
        flat_mask = th.sign(flat_words).float()
        # print ('This is the flat_mask size: {}.'.format(flat_mask.size()))
        # enc_outs: (batch_size, max_utt_len, num_directions*utt_cell_size)
        # enc_last: (num_layers*num_directions, batch_size*max_ctx_len, utt_cell_size)
        # enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)
        
        # print ('This is the size of enc_last: {}.'.format(enc_last.size()))
        # print ('This is the size of enc_outs: {}.'.format(enc_outs.size()))

        if self.use_attn:
            if init_state == None:
                init_state = th.zeros(batch_size, 1, self.utt_cell_size).cuda()
            hiddens = [init_state]
            hidden_state = init_state
            bs_indice = self.get_index(bs)
            db_indice = self.get_index(db)
            for t_id in range(word_embeddings.size(1)):
                # self_attn_values, self_attn = self.self_attention(hidden_state, th.cat(hiddens, dim=1), attn_mask=flat_mask[:, t_id:t_id+1])
                bs_attn_values, bs_attn = self.inter_attention_bs(hidden_state, bs_indice, attn_mask=flat_mask[:, t_id:t_id+1])
                # print ('This is the size of bs_attn_values: {}.'.format(bs_attn_values.size()))
                db_attn_values, db_attn = self.inter_attention_db(hidden_state, db_indice, attn_mask=flat_mask[:, t_id:t_id+1])
                # print ('This is the size of db_attn_values: {}.'.format(db_attn_values.size()))
                enc_outs, enc_last = self.rnn(th.cat([word_embeddings[:, t_id:t_id+1, :],bs_attn_values,db_attn_values], dim=-1), init_state=hidden_state.transpose(0, 1))
                # print ('This is the size of enc_outs: {}'.format(enc_outs.size()))
                # print ('This is the size of enc_last: {}'.format(enc_last.size()))
                hidden_state = enc_outs
                hiddens.append(enc_outs)
            # print ('This is the enc_last size: ', enc_last.size())
            hidden_states = th.cat(hiddens[1:], dim=1)
            fc1 = th.tanh(self.key_w(hidden_states)) # (batch_size, max_utt_len, utt_cell_size)
            # print ('This is the size of fc1: {}.'.format(fc1.size()))
            attn = self.query(fc1).squeeze(2)  # (batch_size, max_utt_len)
            # print ('This is the size of attn: {}.'.format(attn.size()))
            attn = F.softmax(attn, attn.dim()-1) # (batch_size, max_utt_len, 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True)+1e-10)).unsqueeze(2)
            # print ('This is the unsqueezed attn size: {}.'.format(attn.size()))
            utt_embedded = attn * enc_outs # (batch_size*max_ctx_len, max_utt_len, num_directions*utt_cell_size)
            utt_embedded = th.sum(utt_embedded, dim=1) # (batch_size*max_ctx_len, num_directions*utt_cell_size)
        else:
            # FIXME bug for multi-layer
            attn = None
            enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)
            utt_embedded = enc_last.transpose(0, 1).contiguous() # (batch_size, num_layers*num_directions, utt_cell_size)
            utt_embedded = utt_embedded.view(-1, self.output_size) # (batch_size*num_layers, num_directions*utt_cell_size)

        utt_embedded = utt_embedded.view(batch_size, max_ctx_len, self.output_size)
        # print ('This is the output utt_embedded size: {}.'.format(utt_embedded.size()))
        return utt_embedded, word_embeddings.contiguous().view(batch_size, max_ctx_len*max_utt_len, -1), \
               enc_outs.contiguous().view(batch_size, max_ctx_len*max_utt_len, -1)
