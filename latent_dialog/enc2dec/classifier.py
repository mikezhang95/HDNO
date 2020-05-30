import torch as th
import torch.nn as nn
import torch.nn.functional as F
from latent_dialog.enc2dec.base_modules import BaseRNN


class EncoderGRUATTN(BaseRNN):
    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional, variable_lengths):
        super(EncoderGRUATTN, self).__init__(input_dropout_p=input_dropout_p, 
                                             rnn_cell=rnn_cell, 
                                             input_size=input_size, 
                                             hidden_size=hidden_size, 
                                             num_layers=num_layers, 
                                             output_dropout_p=output_dropout_p, 
                                             bidirectional=bidirectional)
        self.variable_lengths = variable_lengths
        self.nhid_attn = hidden_size
        self.output_size = hidden_size*2 if bidirectional else hidden_size

        # attention to combine selection hidden states
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), 
            nn.Tanh(), 
            nn.Linear(hidden_size, 1)
        )

    def forward(self, residual_var, input_var, turn_feat, mask=None, init_state=None, input_lengths=None):
        # residual_var: (batch_size, max_dlg_len, 2*utt_cell_size)
        # input_var: (batch_size, max_dlg_len, dlg_cell_size)

        # TODO switch of mask
        # mask = None
        
        require_embed = True
        if require_embed:
            # input_cat = th.cat([input_var, residual_var], 2) # (batch_size, max_dlg_len, dlg_cell_size+2*utt_cell_size)
            input_cat = th.cat([input_var, residual_var, turn_feat], 2) # (batch_size, max_dlg_len, dlg_cell_size+2*utt_cell_size)
        else:
            # input_cat = th.cat([input_var], 2)
            input_cat = th.cat([input_var, turn_feat], 2)
        if mask is not None:
            input_mask = mask.view(input_cat.size(0), input_cat.size(1), 1) # (batch_size, max_dlg_len*max_utt_len, 1)
            input_cat = th.mul(input_cat, input_mask)
        embedded = self.input_dropout(input_cat)
        
        require_rnn = True
        if require_rnn:
            if init_state is not None:
                h, _ = self.rnn(embedded, init_state)
            else:
                h, _ = self.rnn(embedded) # (batch_size, max_dlg_len, 2*nhid_attn)
    
            logit = self.attn(h.contiguous().view(-1, 2*self.nhid_attn)).view(h.size(0), h.size(1)) # (batch_size, max_dlg_len)
            # if mask is not None:
            #     logit_mask = mask.view(input_cat.size(0), input_cat.size(1))
            #     logit_mask = -999.0 * logit_mask
            #     logit = logit_mask + logit
    
            prob = F.softmax(logit, dim=1).unsqueeze(2).expand_as(h) # (batch_size, max_dlg_len, 2*nhid_attn)
            attn = th.sum(th.mul(h, prob), 1) # (batch_size, 2*nhid_attn)
            
            return attn

        else:
            logit = self.attn(embedded.contiguous().view(input_cat.size(0)*input_cat.size(1), -1)).view(input_cat.size(0), input_cat.size(1))
            if mask is not None:
                logit_mask = mask.view(input_cat.size(0), input_cat.size(1))
                logit_mask = -999.0 * logit_mask
                logit = logit_mask + logit

            prob = F.softmax(logit, dim=1).unsqueeze(2).expand_as(embedded) # (batch_size, max_dlg_len, 2*nhid_attn)
            attn = th.sum(th.mul(embedded, prob), 1) # (batch_size, 2*nhid_attn)
            
            return attn
            


class Discriminator(BaseRNN):
    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p,
                 bidirectional, vocab_size, ctx_cell_size, sys_id, eos_id, use_gpu,
                 max_dec_len, embedding=None):
        super(Discriminator, self).__init__(input_dropout_p=input_dropout_p, 
                                            rnn_cell=rnn_cell, 
                                            input_size=input_size, 
                                            hidden_size=hidden_size, 
                                            num_layers=num_layers, 
                                            output_dropout_p=output_dropout_p, 
                                            bidirectional=bidirectional)

        # TODO embedding is None or not
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, input_size)
        else:
            self.embedding = embedding

        # share parameters between encoder and decoder
        # self.rnn = ctx_encoder.rnn
        # self.FC = nn.Linear(input_size, utt_encoder.output_size)
        
        self.dec_cell_size = hidden_size
        self.output_size = vocab_size
        self.project = nn.Linear(self.dec_cell_size, self.output_size)
        self.log_softmax = F.log_softmax

        self.sys_id = sys_id
        self.eos_id = eos_id
        self.use_gpu = use_gpu
        self.max_dec_len = max_dec_len
        
    def forward(self, batch_size, dec_inputs, dec_init_state):
        # dec_inputs: (batch_size, response_size-1)
        # attn_context: (batch_size, max_ctx_len, ctx_cell_size)
        # : attn_context is the embedding for the selected z
        # goal_hid: (batch_size, goal_nhid)

        # if mode == GEN:
        #     dec_inputs = None

        if dec_inputs is not None:
            decoder_input = dec_inputs
        else:
            # prepare the BOS inputs
            with th.no_grad():
                bos_var = Variable(th.LongTensor([self.sys_id]))
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size, 1) # (batch_size, 1)

        decoder_hidden_state = dec_init_state

        prob_outputs, prob_logits = self.forward_step(input_var=decoder_input, hidden_state=decoder_hidden_state)

        # prob_outputs: (batch_size, max_dec_len, vocab_size)
        # decoder_hidden_state: tuple: (h, c)
        # ret_dict[DecoderRNN.KEY_SEQUENCE]: max_dec_len*(batch_size, 1) 
        return prob_outputs, prob_logits

    def forward_step(self, input_var, hidden_state):
        # input_var: (batch_size, response_size-1 i.e. output_seq_len)
        # hidden_state: tuple: (h, c)
        # encoder_outputs: (batch_size, max_ctx_len, ctx_cell_size), : (b, m, k)
        if type(input_var)==list:
            logits = []
            predictions = []
            i = 0
            for b_var in input_var:
                # output_seq_len = len(b_r)
                # b_r = th.stack(b_r, dim=0).unsqueeze(0)
                b_var = b_var.unsqueeze(0)
                _, output_seq_len = b_var.size()
                embedded = self.embedding(b_var)
                # print ('This is the size of embedding: ', embedded.size())
                # print ('This is the size of hidden_state: ', hidden_state[0].size())
                embedded = self.input_dropout(embedded)
                output, hidden_s = self.rnn(embedded, hidden_state)
                step_logit = self.project(output.contiguous().view(-1, self.dec_cell_size))
                step_prediction = self.log_softmax(step_logit, dim=-1).view(output_seq_len, -1)
                i += 1
                logits.append(step_logit)
                predictions.append(step_prediction)
            return predictions, logits
        else:
            batch_size, output_seq_len = input_var.size()
            embedded = self.embedding(input_var) # (batch_size, output_seq_len, embedding_dim)

            embedded = self.input_dropout(embedded)

            # output: (batch_size, output_seq_len, dec_cell_size)
            # hidden: tuple: (h, c)
            output, hidden_s = self.rnn(embedded, hidden_state)

            # print ('This is the output size: ', output.size())

            logits = self.project(output.contiguous().view(-1, self.dec_cell_size)) # (batch_size*output_seq_len, vocab_size)
            prediction = self.log_softmax(logits, dim=logits.dim()-1).view(batch_size, output_seq_len, -1) # (batch_size, output_seq_len, vocab_size)
            return prediction, logits.view(batch_size, output_seq_len, -1)
