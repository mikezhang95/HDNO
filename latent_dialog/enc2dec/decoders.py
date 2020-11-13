import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from latent_dialog.enc2dec.base_modules import BaseRNN
from latent_dialog.utils import cast_type, LONG, FLOAT
from latent_dialog.woz_util import DECODING_MASKED_TOKENS, EOS


TEACH_FORCE = 'teacher_forcing'
TEACH_GEN = 'teacher_gen'
GEN = 'gen'
GEN_VALID = 'gen_valid'
LENGTH_AVERAGE = True
INF = 1e4
TOPK = 5 # topk sampling


# : this is the attn for decoder
class Attention(nn.Module):
    def __init__(self, dec_cell_size, ctx_cell_size, attn_mode, project):
        super(Attention, self).__init__()
        # : this del_cell_size is the hidden size of decoder
        self.dec_cell_size = dec_cell_size
        # : this ctx_cell_size is the embedding size of z 
        self.ctx_cell_size = ctx_cell_size
        self.attn_mode = attn_mode
        if project:
            # : [h, d] -> h
            self.linear_out = nn.Linear(dec_cell_size+ctx_cell_size, dec_cell_size)
        else:
            self.linear_out = None

        if attn_mode == 'general':
            self.dec_w = nn.Linear(dec_cell_size, ctx_cell_size)
        elif attn_mode == 'cat':
            self.dec_w = nn.Linear(dec_cell_size, dec_cell_size)
            self.attn_w = nn.Linear(ctx_cell_size, dec_cell_size)
            self.query_w = nn.Linear(dec_cell_size, 1)

    def forward(self, output, context):
        # output: (batch_size, output_seq_len, dec_cell_size)
        # context: (batch_size, max_ctx_len, ctx_cell_size)
        # : output = generative response with size (b, seq, dec_h)
        #           context = embedding of z with size (b, m, d)
        batch_size = output.size(0)
        max_ctx_len = context.size(1)

        if self.attn_mode == 'dot':
            attn = th.bmm(output, context.transpose(1, 2)) # (batch_size, output_seq_len, max_ctx_len)
        elif self.attn_mode == 'general':
            mapped_output = self.dec_w(output) # (batch_size, output_seq_len, ctx_cell_size)
            attn = th.bmm(mapped_output, context.transpose(1, 2)) # (batch_size, output_seq_len, max_ctx_len)
        elif self.attn_mode == 'cat':
            mapped_output = self.dec_w(output) # (batch_size, output_seq_len, dec_cell_size)
            mapped_attn = self.attn_w(context) # (batch_size, max_ctx_len, dec_cell_size)
            tiled_output = mapped_output.unsqueeze(2).repeat(1, 1, max_ctx_len, 1) # (batch_size, output_seq_len, max_ctx_len, dec_cell_size)
            tiled_attn = mapped_attn.unsqueeze(1) # (batch_size, 1, max_ctx_len, dec_cell_size)
            fc1 = F.tanh(tiled_output+tiled_attn) # (batch_size, output_seq_len, max_ctx_len, dec_cell_size)
            attn = self.query_w(fc1).squeeze(-1) # (batch_size, otuput_seq_len, max_ctx_len)
        else:
            raise ValueError('Unknown attention mode')

        # TODO mask
        # if self.mask is not None:

        # : eq.10 softmax attn
        attn = F.softmax(attn.view(-1, max_ctx_len), dim=1).view(batch_size, -1, max_ctx_len) # (batch_size, output_seq_len, max_ctx_len)

        # : eq.11
        mix = th.bmm(attn, context) # (batch_size, output_seq_len, ctx_cell_size)

        # : eq.12
        combined = th.cat((mix, output), dim=2) # (batch_size, output_seq_len, dec_cell_size+ctx_cell_size)
        if self.linear_out is None:
            return combined, attn
        else:
            output = F.tanh(
                self.linear_out(combined.view(-1, self.dec_cell_size+self.ctx_cell_size))).view(
                batch_size, -1, self.dec_cell_size) # (batch_size, output_seq_len, dec_cell_size)
            return output, attn


class DecoderRNN(BaseRNN):
    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p,
                 bidirectional, vocab_size, use_attn, ctx_cell_size, attn_mode, sys_id, eos_id, use_gpu,
                 max_dec_len, embedding=None):

        super(DecoderRNN, self).__init__(input_dropout_p=input_dropout_p, 
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

        self.use_attn = use_attn
        if self.use_attn:
            self.attention = Attention(dec_cell_size=hidden_size,
                                       # : this ctx_cell_size is the embedding size of z 
                                       ctx_cell_size=ctx_cell_size, 
                                       attn_mode=attn_mode, 
                                       project=True)
        
        self.dec_cell_size = hidden_size
        self.output_size = vocab_size
        self.project = nn.Linear(self.dec_cell_size, self.output_size)
        self.log_softmax = F.log_softmax

        self.sys_id = sys_id
        self.eos_id = eos_id
        self.use_gpu = use_gpu
        self.max_dec_len = max_dec_len

    # greedy and beam together
    def forward(self, batch_size, dec_inputs, dec_init_state, attn_context, mode, gen_type, beam_size, goal_hid=None):
        # dec_inputs: (batch_size, response_size-1)
        # attn_context: (batch_size, max_ctx_len, ctx_cell_size)
        # : attn_context is the embedding for the selected z
        # goal_hid: (batch_size, goal_nhid)
        # print ('This is the dec_inputs: ', dec_inputs.size())

        # mode: TEACH_FORCE/GEN/TEACH_GEN/GEN_VALID
        # gen_type: greedy/beam
        ret_dict = dict()

        if self.use_attn:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        vocab_size = self.output_size
        if gen_type != 'beam':
            beam_size = 1

        ##### Create Decoder Inputs #####
        # 1. decoder_input 
        if mode !=GEN and dec_inputs is not None:
            decoder_input = dec_inputs
        else:
            # prepare the BOS inputs
            with th.no_grad():
                bos_var = Variable(th.LongTensor([self.sys_id]))
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size*beam_size, 1).clone() # [b*k,1]
            
        # 2.decoder_hidden_state
        if mode == GEN and gen_type == 'beam':
            # beam search: repeat the initial states of the RNN
            # dec_init_state: [num_directions, batch, hidden_size]
            decoder_hidden_state = []
            for d in dec_init_state:
                dd = th.cat([d.squeeze(0)]*beam_size, dim=-1).view(1, batch_size*beam_size, -1) # [1,b*k,h]
                decoder_hidden_state.append(dd)
            decoder_hidden_state = tuple(decoder_hidden_state)
            if attn_context is not None:
                attn_context = th.cat([attn_context]*beam_size, dim=-1).reshape(-1, attn_context.shape[1], attn_context.shape[2]) # [b*k,y_size,d]

        else:
            # : dec_init_state is the embedding of selected z, with size (b, d)
            decoder_hidden_state = dec_init_state

        ##### Decode #####
        symbol_outputs, logits = [], []
        # 1. mode=TEACH_FORCE (for SL training)
        # : this is inconsistent to the statement in paper
        if mode == TEACH_FORCE:
            # : decoder_input = the utt of the groundtruth response
            #           decoder_hidden_state = the embedding of selected z, with size (b, d)
            #           attn_context = the full map of embedding of all possible z, with size (b, m, d)
            #           goal = None
            prob_outputs, decoder_hidden_state, attn, logits = self.forward_step(input_var=decoder_input, hidden_state=decoder_hidden_state, encoder_outputs=attn_context, goal_hid=goal_hid)


        # 2. mode=GEN (for validation and genration)
        else:   
            stored_scores, stored_symbols, stored_predecessors, stored_logits  = [],[],[],[]
            sequence_scores = th.zeros_like(decoder_input.squeeze(), dtype=th.float)# [b*k]
            sequence_scores.fill_(-INF)
            ind = th.LongTensor([i*beam_size for i in range(batch_size)])
            ind = cast_type(ind, LONG, self.use_gpu)
            sequence_scores.index_fill_(0, ind, 0.0)

            for step in range(self.max_dec_len):
                # one step for decoder 
                # --- decoder_input: [b*k,1]  decoder_hidden_state: tuple([1,b*k,h], [1,b*k,h])
                # --- decoder_output: [b*k,1,v]  step_logit: [b*k,1,v]
                decoder_output, decoder_hidden_state, step_attn, step_logit = self.forward_step(decoder_input, decoder_hidden_state, attn_context, goal_hid=goal_hid)

                # greedy search
                if gen_type == "greedy":
                    log_softmax = decoder_output.squeeze()
                    _, symbols = log_softmax.topk(1) # [b,1]
                    # 1. create next step input
                    decoder_input = symbols
                    # 2. save
                    stored_symbols.append(symbols.clone()) # [b,1]
                    stored_scores.append(log_softmax) # [b,v]
                    stored_logits.append(step_logit) # [b*k,1,v]

                # beam search
                elif gen_type == "beam":
                    log_softmax = decoder_output.squeeze() # [b*k,v]
                    # update sequence socres
                    sequence_scores = sequence_scores.unsqueeze(1).repeat(1, vocab_size) # [b*k,v]
                    if LENGTH_AVERAGE:
                        t = step + 2
                        sequence_scores = sequence_scores * (1 - 1/t) + log_softmax / t
                    else:
                        sequence_scores += log_softmax

                    # diverse beam search: penalize short sequence.

                    # select topk
                    scores, candidates = sequence_scores.view(batch_size, -1).topk(beam_size, dim=1) #[b,k], [b,k]

                    # 1.1 create  next step decoder_input
                    input_var = (candidates % vocab_size) #  [b,k] 
                    decoder_input= input_var.view(batch_size * beam_size, 1) # [b*k,1] input for next step

                    # 1.2 create next step decoder_hidden_state
                    pp = candidates // vocab_size # [b,k]
                    predecessors = pp.clone()
                    for b, p in enumerate(pp):
                        predecessors[b] = p + b * beam_size
                    predecessors = predecessors.view(-1) # [b*k]
                    survived_state = []
                    for d in decoder_hidden_state:
                        survived_state.append(d.index_select(1, predecessors))
                    decoder_hidden_state = tuple(survived_state)

                    # 1.3 create next step scores
                    sequence_scores = scores.view(batch_size * beam_size) # [b*k]
                    # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
                    stored_scores.append(sequence_scores.clone()) # [b*k]
                    eos_indices = input_var.data.eq(self.eos_id).view(-1)  # [b*k]
                    if eos_indices.nonzero().dim() > 0:
                        sequence_scores.masked_fill_(eos_indices, -INF)

                    # 2. Cache results for backtracking
                    stored_predecessors.append(predecessors) # [b*k]
                    stored_symbols.append(decoder_input.squeeze()) # [b*k]

                elif gen_type == "sample":
                    log_softmax = decoder_output.squeeze()
                    topk_log, topk_words = log_softmax.topk(TOPK) # [b,topk]
                    word_idx = th.multinomial(th.exp(topk_log), 1, replacement=True) # [b,1]
                    symbols = th.gather(topk_words, 1, word_idx) # [b,1]
                    # 1. create next step input
                    decoder_input = symbols
                    # 2. save
                    stored_symbols.append(symbols.clone()) # [b,1]
                    stored_scores.append(log_softmax) # [b,v]
                    stored_logits.append(step_logit) # [b*k,1,v]

                else:
                    raise NotImplementedError

            if gen_type == "greedy" or gen_type == "sample":
                symbol_outputs = th.cat(stored_symbols, dim=1).squeeze() # [b,len]
                prob_outputs = 0 # dontcare
                logits = th.cat(stored_logits, dim=1) # [b,t,v]

            elif gen_type == "beam":
                # beam search backtrack
                predicts, lengths, scores =  self._backtrack(
                        stored_predecessors, stored_symbols, stored_scores, batch_size, beam_size)
                # only select top1 for beam search
                symbol_outputs = predicts[:,0,:] # [b,len]
                prob_outputs = 0 # dontcare logits for beam search generation
                logits = 0 # dontcare logits for beam search generation
            else:
                raise NotImplementedError

        # : store logits
        ret_dict['logits'] = logits
        ret_dict[DecoderRNN.KEY_SEQUENCE] = symbol_outputs

        # prob_outputs: (batch_size, max_dec_len, vocab_size)
        # decoder_hidden_state: tuple: (h, c)
        # ret_dict[DecoderRNN.KEY_ATTN_SCORE]: max_dec_len*(batch_size, 1, max_ctx_len)
        # ret_dict[DecoderRNN.KEY_SEQUENCE]: max_dec_len*(batch_size) 
        return prob_outputs, decoder_hidden_state, ret_dict

    # beam search backtrack
    def _backtrack(self, predecessors, symbols, scores, batch_size, beam_size):
        p = list()
        l = [[self.max_dec_len] * beam_size for _ in range(batch_size)] # length of each seq

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(batch_size, beam_size).topk(beam_size, dim=-1)

        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone() # [b,k]

        # the number of EOS found in the backward loop below for each batch
        batch_eos_found = [0] * batch_size

        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = sorted_idx.clone()
        for b, idx in enumerate(sorted_idx):
            t_predecessors[b] = idx + b * beam_size
        t_predecessors = t_predecessors.view(-1) # [b*k]

        t = self.max_dec_len - 1
        while t >= 0 :
            # Re-order the variables with the back pointer
            current_symbol = symbols[t].index_select(0, t_predecessors) # [b*k]
            # Re-order the back pointer of the previous step with the back pointer of the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors) # [b*k]


            # Deal with EOS
            eos_indices = symbols[t].data.eq(self.eos_id).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = idx[0].item() // beam_size 
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = beam_size - (batch_eos_found[b_idx] % beam_size) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * beam_size + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_symbol[res_idx] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]]
                    l[b_idx][res_k_idx] = t + 1

            # save current_symbol 
            p.append(current_symbol) # [b*k]

            t -= 1

        # Sort and re-order again as the added ended sequences may change the order (very unlikely)
        s, re_sorted_idx = s.topk(beam_size)
        for b_idx in range(batch_size):
            l[b_idx] = [l[b_idx][k_idx.item()]
                        for k_idx in re_sorted_idx[b_idx, :]]
        rr = re_sorted_idx.clone()
        for b, idx in enumerate(re_sorted_idx):
            rr[b] = idx + b * beam_size
        re_sorted_idx = rr.view(-1) # [b*k]

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        predicts = th.stack(p[::-1]).t() # [b*k,t]
        predicts = predicts[re_sorted_idx].contiguous().view(batch_size, beam_size, -1)
        scores = s
        lengths = l
        return predicts, scores, lengths


    def forward_step(self, input_var, hidden_state, encoder_outputs, goal_hid):
        # input_var: (batch_size, response_size-1 i.e. output_seq_len)
        # hidden_state: tuple: (h, c)
        # encoder_outputs: (batch_size, max_ctx_len, ctx_cell_size), : (b, m, k)
        # goal_hid: (batch_size, goal_nhid)
        batch_size, output_seq_len = input_var.size()
        embedded = self.embedding(input_var) # (batch_size, output_seq_len, embedding_dim)

        # add goals
        if goal_hid is not None:
            goal_hid = goal_hid.view(goal_hid.size(0), 1, goal_hid.size(1)) # (batch_size, 1, goal_nhid)
            goal_rep = goal_hid.repeat(1, output_seq_len, 1) # (batch_size, output_seq_len, goal_nhid)
            embedded = th.cat([embedded, goal_rep], dim=2) # (batch_size, output_seq_len, embedding_dim+goal_nhid)

        embedded = self.input_dropout(embedded)

        # ############
        # embedded = self.FC(embedded.view(-1, embedded.size(-1))).view(batch_size, output_seq_len, -1)

        # output: (batch_size, output_seq_len, dec_cell_size)
        # hidden: tuple: (h, c)

        # print ('This is the embedding size: ', embedded.size())
        output, hidden_s = self.rnn(embedded, hidden_state)

        # print ('This is the output size: ', output.size())
        
        attn = None
        if self.use_attn:
            # output: (batch_size, output_seq_len, dec_cell_size)
            # encoder_outputs: (batch_size, max_ctx_len, ctx_cell_size)
            # attn: (batch_size, output_seq_len, max_ctx_len)
            output, attn = self.attention(output, encoder_outputs)

        logits = self.project(output.contiguous().view(-1, self.dec_cell_size)) # (batch_size*output_seq_len, vocab_size)
        prediction = self.log_softmax(logits, dim=logits.dim()-1).view(batch_size, output_seq_len, -1) # (batch_size, output_seq_len, vocab_size)
        return prediction, hidden_s, attn, logits.view(batch_size, output_seq_len, -1)


    # special for rl
    def _step(self, input_var, hidden_state, encoder_outputs, goal_hid):
        # input_var: (1, 1)
        # hidden_state: tuple: (h, c)
        # encoder_outputs: (1, max_dlg_len, dlg_cell_size)
        # goal_hid: (1, goal_nhid)
        batch_size, output_seq_len = input_var.size()
        embedded = self.embedding(input_var) # (1, 1, embedding_dim)

        if goal_hid is not None:
            goal_hid = goal_hid.view(goal_hid.size(0), 1, goal_hid.size(1)) # (1, 1, goal_nhid)
            goal_rep = goal_hid.repeat(1, output_seq_len, 1) # (1, 1, goal_nhid)
            embedded = th.cat([embedded, goal_rep], dim=2) # (1, 1, embedding_dim+goal_nhid)

        embedded = self.input_dropout(embedded)

        # ############
        # embedded = self.FC(embedded.view(-1, embedded.size(-1))).view(batch_size, output_seq_len, -1)

        # output: (1, 1, dec_cell_size)
        # hidden: tuple: (h, c)
        output, hidden_s = self.rnn(embedded, hidden_state)

        attn = None
        if self.use_attn:
            # output: (1, 1, dec_cell_size)
            # encoder_outputs: (1, max_dlg_len, dlg_cell_size)
            # attn: (1, 1, max_dlg_len)
            output, attn = self.attention(output, encoder_outputs)

        logits = self.project(output.view(-1, self.dec_cell_size)) # (1*1, vocab_size)
        prediction = logits.view(batch_size, output_seq_len, -1) # (1, 1, vocab_size)
        # prediction = self.log_softmax(logits, dim=logits.dim()-1).view(batch_size, output_seq_len, -1) # (batch_size, output_seq_len, vocab_size)
        return prediction, hidden_s

    # special for rl
    def write(self, input_var, hidden_state, encoder_outputs, max_words, vocab, stop_tokens, goal_hid=None, mask=True,
              decoding_masked_tokens=DECODING_MASKED_TOKENS):
        # input_var: (1, 1)
        # hidden_state: tuple: (h, c)
        # encoder_outputs: max_dlg_len*(1, 1, dlg_cell_size)
        # goal_hid: (1, goal_nhid)
        logprob_outputs = [] # list of logprob | max_dec_len*(1, )
        symbol_outputs = [] # list of word ids | max_dec_len*(1, )
        decoder_input = input_var
        decoder_hidden_state = hidden_state
        if type(encoder_outputs) is list:
            encoder_outputs = th.cat(encoder_outputs, 1) # (1, max_dlg_len, dlg_cell_size)
        # print('encoder_outputs.size() = {}'.format(encoder_outputs.size()))
        
        if mask:
            special_token_mask = Variable(th.FloatTensor([-999. if token in decoding_masked_tokens else 0. for token in vocab]))
            special_token_mask = cast_type(special_token_mask, FLOAT, self.use_gpu) # (vocab_size, )

        def _sample(dec_output, num_i):
            # dec_output: (1, 1, vocab_size), need to softmax and log_softmax
            dec_output = dec_output.view(-1) # (vocab_size, )
            # TODO temperature
            prob = F.softmax(dec_output/0.6, dim=0) # (vocab_size, )
            logprob = F.log_softmax(dec_output, dim=0) # (vocab_size, )
            symbol = prob.multinomial(num_samples=1).detach() # (1, )
            # _, symbol = prob.topk(1) # (1, )
            # _, tmp_symbol = prob.topk(1) # (1, )
            # print('multinomial symbol = {}, prob = {}'.format(symbol, prob[symbol.item()]))
            # print('topk symbol = {}, prob = {}'.format(tmp_symbol, prob[tmp_symbol.item()]))
            logprob = logprob.gather(0, symbol) # (1, )
            return logprob, symbol

        for i in range(max_words):
            decoder_output, decoder_hidden_state = self._step(decoder_input, decoder_hidden_state, encoder_outputs, goal_hid)
            # disable special tokens from being generated in a normal turn
            if mask:
                decoder_output += special_token_mask.expand(1, 1, -1)
            logprob, symbol = _sample(decoder_output, i)
            logprob_outputs.append(logprob)
            symbol_outputs.append(symbol)
            decoder_input = symbol.view(1, -1)

            if vocab[symbol.item()] in stop_tokens:
                break

        assert len(logprob_outputs) == len(symbol_outputs)
        # logprob_list = [t.item() for t in logprob_outputs]
        logprob_list = logprob_outputs
        symbol_list = [t.item() for t in symbol_outputs]
        return logprob_list, symbol_list

    # For MultiWoz RL
    def forward_rl(self, batch_size, dec_init_state, attn_context, vocab, max_words, goal_hid=None, mask=True, temp=0.1):
        # prepare the BOS inputs
        with th.no_grad():
            bos_var = Variable(th.LongTensor([self.sys_id]))
        bos_var = cast_type(bos_var, LONG, self.use_gpu)
        decoder_input = bos_var.expand(batch_size, 1) # (1, 1)
        decoder_hidden_state = dec_init_state # tuple: (h, c)
        encoder_outputs = attn_context # (1, ctx_len, ctx_cell_size)

        logprob_outputs = [] # list of logprob | max_dec_len*(1, )
        symbol_outputs = [] # list of word ids | max_dec_len*(1, )
        log_softmax_outputs = [] # : store the softmax vec

        if mask:
            special_token_mask = Variable(th.FloatTensor([-999. if token in DECODING_MASKED_TOKENS else 0. for token in vocab]))
            special_token_mask = cast_type(special_token_mask, FLOAT, self.use_gpu) # (vocab_size, )

        def _sample(dec_output, num_i):
            # dec_output: (1, 1, vocab_size), need to softmax and log_softmax
            dec_output = dec_output.view(batch_size, -1) # (batch_size, vocab_size, )
            prob = F.softmax(dec_output/temp, dim=1) # (batch_size, vocab_size, )
            # print ('This is the temp: ', temp)
            # print ('This is the size of prob: ', prob.size())
            logprob = F.log_softmax(dec_output, dim=1) # (batch_size, vocab_size, )
            symbol = prob.multinomial(num_samples=1).detach() # (batch_size, 1)
            # _, symbol = prob.topk(1) # (1, )
            # _, tmp_symbol = prob.topk(1) # (1, )
            # print('multinomial symbol = {}, prob = {}'.format(symbol, prob[symbol.item()]))
            # print('topk symbol = {}, prob = {}'.format(tmp_symbol, prob[tmp_symbol.item()]))
            logprob_ = logprob.gather(1, symbol) # (1, )
            # return logprob, symbol
            # : output the original softmax distr
            return logprob_, symbol, logprob

        stopped_samples = set()
        for i in range(max_words):
            decoder_output, decoder_hidden_state = self._step(decoder_input, decoder_hidden_state, encoder_outputs, goal_hid)
            # disable special tokens from being generated in a normal turn
            if mask:
                decoder_output += special_token_mask.expand(1, 1, -1)
            # logprob, symbol = _sample(decoder_output, i)
            # 
            logprob, symbol, logprob_vec = _sample(decoder_output, i)
            logprob_outputs.append(logprob)
            symbol_outputs.append(symbol)
            log_softmax_outputs.append(logprob_vec)
            decoder_input = symbol.view(batch_size, -1)
            for b_id in range(batch_size):
                if vocab[symbol[b_id].item()] == EOS:
                    stopped_samples.add(b_id)

            if len(stopped_samples) == batch_size:
                break

        # assert len(logprob_outputs) == len(symbol_outputs)
        # 
        assert len(logprob_outputs) == len(symbol_outputs) == len(log_softmax_outputs)
        symbol_outputs = th.cat(symbol_outputs, dim=1).cpu().data.numpy().tolist()
        logprob_outputs = th.cat(logprob_outputs, dim=1)
        # print ('This is the size of logprob: ', logprob_outputs.size())
        # 
        log_softmax_outputs = th.stack(log_softmax_outputs, dim=1)
        # print ('This is the size of log_softmax: ', log_softmax_outputs.size())
        logprob_list = []
        symbol_list = []
        # 
        log_softmax_list = []
        for b_id in range(batch_size):
            b_logprob = []
            b_symbol = []
            # 
            b_log_softmax = []
            for t_id in range(logprob_outputs.shape[1]):
                symbol = symbol_outputs[b_id][t_id]
                if vocab[symbol] == EOS and t_id != 0:
                    break

                b_symbol.append(symbol_outputs[b_id][t_id])
                b_logprob.append(logprob_outputs[b_id][t_id])
                # 
                b_log_softmax.append(log_softmax_outputs[b_id][t_id])

            logprob_list.append(b_logprob)
            symbol_list.append(b_symbol)
            # 
            log_softmax_list.append(b_log_softmax)

        # TODO backward compatible, if batch_size == 1, we remove the nested structure
        if batch_size == 1:
            logprob_list = logprob_list[0]
            symbol_list = symbol_list[0]
            # 
            log_softmax_list = log_softmax_list[0]

        return logprob_list, symbol_list, log_softmax_list
