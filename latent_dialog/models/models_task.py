import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from latent_dialog.woz_util  import SYS, EOS, PAD, BOS
from latent_dialog.utils import INT, FLOAT, LONG, Pack, cast_type
from latent_dialog.enc2dec.encoders import RnnUttEncoder
from latent_dialog.enc2dec.decoders import DecoderRNN, GEN, TEACH_FORCE
from latent_dialog.criterions import NLLEntropy, CatKLLoss, Entropy, NormKLLoss
from latent_dialog import nn_lib
from latent_dialog.models import BaseModel
from latent_dialog.enc2dec.classifier import Discriminator



class HDNO(BaseModel):
    def __init__(self, corpus, config):
        super(HDNO, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.beta = config.beta
        self.gamma = config.gamma
        self.simple_posterior = config.simple_posterior
        
        self.init_net()
    

    def init_net(self):
        ### encoder
        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=self.config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=self.config.utt_rnn_cell,
                                         utt_cell_size=self.config.utt_cell_size,
                                         num_layers=self.config.num_layers,
                                         input_dropout_p=self.config.dropout,
                                         output_dropout_p=self.config.dropout,
                                         bidirectional=self.config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=self.config.enc_use_attn,
                                         embedding=self.embedding)
        
        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + + self.bs_size + self.db_size, 
                                          self.config.y_size, is_lstm=False)

        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)

        ### decoder 
        self.z_embedding_x2y = nn.Linear(self.y_size + self.utt_encoder.output_size + self.bs_size + self.db_size, self.config.dec_cell_size, bias=True)

        self.decoder_x2y = DecoderRNN(input_dropout_p=self.config.dropout,
                                  rnn_cell=self.config.dec_rnn_cell,
                                  input_size=self.config.embed_size,
                                  hidden_size=self.config.dec_cell_size,
                                  num_layers=self.config.num_layers,
                                  output_dropout_p=self.config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=self.config.dec_use_attn,
                                  ctx_cell_size=self.config.dec_cell_size,
                                  attn_mode=self.config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=self.config.use_gpu,
                                  max_dec_len=self.config.max_dec_len,
                                  embedding=self.embedding)

        if self.config.disc:
            # define discriminator
            self.discriminator = Discriminator(input_dropout_p=self.config.dropout,
                                            rnn_cell=self.config.dec_rnn_cell,
                                            input_size=self.config.embed_size,
                                            hidden_size=self.config.dec_cell_size,
                                            num_layers=self.config.num_layers,
                                            output_dropout_p=self.config.dropout,
                                            bidirectional=False,
                                            vocab_size=self.vocab_size,
                                            ctx_cell_size=self.config.dec_cell_size,
                                            sys_id=self.bos_id,
                                            eos_id=self.eos_id,
                                            use_gpu=self.config.use_gpu,
                                            max_dec_len=self.config.max_dec_len,
                                            embedding=self.embedding)

        # aux
        self.gumbel_connector = nn_lib.GumbelConnector(self.config.use_gpu)
        self.relu = nn.ReLU()
        self.nll = NLLEntropy(self.pad_id, self.config.avg_type)
        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)


    def valid_loss(self, losses, batch_cnt=None):
        total_loss = 0
        for key, loss in losses.items():
            if key == 'pi_kl':
                total_loss += self.beta * loss
            else:
                total_loss += loss
        return total_loss


    def forward(self, data_feed, mode, gen_type='greedy', use_py=None, return_latent=False, epoch=np.inf):
        # user_utts, sys_utts
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        batch_size = len(ctx_lens)
        user_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        sys_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        # <s> := 4, pad := 0, <eos> := 6
        # print ('This is the start of sys: ', sys_utts[:, 0])
        # print ('This is the end of sys: ', sys_utts[:, -1])
        # print ('This is the start id: ', self.bos_id)
        # print ('This is the end id: ', self.eos_id)
        # print ('This is the pad id: ', self.pad_id)

        ### bs, db  
        bs = self.np2var(data_feed['bs'],FLOAT)
        db = self.np2var(data_feed['db'],FLOAT)

        # prior ~ N(0,0.1)
        prior_mu = self.np2var(0.0*np.ones((batch_size,1)), FLOAT)
        prior_logvar = self.np2var(-2.3*np.ones((batch_size,1)), FLOAT)
        
        # encode for x
        user_utt_summary, _, _ = self.utt_encoder(user_utts.unsqueeze(1))
        # embed bs and db
        x_enc = th.cat([bs, db, user_utt_summary.squeeze(1)], dim=1)
        # compute the variational posterior
        x_q_mu, x_q_logvar = self.c2z(x_enc) # q(z|x,db,ds)
        if mode == GEN:
            x_z = x_q_mu
        else:
            x_z = self.gauss_connector(x_q_mu, x_q_logvar)

        # create decoder dict
        decoder_settings = {}

        ### transform from x to y
        dec_init_state = self.relu(self.z_embedding_x2y(th.cat([x_z, x_enc], dim=1))) # concat z and ctx
        dec_inputs = sys_utts[:, :-1]
        labels = sys_utts[:, 1:].contiguous()
        q_mu, q_logvar, z = x_q_mu, x_q_logvar, x_z
        decoder_settings["x2y"] = [dec_init_state, dec_inputs, labels, q_mu, q_logvar, z]

        # decoder
        result = {}
        for name, item in decoder_settings.items():
            # unpack
            dec_init_state, dec_inputs, labels, q_mu, q_logvar, z = item
            dec_init_state = dec_init_state.unsqueeze(0)

            # construct lstm init state
            if self.config.dec_rnn_cell == 'lstm':
                dec_init_state_dec = tuple([dec_init_state, dec_init_state])
            else:
                dec_init_state_dec = dec_init_state

            if name == 'x2y':
                dec_outputs, dec_hidden_state, ret_dict = self.decoder_x2y(batch_size=batch_size,
                                                                        dec_inputs=dec_inputs,
                                                                        dec_init_state=dec_init_state_dec,  # tuple: (h, c)
                                                                        attn_context=None,
                                                                        mode=mode,
                                                                        gen_type=gen_type,
                                                                        beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
                if mode == GEN: # only return x->y when GEN mode
                    ret_dict['sample_z'] = z
                    return ret_dict, labels
                # option to learn disc
                if self.config.disc: 
                    dec_init_state_disc = th.zeros_like(dec_init_state)
                    if self.config.dec_rnn_cell == 'lstm':
                        dec_init_state_disc = tuple([dec_init_state_disc, dec_init_state_disc])
                    else:
                        dec_init_state_disc = dec_init_state_disc
                    # randomly generate fake samples
                    _, _, ret_dict_gen = self.decoder_x2y(batch_size=batch_size,
                                                        dec_inputs=dec_inputs,
                                                        dec_init_state=dec_init_state_dec,  # tuple: (h, c)
                                                        attn_context=None,
                                                        mode='gen',
                                                        gen_type="greedy",
                                                        beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
                    disc_gen_labels = self.gumbel_connector(ret_dict_gen['logits'], hard=False)
                    disc_gen_ind = disc_gen_labels.argmax(dim=-1).detach() # var with no grad
                    # print ('This is the disc_gen_ind: ', disc_gen_ind)
                    bos = self.np2var(self.bos_id*np.ones((batch_size,1)), LONG)
                    disc_gen_inputs = th.cat([bos, disc_gen_ind[:, :-1]], dim=-1)
                    # discriminate fake samples
                    disc_log_prob_fake, disc_logits_fake = self.discriminator(batch_size=batch_size, 
                                                                            dec_inputs=disc_gen_inputs, 
                                                                            dec_init_state=dec_init_state_disc)
                    # discriminate real samples
                    disc_real_inputs = dec_inputs
                    disc_log_prob_real, disc_logits_real = self.discriminator(batch_size=batch_size, 
                                                                            dec_inputs=disc_real_inputs, 
                                                                            dec_init_state=dec_init_state_disc)
                    # construct loss for learning disc from real samples
                    disc_loss = self.nll(disc_log_prob_real, labels)
                    # construct loss for learning disc by synthetic samples from gen
                    if self.config.gen_guide and epoch >= self.config.warmup:
                        disc_loss += - self.config.eta * th.mean(th.sum(disc_log_prob_fake*disc_gen_labels.detach(), dim=-1))
                    result['disc_loss'] = disc_loss
                # use a kl-divergence as a reg
                if self.config.reg in ['kl']:
                    kl_loss = self.gauss_kl(q_mu, q_logvar, prior_mu, prior_logvar)
                    result['pi_kl'] = kl_loss
                # loss for cond likelihood
                result['nll_%s'%name] = self.nll(dec_outputs, labels) 
        return result


    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob


    def forward_rl(self, data_feed, max_words, temp=0.1, args=None):
        # user_utts, sys_utts
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        batch_size = len(ctx_lens)
        user_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        sys_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        labels = sys_utts[:, 1:].contiguous()

        ### user_bs, sys_utt, user_db, sys_db 
        bs = self.np2var(data_feed['bs'],FLOAT)
        db = self.np2var(data_feed['db'],FLOAT)

        # encode for x
        user_utt_summary, _, _ = self.utt_encoder(user_utts.unsqueeze(1))
        # embed bs and db
        x_enc = th.cat([bs, db, user_utt_summary.squeeze(1)], dim=1)
        # compute the variational posterior
        x_q_mu, x_q_logvar = self.c2z(x_enc) # q(z|x,db,ds)
        
        if args.kl:
            self.q_mu = x_q_mu
            self.q_logvar = x_q_logvar

        ### 1 Transfer from x to y
        if args.end2end:
            x_sample_z = self.gauss_connector(x_q_mu, x_q_logvar)
        else:
            x_sample_z = th.normal(x_q_mu, th.sqrt(th.exp(x_q_logvar))).detach()
        if args.simple_controller:
            logprob_x_sample_z = self.gaussian_logprob(x_q_mu, self.zero, x_sample_z)
        else:
            logprob_x_sample_z = self.gaussian_logprob(x_q_mu, x_q_logvar, x_sample_z)
        joint_logpz = th.sum(logprob_x_sample_z, dim=1)
        
        # decode
        dec_init_state = self.relu(self.z_embedding_x2y(th.cat([x_sample_z, x_enc], dim=1))).unsqueeze(0)
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state_xy = tuple([dec_init_state, dec_init_state])
        else:
            dec_init_state_xy = dec_init_state

        logprobs, outs, log_softmax = self.decoder_x2y.forward_rl(batch_size=batch_size,
                                                                dec_init_state=dec_init_state_xy,
                                                                attn_context=None,
                                                                vocab=self.vocab,
                                                                max_words=max_words,
                                                                temp=temp)

        # discriminator to augment reward
        if self.config.disc:
            dec_init_state_disc = th.zeros_like(dec_init_state[:, 0:1, :])
            if self.config.dec_rnn_cell == 'lstm':
                dec_init_state_disc = tuple([dec_init_state_disc, dec_init_state_disc])
            else:
                dec_init_state_disc = dec_init_state_disc
            disc_gen_inputs = []
            log_softmax_gen = []
            bos = self.np2var(self.bos_id*np.ones(1), LONG)
            if batch_size == 1:
                log_softmax = [log_softmax]
            for b_log in log_softmax:
                b_log = th.stack(b_log, dim=0)
                # print ('This is the size of b_log argmax: ', b_log.argmax(dim=-1).size())
                disc_gen_inputs.append(th.cat([bos, b_log.argmax(dim=-1)[:-1]], dim=0))
                log_softmax_gen.append(b_log)
                # log_softmax_gen.append(b_log.argmax(dim=-1, keepdims=True))
            disc_log_prob_fake, disc_logits_fake = self.discriminator(batch_size=batch_size, 
                                                                    dec_inputs=disc_gen_inputs, 
                                                                    dec_init_state=dec_init_state_disc)
            disc_logprob = []
            disc_logprob_np = []
            for i in range(len(log_softmax_gen)):
                b_logs = th.sum(th.exp(log_softmax_gen[i]).detach()*disc_log_prob_fake[i].detach(), dim=-1)
                # print ('This is the fake disc: ', b_logs)
                # b_logs = disc_log_prob_fake[i].detach().gather(-1, log_softmax_gen[i].detach())
                # b_logs = th.sum(th.exp(log_softmax_gen[i]).detach()*disc_log_prob_fake[i].detach(), dim=-1) - th.sum(th.exp(log_softmax_gen[i]).detach() * log_softmax_gen[i].detach(), dim=-1)
                # b_logs = th.clamp(b_logs, min=-0.5, max=0.0)
                disc_logprob.append(list(b_logs.unbind(dim=0)))
                disc_logprob_np.append(b_logs.cpu().numpy().tolist())
        else:
            disc_logprob = []
            disc_logprob_np = []
        return logprobs, outs, joint_logpz, x_sample_z, disc_logprob, disc_logprob_np
