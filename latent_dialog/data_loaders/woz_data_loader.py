"""
    This is used to load multiwoz type data, include: <utterance, db, bs, response>
"""

import json
import logging
import numpy as np
from latent_dialog.utils import Pack
from latent_dialog.data_loaders import BaseDataLoader
from latent_dialog.data_loaders.woz_corpora import USR, SYS

logger = logging.getLogger()


class MultiWozDataLoader(BaseDataLoader):

    def __init__(self, mode, data, config):
        super(MultiWozDataLoader, self).__init__(mode)
        self.mode = mode  
        self.raw_data = data
        self.config = config
        self.max_utt_len = config.max_utt_len
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data)
        self.data_size = len(self.data)
        self.domains = ['hotel', 'restaurant', 'train', 'attraction', 'hospital', 'police', 'taxi']

    def flatten_dialog(self, data):
        """
            Reorganize data in <conext, response> pairs.
            Args:
                - data: 
        """
        results = []
        indexes = []
        batch_indexes = []
        resp_set = set()
        for dlg in data:
            goal = dlg.goal
            key = dlg.key
            batch_index = []
            for i in range(1, len(dlg.dlg)):
                if dlg.dlg[i].speaker == USR:
                    continue
                e_idx = i
                s_idx = max(0, e_idx - 1)
                response = dlg.dlg[i].copy()
                response['utt'] = self.pad_to(self.max_utt_len, response.utt, do_pad=False)
                resp_set.add(json.dumps(response.utt))
                context = []
                for turn in dlg.dlg[s_idx: e_idx]:
                    turn['utt'] = self.pad_to(self.max_utt_len, turn.utt, do_pad=False)
                    context.append(turn)
                results.append(Pack(context=context, response=response, goal=goal, key=key))
                indexes.append(len(indexes))
                batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)

        logger.info("Unique/Total Response {}/{} - {}".format(len(resp_set), len(results), self.mode))
        # indexes: [0,1,2,...]   batch_indexes: [[0,1,2,3,4],[5,6,7,8,9],...]
        return results, indexes, batch_indexes

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        """
            Args:
                - fix_batch: if True, one session becomes a batch

        """
        self.ptr = 0
        if fix_batch:
            self.batch_size = None
            self.num_batch = len(self.batch_indexes)
            if shuffle:
                self._shuffle_batch_indexes()
        else:
            if shuffle:
                self._shuffle_indexes()
            self.batch_size = config.batch_size
            self.num_batch = self.data_size // config.batch_size
            self.batch_indexes = []
            for i in range(self.num_batch):
                self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
            if verbose:
                print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
        # if shuffle:
        #     if fix_batch:
        #         self._shuffle_batch_indexes()
        #     else:
        #         self._shuffle_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        ctx_utts, ctx_lens = [], []
        out_utts, out_lens = [], []

        out_bs, out_db = [] , []
        goals, goal_lens = [], [[] for _ in range(len(self.domains))]
        keys = []

        # bs_{t-1}, bs_{t+1}, db_{t-1}, db_{t+1}
        out_bs_prev, out_bs_next, out_db_prev, out_db_next = [] , [], [], []

        for j, row in enumerate(rows):
            in_row, out_row, goal_row = row.context, row.response, row.goal

            # source context
            keys.append(row.key)
            batch_ctx = []
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_len, turn.utt, do_pad=True))
            ctx_utts.append(batch_ctx)
            ctx_lens.append(len(batch_ctx))

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            out_bs.append(out_row.bs)
            out_db.append(out_row.db)

            # :
            if j==0:
                out_bs_prev.append(np.zeros_like(out_bs[0]))
                out_db_prev.append(np.zeros_like(out_db[0]))
            else:
                out_bs_prev.append(out_bs[-2])
                out_db_prev.append(out_db[-2])
            if j==0:
                pass
            else:
                out_bs_next.append(out_bs[-1])
                out_db_next.append(out_db[-1])


            # goal
            goals.append(goal_row)
            for i, d in enumerate(self.domains):
                goal_lens[i].append(len(goal_row[d]))

        # :
        out_bs_next.append(np.zeros_like(out_bs[0]))
        out_db_next.append(np.zeros_like(out_db[0]))
        vec_out_bs_next = np.array(out_bs_next) # (batch_size, 94)
        vec_out_db_next = np.array(out_db_next) # (batch_size, 30)
        vec_out_bs_prev = np.array(out_bs_prev) # (batch_size, 94)
        vec_out_db_prev = np.array(out_db_prev) # (batch_size, 30)


        batch_size = len(ctx_lens)
        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((batch_size, max_ctx_len, self.max_utt_len), dtype=np.int32)
        vec_out_bs = np.array(out_bs) # (batch_size, 94)
        vec_out_db = np.array(out_db) # (batch_size, 30)
        vec_out_lens = np.array(out_lens)  # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((batch_size, max_out_len), dtype=np.int32)

        max_goal_lens, min_goal_lens = [max(ls) for ls in goal_lens], [min(ls) for ls in goal_lens]
        if max_goal_lens != min_goal_lens:
            print('Fatal Error!')
            exit(-1)
        self.goal_lens = max_goal_lens
        vec_goals_list = [np.zeros((batch_size, l), dtype=np.float32) for l in self.goal_lens]

        for b_id in range(batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]
            for i, d in enumerate(self.domains):
                vec_goals_list[i][b_id, :] = goals[b_id][d]

        return Pack(context_lens=vec_ctx_lens, # (batch_size, )
                    contexts=vec_ctx_utts, # (batch_size, max_ctx_len, max_utt_len)
                    output_lens=vec_out_lens, # (batch_size, )
                    outputs=vec_out_utts, # (batch_size, max_out_len)
                    bs=vec_out_bs, # (batch_size, 94)
                    db=vec_out_db, # (batch_size, 30)
                    goals_list=vec_goals_list, # 7*(batch_size, bow_len), bow_len differs w.r.t. domain
                    keys=keys,
                    bs_next=vec_out_bs_next,
                    bs_prev=vec_out_bs_prev,
                    db_next=vec_out_db_next,
                    db_prev=vec_out_db_prev
                    )

    def clone(self):
        return MultiWozDataLoader(self.mode, self.raw_data, self.config)
