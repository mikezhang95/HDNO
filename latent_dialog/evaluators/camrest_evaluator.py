
import json

from latent_dialog.evaluators import BaseEvaluator, BLEUScorer
from latent_dialog.utils import DATA_DIR


class CamRestEvaluator(BaseEvaluator):
    def __init__(self, data_name):
        super().__init__()
        delex_json = open(DATA_DIR + data_name +  '/delex.json')
        self.delex_data = json.load(delex_json)

    def match_metric(self, dials):
        # we feed the real beleif state, so we only check if sys tells the name
        match,total = 0, 0
        real_dials = self.delex_data
        for dial_id, dial in dials.items():
            real_match, gen_match = 0, 0 
            real_dial = real_dials[dial_id]["sys"]
            for turn_num, turn in enumerate(dial):
                gen_response_token = turn.split()
                response_token = real_dial[turn_num].split()
                for idx, w in enumerate(gen_response_token):
                    if w == "name_slot":
                        gen_match = 1
                for idx, w in enumerate(response_token):
                    if w == "name_slot":
                        real_match = 1
            if real_match==gen_match:
                match += 1
            total += 1

        return match / total


    def success_f1_metric(self, dials):
        tp,fp,fn = 0,0,0
        real_dials = self.delex_data
        for dial_id, dial in dials.items():
            truth_req, gen_req = set(),set()
            real_dial = real_dials[dial_id]["sys"]
            for turn_num, turn in enumerate(dial):
                gen_response_token = turn.split()
                response_token = real_dial[turn_num].split()
                for idx, w in enumerate(gen_response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        gen_req.add(w.split('_')[0])
                for idx, w in enumerate(response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        truth_req.add(w.split('_')[0])

            truth_req.discard('name')
            gen_req.discard('name')

            # MARK
            # use goal to evaluate
            truth_req = set(real_dials[dial_id]["goal"]["request-slots"])
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1

        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        # MRAK
        # return f1
        return recall


    def evaluateModel(self, dialogues, real_dialogues=False, mode="valid"):
        """Gathers statistics for the whole sets."""

        # Success F1 score
        success_f1 = self.success_f1_metric(dialogues)

        # Match
        match = self.match_metric(dialogues)

        # BLEU
        if real_dialogues:
            bscorer = BLEUScorer()
            corpus, real_corpus = [], []
            for dial_id in dialogues:
                dial = dialogues[dial_id]
                real_dial = real_dialogues[dial_id]
                turns, real_turns = [], []
                for turn in dial:
                    turns.append([turn])
                for turn in real_dial:
                    real_turns.append([turn])
                if len(turns) == len(real_turns):
                    corpus.extend(turns)
                    real_corpus.extend(real_turns)
                else:
                    raise('Wrong amount of turns')
            bleu_score = bscorer.score(corpus, real_corpus)
        else:
            bleu_score = 0.0


        total = len(list(dialogues.keys()))
        report = ""
        report += '{} Corpus Matches : {:2.2f}%'.format(mode, (match * 100)) + "\n"
        report += '{} Corpus Success F1 : {:2.2f}%'.format(mode, (success_f1 * 100)) + "\n"
        report += '{} Corpus BLEU: {:2.2f}%'.format(mode, bleu_score* 100) + "\n"
        report += 'Total number of dialogues: %s ' % total
        
        return report, success_f1, match, bleu_score


