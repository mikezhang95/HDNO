import json
import sqlite3
import os
import random
import logging

from latent_dialog.evaluators import BaseEvaluator, BLEUScorer
from latent_dialog.utils import get_tokenize, DATA_DIR
from latent_dialog.woz_util import SYS, USR, BOS, EOS
from latent_dialog.normalizer.delexicalize import normalize

class MultiWozDB(object):
    def __init__(self, data_name):
        
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']

        # loading databases
        self.dbs = {}
        db_pattern = DATA_DIR + data_name + '/db/{}-dbase.db'
        for domain in domains:
            db = db_pattern.format(domain)
            conn = sqlite3.connect(db)
            c = conn.cursor()
            self.dbs[domain] = c

            
    def queryResultVenues(self, domain, turn, real_belief=False):

        if real_belief == True:
            # print ('This is one turn: \n', turn)
            # : extract the belief state at each turn
            # items = turn.items()
            constraints = turn
        else:
            # items = turn['metadata'][domain]['semi'].items()
            constraints = turn['metadata'][domain]['semi']

        return self.querySQL(domain, constraints)            
        
    def querySQL(self, domain, constraints):

        # query the db
        sql_query = "select * from {}".format(domain)

        flag = True
        for key, val in constraints.items():
            if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        try:  # "select * from attraction  where name = 'queens college'"
            return self.dbs[domain].execute(sql_query).fetchall()
        except:
            return []  # TODO test it


class MultiWozEvaluator(BaseEvaluator):
    def __init__(self, data_name):
        self.data_name = data_name
        delex_path = DATA_DIR + data_name + '/delex.json' 
        self.delex_dialogues = json.load(open(delex_path))
        self.db = MultiWozDB(self.data_name)

    # extract the relevant info about the target goal
    def _parseGoal(self, goal, d, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
        if 'info' in d['goal'][domain]:
        # if d['goal'][domain].has_key('info'):
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    if 'trainID' in d['goal'][domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    for s in d['goal'][domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append("reference")

            goal[domain]["informable"] = d['goal'][domain]['info']
            if 'book' in d['goal'][domain]:
            # if d['goal'][domain].has_key('book'):
                goal[domain]["booking"] = d['goal'][domain]['book']
        return goal

    def _evaluateGeneratedDialogue(self, dialog, goal, realDialogue, real_requestables, soft_acc=False):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""

        random.seed(0)

        # for computing corpus success
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, sent_t in enumerate(dialog):
            # : search through the whole sentences and
            # check whether domain_name or _id has been mentioned
            for domain in goal.keys():
                # for computing success
                if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        # : extract the groundtruth belief state
                        venues = self.db.queryResultVenues(domain, realDialogue['log'][t * 2 + 1])
                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues: # : randomly assign a new venue when it firstly meets a new domain
                            # : this is for randomly assigning a piece of data from the database
                            venue_offered[domain] = random.sample(venues, 1)
                        else: # : check whether the assigned venue still the same
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                # : go throuth the dialogue and
                # collect requestable slot from the sentence generated by the system
                for requestable in requestables:
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'train_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # print ('This is the goal in domain {}: \n {}'.format(domain, goal[domain]))
            # if name was provided for the user, the match is being done automatically
            # if realDialogue['goal'][domain].has_key('info'):
            if 'info' in realDialogue['goal'][domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in realDialogue['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # the original method
            # if domain == 'train':
                # if not venue_offered[domain]:
                    # # if realDialogue['goal'][domain].has_key('reqt') and 'id' not in realDialogue['goal'][domain]['reqt']:
                    # if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
                        # venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'


        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        # for all domains, we just check whether the domain name has been all
        # informable and the inform rate is calculated based on the domain num
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, goal[domain]['informable'], real_belief=True)
                # print ('This is the goal venues: \n', goal_venues)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1
            else:
                if domain + '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                # if the domain doesn't contain any requestables,
                # then it counts up for one success
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                # if possible requestng the extra requestables will not
                # be punished
                # and requesting the same requestable will not be punished
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        # rint requests, 'DIFF', requests_real, 'SUCC', success
        return success, match, stats

    def evaluateModel(self, dialogues, real_dialogues=False, mode='valid'):
        """Gathers statistics for the whole sets."""
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
        delex_dialogues = self.delex_dialogues
        successes, matches = 0, 0
        total = 0
        
#         # compute corpus success
        # gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     # 'taxi': [0, 0, 0],
                     # 'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        # sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                         # 'taxi': [0, 0, 0], 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        for filename, dial in dialogues.items():

            real_dialog = delex_dialogues[filename]
            goal, real_requestables = {}, {}
            for domain in domains:
                if real_dialog['goal'][domain]:
                    goal = self._parseGoal(goal, real_dialog, domain)
            for domain in goal.keys():
                real_requestables[domain] = goal[domain]["requestable"]

            success, match, stats = self._evaluateGeneratedDialogue(dial, goal, real_dialog, real_requestables, soft_acc=mode=='offline_rl')


            successes += success
            matches += match
            total += 1

#             for domain in gen_stats.keys():
                # gen_stats[domain][0] += stats[domain][0]
                # gen_stats[domain][1] += stats[domain][1]
                # gen_stats[domain][2] += stats[domain][2]

            # if 'SNG' in filename:
                # for domain in gen_stats.keys():
                    # sng_gen_stats[domain][0] += stats[domain][0]
                    # sng_gen_stats[domain][1] += stats[domain][1]
                    # sng_gen_stats[domain][2] += stats[domain][2]

        if real_dialogues: 
            # BLUE SCORE
            corpus = []
            model_corpus = []
            bscorer = BLEUScorer()

            for filename, dial in dialogues.items():
                real_dial = real_dialogues[filename]
                model_turns, corpus_turns = [], []
                for turn in real_dial:
                    corpus_turns.append([turn])
                for turn in dial:
                    model_turns.append([turn])

                if len(model_turns) == len(corpus_turns):
                    corpus.extend(corpus_turns)
                    model_corpus.extend(model_turns)
                else:
                    raise('Wrong amount of turns')

            bleu_score = bscorer.score(model_corpus, corpus)
        else:
            bleu_score = 0.


        report = ""
        report += '{} Corpus Matches : {:2.2f}%'.format(mode, (matches / float(total) * 100)) + "\n"
        report += '{} Corpus Success : {:2.2f}%'.format(mode, (successes / float(total) * 100)) + "\n"
        report += '{} Corpus BLEU: {:2.2f}%'.format(mode, bleu_score* 100) + "\n"
        report += 'Total number of dialogues: %s ' % total

        return report, successes/float(total), matches/float(total), bleu_score

