import os
import statistics
from jnius import autoclass


class Evaluator:
    
    def _getKeySENSEVAL2(self, goldPath): #+RL
        with open(os.path.join(goldPath),'r') as fgold:
            goldKey = dict()
            for line in fgold.readlines():
                splitted = line.strip().split()
                #if splitted[0] == lemma:
                instance = dict()
                graded = dict()
                rest = splitted[2:]
                for index in rest:
                    graded[splitted[0]+'.'+index] = 1.0 / len(rest)
                instance[splitted[1]] = graded 
                if not splitted[0] in goldKey:
                    goldKey[splitted[0]] = instance
                else:
                    goldKey[splitted[0]].update(instance)
        return goldKey

    def __init__(self, gold_key, answers):

        self.answers = answers
        self.gold = {}
        for lemma, instances in gold_key.items():
            for instance, labels in instances.items():
                self.gold[instance] = labels
        self.java_answers = {}
        self.java_gold = {}
        HashMap = autoclass('java.util.HashMap')
        String = autoclass('java.lang.String')
        Double = autoclass('java.lang.Double')
        for instance, labels in self.gold.items():
            label_list = labels.items()
            if instance in self.answers:
                answers_list = self.answers[instance]
                j_gold_map = HashMap()
                for label, applicability in label_list:
                    j_label = String(label)
                    j_applicability = Double(applicability)
                    j_gold_map.put(j_label, j_applicability)
                self.java_gold[instance] = j_gold_map
                j_answer_map = HashMap()
                for answer, applicability in answers_list:
                    j_answer = String(answer)
                    j_applicability = Double(applicability)
                    j_answer_map.put(j_answer, j_applicability)
                self.java_answers[instance] = j_answer_map

    def jaccard_index(self):
        jaccard_indexes = {}
        jaccard_index_class = autoclass('scoring.JniusJaccardIndex')
        java_jaccard = jaccard_index_class()
        for instance, java_gold in self.java_gold.items():
            jaccard_indexes[instance] = java_jaccard.evaluate(java_gold, self.java_answers[instance])
        return statistics.mean(jaccard_indexes.values()), jaccard_indexes

    def positional_tau(self):
        positional_taus = {}
        positional_tau_class = autoclass('scoring.JniusPositionalTau')

        for instance, java_gold in self.java_gold.items():
            num_senses = len(self.gold[instance]) + len(self.answers[instance])
            java_positional = positional_tau_class()
            positional_taus[instance] = java_positional.evaluateInstance(java_gold, self.java_answers[instance], num_senses)
        return statistics.mean(positional_taus.values()), positional_taus

    def weighted_ndcg(self):
        weighted_ndcgs = {}
        weighted_ndcg_class = autoclass('scoring.JniusWeightedNormalizedDiscountedCumulativeGain')

        for instance, java_gold in self.java_gold.items():
            java_wndcg = weighted_ndcg_class()
            weighted_ndcgs[instance] = java_wndcg.evaluate(java_gold, self.java_answers[instance])
        return statistics.mean(weighted_ndcgs.values()), weighted_ndcgs

    def semeval_2013_task_13_metrics(self):
        metrics = {}
        metrics['jaccard_index'] = self.jaccard_index()[0]
        metrics['positional_tau'] = self.positional_tau()[0]
        metrics['weighted_ndcg'] = self.weighted_ndcg()[0]
        return metrics
