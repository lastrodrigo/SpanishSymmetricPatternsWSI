import os
import statistics
from jnius import autoclass


class Evaluator:

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

    def _get_lemma_hashmaps(self):
        lemma_gold_hashmaps = {}
        lemma_answers_hashmaps = {}
        for instance, java_gold in self.java_gold.items():
            lemma = instance.split('.')[0]
            if lemma in lemma_gold_hashmaps:
                lemma_gold_hashmaps[lemma].update({instance: java_gold})
            else:
                lemma_gold_hashmaps[lemma] = {instance: java_gold}
        for instance, java_answer in self.java_answers.items():
            lemma = instance.split('.')[0]
            if lemma in lemma_answers_hashmaps:
                lemma_answers_hashmaps[lemma].update({instance: java_answer})
            else:
                lemma_answers_hashmaps[lemma] = {instance: java_answer}
        return lemma_gold_hashmaps, lemma_answers_hashmaps

    def fuzzy_bcubed(self):
        fuzzy_bcubed_class = autoclass('edu.ucla.clustercomparison.FuzzyBCubed')
        java_fuzzy_bcubed = fuzzy_bcubed_class()
        HashMap = autoclass('java.util.HashMap')
        String = autoclass('java.lang.String')
        answer_hashmap = HashMap()
        lemma_gold_hashmaps, lemma_answers_hashmaps = self._get_lemma_hashmaps()
        results = {}
        precisions = []
        recalls = []
        f1s = []
        for lemma, inst_dict in lemma_gold_hashmaps.items():
            gold_hashmap = HashMap()
            answer_hashmap = HashMap()
            for instance, label_dict in inst_dict.items():
                java_instance = String(instance)
                gold_hashmap.put(java_instance, label_dict)
            for instance, label_dict in lemma_answers_hashmaps[lemma].items():
                java_instance = String(instance)
                answer_hashmap.put(java_instance, label_dict)
            result = java_fuzzy_bcubed.computeBCubed(gold_hashmap, answer_hashmap)
            precisions.append(result[0])
            recalls.append(result[1])
            f1 = 2 * result[0] * result[1] / (result[0] + result[1]) if result[0] + result[1] > 0 else 0
            result.append(f1)
            f1s.append(f1)
            results.update({lemma: result})
        return [statistics.mean(precisions), statistics.mean(recalls), statistics.mean(f1s)], results

    def fuzzy_nmi(self):
        fuzzy_nmi_class = autoclass('edu.ucla.clustercomparison.FuzzyNormalizedMutualInformation')
        java_fuzzy_nmi = fuzzy_nmi_class()
        HashMap = autoclass('java.util.HashMap')
        String = autoclass('java.lang.String')
        lemma_gold_hashmaps, lemma_answers_hashmaps = self._get_lemma_hashmaps()
        results = {}
        for lemma, inst_dict in lemma_gold_hashmaps.items():
            gold_hashmap = HashMap()
            answer_hashmap = HashMap()
            for instance, label_dict in inst_dict.items():
                java_instance = String(instance)
                gold_hashmap.put(java_instance, label_dict)
            for instance, label_dict in lemma_answers_hashmaps[lemma].items():
                java_instance = String(instance)
                answer_hashmap.put(java_instance, label_dict)
            result = java_fuzzy_nmi.computeNmi(gold_hashmap, answer_hashmap)
            results.update({lemma: result})
        return statistics.mean(results.values()), results

    def semeval_2013_task_13_metrics(self):
        metrics = {}
        metrics['jaccard_index'] = self.jaccard_index()[0]
        metrics['positional_tau'] = self.positional_tau()[0]
        metrics['weighted_ndcg'] = self.weighted_ndcg()[0]
        bcubed = self.fuzzy_bcubed()[0]
        metrics['bcubed_precision'] = bcubed[0]
        metrics['bcubed_recall'] = bcubed[1]
        metrics['bcubed_f1'] = bcubed[2]
        metrics['fuzzy_nmi'] = self.fuzzy_nmi()[0]
        return metrics
