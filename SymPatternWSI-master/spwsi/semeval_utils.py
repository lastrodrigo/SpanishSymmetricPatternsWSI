import spacy
import os
from xml.etree import ElementTree
from typing import Dict
import tempfile
import subprocess
import logging
from jnius import autoclass #RL

#+RL
import random
from enum import Enum 
import subprocess

class Task(Enum):
    SENSEVAL_2_SLS = 'SE2SLS'
    SEMEVAL_2015_T13 = 'SE2015T13'
    SEMEVAL_2013_T13 = 'SE2013T13'
#-

def generate_sem_eval_2013(dir_path: str):
    logging.info('reading SemEval dataset from %s' % dir_path)
    nlp = spacy.load("en", disable=['ner', 'parser'])
    in_xml_path = os.path.join(dir_path, 'contexts/senseval2-format/semeval-2013-task-13-test-data.senseval2.xml')
    gold_key_path = os.path.join(dir_path, 'keys/gold/all.key')
    with open(in_xml_path, encoding="utf-8") as fin_xml, open(gold_key_path, encoding="utf-8") as fin_key:
        instid_in_key = set()
        for line in fin_key:
            lemma_pos, inst_id, _ = line.strip().split(maxsplit=2)
            instid_in_key.add(inst_id)
        et_xml = ElementTree.parse(fin_xml)
        for word in et_xml.getroot():
            for inst in word.getchildren():
                inst_id = inst.attrib['id']
                if inst_id not in instid_in_key:
                    # discard unlabeled instances
                    continue
                context = inst.find("context")
                before, target, after = list(context.itertext())
                before = [x.text for x in nlp(before.strip(), disable=['parser', 'tagger', 'ner'])]
                target = target.strip()
                after = [x.text for x in nlp(after.strip(), disable=['parser', 'tagger', 'ner'])]
                yield before + [target] + after, len(before), inst_id

def replace_acuted(word: str): #+RL
    return word.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')

def generate_senseval_2(dir_path: str): #+RL
    logging.info('reading Senseval dataset from %s' % dir_path)
    nlp = spacy.load("es", disable=['ner','parser'])
    in_xml_path = os.path.join(dir_path,'test/test.xml')
    gold_key_path = os.path.join(dir_path,'key')
    dict_path = os.path.join(dir_path,'test/senseval.dict')
    with open(in_xml_path,encoding='ISO-8859-1') as fin_xml, open(gold_key_path, encoding="utf8") as fin_key:
        instid_in_key = set()
        lemmas = dict()
        for line in fin_key:
            lemma_pos, inst_id, _ = line.strip().split(maxsplit=2)
            if not (lemma_pos in lemmas.keys()):
                with open(dict_path,encoding='ISO-8859-1') as fin_dict:
                    for line in fin_dict:
                        dict_entry = replace_acuted(line.split('#')[0]) 
                        if dict_entry not in ['','\n']:
                            if dict_entry[-1] == 's':
                                dict_entry = dict_entry[:-1]
                            if  dict_entry == lemma_pos:
                                pos = line.split('#')[1][0].lower()
                                if pos == 'a':
                                    pos= 'j'
                                lemmas[lemma_pos] = pos
                                break
            instid_in_key.add(inst_id)
        et_xml = ElementTree.parse(fin_xml)
        for word in et_xml.getroot():
            for inst in word.getchildren():
                inst_id = inst.attrib['id']
                if inst_id not in instid_in_key:
                    continue
                lemma_pos = inst_id.split('.')[0]
                lemma_pos += '.' + lemmas[lemma_pos] 
                context = inst.find("context")
                before, target, after = list(context.itertext())
                before = [x.text for x in nlp(before.encode(encoding='ISO-8859-1').decode(encoding='UTF-8').strip(),disable=['parser','tagger','ner'])]
                target = target.encode(encoding='ISO-8859-1').decode(encoding='UTF-8').strip()
                after = [x.text for x in nlp(after.encode(encoding='ISO-8859-1').decode(encoding='UTF-8').strip(), disable=['parser','tagger','ner'])]
                yield before + [target] + after, len(before), inst_id, lemma_pos

def generate_sem_eval_2015(dir_path: str): #+RL
    logging.info('reading SemEval 2015 T13 dataset from %s' % dir_path)
    nlp = spacy.load("es", disable=['ner','parser'])
    in_xml_path = os.path.join(dir_path,'data/semeval-2015-task-13-es.xml')
    gold_key_path = os.path.join(dir_path,'keys/gold_keys/ES/semeval-2015-task-13-es-WSD.key')
    with open(in_xml_path, encoding="utf8") as fin_xml, open(gold_key_path, encoding="utf8") as fin_key:
        instid_in_key = set()
        for line in fin_key:
            #lemma_pos =
            if line.strip().split(maxsplit= 2)[0] == line.strip().split(maxsplit=2)[1]:
                inst_id = line.strip().split(maxsplit= 2)[0]
                instid_in_key.add(inst_id)
        et_xml = ElementTree.parse(fin_xml)
        for text in et_xml.getroot():
            for sentence in text:
                for wf in sentence:
                    inst_id = wf.attrib["id"]
                    if not inst_id in instid_in_key:
                        continue
                    lemma = wf.attrib["lemma"]
                    pos = wf.attrib["pos"].lower()
                    lemma_pos = lemma + '.' + pos
                    before = str()
                    afterTarget = False
                    target = str()
                    after = str()
                    for wfIter in sentence:
                        if inst_id != wfIter.attrib["id"]:
                            if not afterTarget:
                                before += ' ' + wfIter.text
                            else:
                                after += ' ' + wfIter.text
                        elif inst_id == wfIter.attrib["id"]:
                            target = wfIter.text.strip()
                            afterTarget = True
                    before = [x.text for x in nlp(before.strip(), disable=['parser', 'tagger', 'ner'])]
                    after = [x.text for x in nlp(after.strip(), disable=['parser','tagger','ner'])]
                    yield before + [target] + after, len(before), inst_id,lemma_pos

def evaluate_labeling(dir_path, labeling: Dict[str, Dict[str, int]], key_path: str = None, task=Task.SENSEVAL_2_SLS,maxLabels= 2) \
        -> Dict[str, Dict[str, float]]: #RL task added
    """
    labeling example : {'become.v.3': {'become.sense.1':3,'become.sense.5':17} ... }
    means instance become.v.3' is 17/20 in sense 'become.sense.5' and 3/20 in sense 'become.sense.1'
    :param key_path: write produced key to this file
    :param dir_path: SemEval dir
    :param labeling: instance id labeling
    :return: FNMI, FBC as calculated by SemEval provided code
    """
    logging.info('starting evaluation key_path: %s' % key_path)
    
    def get_scores(gold_key, eval_key,task): #RL task added
        
        ret = {}
        if task is Task.SEMEVAL_2013_T13: #RL
            for metric, jar, column in [
                #         ('jaccard-index','SemEval-2013-Task-13-test-data/scoring/jaccard-index.jar'),
                #         ('pos-tau', 'SemEval-2013-Task-13-test-data/scoring/positional-tau.jar'),
                #         ('WNDC', 'SemEval-2013-Task-13-test-data/scoring/weighted-ndcg.jar'),
                ('FNMI', os.path.join(dir_path, 'scoring/fuzzy-nmi.jar'), 1),
                ('FBC', os.path.join(dir_path, 'scoring/fuzzy-bcubed.jar'), 3),
            ]:
                logging.info('calculating metric %s' % metric)
                res = subprocess.Popen(['java', '-jar', jar, gold_key, eval_key], stdout=subprocess.PIPE).stdout.readlines()
                # columns = []
                for line in res:
                    line = line.decode().strip()
                    if line.startswith('term'):
                        # columns = line.split('\t')
                        pass
                    else:
                        split = line.split('\t')
                        if len(split) > column:
                            word = split[0]
                            # results = list(zip(columns[1:], map(float, split[1:])))
                            result = split[column]
                            if word not in ret:
                                ret[word] = {}
                            ret[word][metric] = float(result)

        #+RL
        elif task is Task.SENSEVAL_2_SLS:
            script = ["python2.7","./spanish-lex-sample/score/score",eval_key, gold_key,'./spanish-lex-sample/test/emptysensemap']
            res = subprocess.Popen(" ".join(script),shell=True, env={"PYTHONPATH":"."},stdout=subprocess.PIPE).stdout.readlines()
            
            ret['all']={}
            splitted = res[2].strip().split()
            ret['all']['precision'] = float(splitted[1])
            ret['all']['correct'] = float(str(splitted[2].decode()).replace('(',''))
            ret['all']['attempted'] = float(splitted[5])
            splitted = res[3].strip().split()
            ret['all']['recall'] = float(splitted[1])
            ret['all']['total'] = float(splitted[5])
            ret['all']['attemptedPct'] = float(splitted[1])
        elif task is Task.SEMEVAL_2015_T13:
            pass
        #-
        return ret
            

    def getGoldKeySENSEVAL2(goldPath): #+RL
        with open(os.path.join(dir_path,goldPath),'r') as fgold:
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

    def dictToJ(dictionary): #+RL
        HashMap = autoclass('java.util.HashMap')
        String = autoclass('java.lang.String')
        Double = autoclass('java.lang.Double')
        map = HashMap()
        for token, instances in dictionary.items():
            jToken = String(token)
            instanceMap = HashMap()
            for instance, labels in instances.items():
                jInstance = String(instance)
                labelMap = HashMap()
                for label, applicability in labels.items():
                    jLabel = String(label)
                    jApplicability = Double(applicability)
                    labelMap.put(jLabel,jApplicability)
                instanceMap.put(jInstance,labelMap)
            map.put(jToken,instanceMap)
        return map

    def getTrainingInstances(trainingSets): #+RL

        HashSet = autoclass('java.util.HashSet')
        String = autoclass('java.lang.String')
        listJTrainingSets = []
        for trainingSet in trainingSets:
            jTrainingSet = HashSet()
            for instance in trainingSet:
                jInstance = String(instance)
                jTrainingSet.add(jInstance)
            listJTrainingSets.append(jTrainingSet)
        return listJTrainingSets

    def printTrainingSets(listJTrainingSets): #+RL
        trainingSet = 1
        
        for trainingInstances in listJTrainingSets:
            print('---------------------------------------------Training set %d \n' % trainingSet)
            entrySetIterator = trainingInstances.iterator()
            while entrySetIterator.hasNext():
                e = entrySetIterator.next()
                print(e+', ')
            trainingSet += 1

    def mapSenses(trainingInstances,goldMap,labelingMap): #+RL
        GradedReweightedKeyMapper = autoclass('edu.ucla.clustercomparison.GradedReweightedKeyMapper')
        mapper = GradedReweightedKeyMapper()
        allRemappedTestKey = {}
        remappedTestKey = mapper.convert(goldMap,labelingMap,trainingInstances)
        #print(remappedTestKey)
        convertedSet = remappedTestKey.entrySet()
        convertedIterator = convertedSet.iterator()
        while convertedIterator.hasNext():
            e = convertedIterator.next()
            doc = e.getKey()
            instRatings = e.getValue()
            instanceIterator = instRatings.entrySet().iterator()
            while instanceIterator.hasNext():
                i = instanceIterator.next()
                instance = i.getKey()
                labelIterator = i.getValue().entrySet().iterator()
                labelList = []
                while labelIterator.hasNext():
                    l = labelIterator.next()
                    label = l.getKey()
                    applicability = l.getValue()
                    labelList.append((label,applicability))
                labelList.sort(key=lambda x:x[1],reverse=True)
                allRemappedTestKey[instance] = labelList
        return allRemappedTestKey

    with tempfile.NamedTemporaryFile('wt') as fout:
        lines = []
        #+RL
        print(labeling)
        if task is Task.SENSEVAL_2_SLS:
            goldPath = 'key'
            #SENSEVAL 2
            goldKey = getGoldKeySENSEVAL2(goldPath)
            allInstances = []
            for _,v in goldKey.items():
                for k1,_ in v.items():
                    allInstances.append(k1)
            indices = list(range(0,len(allInstances)-1))
            random.seed(18)
            random.shuffle(indices)
            trainingSets = [set() for _ in range(0,5)]
            for i in range(0,len(allInstances)):
                instance = allInstances[i]
                toExclude = i % len(trainingSets)
                for j in range(0,len(trainingSets)):
                    if j != toExclude:
                        trainingSets[j].add(instance)
            termToNumberSenses = {}
            for e in goldKey.items():
                term = e[0]
                
                senses = set()
                for ratings in goldKey[term].values():
                    for sense in ratings.keys():
                        senses.update(sense)
                
                termToNumberSenses[term] = len(senses)
            
            
            listJTrainingInstances = getTrainingInstances(trainingSets)
            goldMap = dictToJ(goldKey)
            lemmaLabeling = {}
            for k,v in labeling.items():
                lemma = k.split('.')[0]
                if not lemma in lemmaLabeling:
                    lemmaLabeling[lemma] = {k:v}
                else:
                    lemmaLabeling[lemma][k] = v
            labelingMap = dictToJ(lemmaLabeling)
            #print(trainingSets)
            num = 1
            print(maxLabels)
            for jTrainingInstances in listJTrainingInstances:
                testKey = mapSenses(jTrainingInstances,goldMap,labelingMap)
                lines = []
                for instance, label in testKey.items():
                    
                    clusters_str = ' '.join(x[0].split('.')[1] for x in label[0:maxLabels])
                    

                    lines.append('%s %s %s' % (instance.split('.')[0],instance,clusters_str))
                #print(lines)
                evalKey = key_path+str(num)
                logging.info('writing key to file %s' % evalKey)
                
                with open(evalKey, 'w', encoding="utf-8") as fout2:
                    fout2.write('\n'.join(lines))
                scores = get_scores(os.path.join(dir_path, goldPath), #'keys/gold/all.key'), RL goldPath added
                             evalKey,task) #RL  task added
                num += 1
                print(scores)
        elif task is Task.SEMEVAL_2013_T13:
        #-
            goldPath = 'keys/gold/all.key'
            for instance_id, clusters_dict in labeling.items():
                clusters = sorted(clusters_dict.items(), key=lambda x: x[1])
                clusters_str = ' '.join([('%s/%d' % (cluster_name, count)) for cluster_name, count in clusters])
                lemma_pos = instance_id.rsplit('.', 1)[0]
                lines.append('%s %s %s' % (lemma_pos, instance_id, clusters_str))
            fout.write('\n'.join(lines))
            fout.flush()
        
            scores = get_scores(os.path.join(dir_path, goldPath), #'keys/gold/all.key'), RL goldPath added
                            fout.name,task) #RL  task added

            if key_path: 
                logging.info('writing key to file %s' % key_path)
                with open(key_path, 'w', encoding="utf-8") as fout2:
                    fout2.write('\n'.join(lines))

        
        return scores

