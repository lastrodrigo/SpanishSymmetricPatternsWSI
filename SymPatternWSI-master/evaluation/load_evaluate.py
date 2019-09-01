import os
#from SymPatternWSI-master.evaluation import Evaluator

def getGoldKeySENSEVAL2(goldPath): #+RL
    lemmas = {}
    
    with open(os.path.join(goldPath),'r') as fgold:
        for line in fgold.readlines():
            instances = {}
            splitted = line.strip().split()
            graded = dict()
            rest = splitted[2:]
            for index in rest:
                graded[splitted[0]+'.'+index] = 1.0 / len(rest)
            instances[splitted[1]] = graded 
            if not splitted[0] in lemmas:
                lemmas[splitted[0]] = instances
            else:
                lemmas[splitted[0]].update(instances)
    return lemmas

def getAnswersKeySENSEVAL2(answersPath): #+RL
    instances = {}
    with open(os.path.join(answersPath),'r') as f:
        for line in f.readlines():
            
            splitted = line.strip().split()
            graded = []
            rest = splitted[2:]
            for index in rest:
                graded.append((splitted[0]+'.'+index, 1.0 / len(rest)))
            instances[splitted[1]] = graded 
    return instances

#gold_key = getGoldKeySENSEVAL2('SymPatternWSI-master/spanish-lex-score/key')
#answer_key = getKeySENSEVAL2('embeddingBaseline')

#ev = Evaluator({},{})
#ev.