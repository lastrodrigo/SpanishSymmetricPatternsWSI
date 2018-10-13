import argparse
import os
import unicodedata as ud
import random
import math

latin_letters = {}


def is_latin(uchr):
    try:
        return latin_letters[uchr]
    except KeyError:
        return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))


def only_roman_chars(unistr):
    return all(is_latin(uchr)
               for uchr in unistr
               if uchr.isalpha())

def  word_count(word,counts):
    """Adds 1 to the occurrences of word in str in the dictionary counts"""

    if word in counts:
        counts[word] += 1
    else:
        counts[word] = 1
    return counts

def writeShuffledFile(tokens,tokensPerFile,data,directory,prefix,lineCount,vocab):
    fileCount = 0
    totalTokenCount = 0
    lines = 0
    while totalTokenCount < tokens and lineCount < len(data):
        fileCount += 1
        fileName = os.path.join(directory,prefix+'.'+str(fileCount)+'.clean.shuffled')
        tokenCount = 0
        with open(fileName,'w+',encoding='utf8') as f:
            print('Processsing output %s' % fileName)
            while tokenCount < tokensPerFile and totalTokenCount < tokens and lineCount < len(data):
                hasUnfrequentWord = False
                line = data[lineCount][1]
                for word in line.split():
                    if not word in freqWords:
                        hasUnfrequentWord = True
                if not hasUnfrequentWord:
                    f.write(line+'\n')
                    lines += 1
                    tokenCount += len(line.split())
                    for word in line.split():
                        vocab = word_count(word,vocab) 
                    totalTokenCount += len(line.split())
                lineCount += 1
    return totalTokenCount,lineCount,vocab,lines

parser = argparse.ArgumentParser(description='Clean corpus')
parser.add_argument('path',metavar='Path to corpus',type=str, nargs=1)
parser.add_argument('--trainTokens',default=800000000)
parser.add_argument('--testTokens',default=200000000)
parser.add_argument('--splitFiles',default=100)
parser.add_argument('--trainPrefix',default='trainingCorpus')
parser.add_argument('--testPrefix',default='testingCorpus')
parser.add_argument('--minOccurrence',default=8)
parser.add_argument('--keepNonLatin', action="store_true")
args = parser.parse_args()

trainPrefix = args.trainPrefix
testPrefix = args.testPrefix
trainTokens = args.trainTokens
testTokens= args.testTokens
splitFiles = args.splitFiles
minOccurrence = args.minOccurrence
keepNonLatin = args.keepNonLatin
path = args.path[0]


if os.path.isfile(path):
    files = [path]

elif os.path.isdir(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

else:
    print("No corpus found")

outdir = 'out'
if not os.path.exists(outdir) or not os.path.isdir(outdir):
    os.makedirs(outdir)

if not os.path.exists(os.path.join(outdir,'training')) or not os.path.isdir(os.path.join(outdir,'training')):
    os.makedirs(os.path.join(outdir,'training'))

if not os.path.exists(os.path.join(outdir,'testing')) or not os.path.isdir(os.path.join(outdir,'testing')):
    os.makedirs(os.path.join(outdir,'testing'))

fileCount = 0
invalidTokenCount = 0
foreignTokenCount = 0
foreignWords = dict()
lines = 0
freqWords = dict()
unFreqWords = dict()
data = []
for file in files:
    fileCount += 1
    filePath = os.path.join(path,file)
    print("Processing "+filePath+(" (%d in %d)" % (fileCount, len(files))))
    with open(filePath,encoding="utf8") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            hasForeignToken = False
            if not keepNonLatin:
                for word in line.split():
                    if not only_roman_chars(word):
                        foreignTokenCount += 1
                        hasForeignToken = True
                        if  word in foreignWords:
                            foreignWords[word] += 1
                        else:
                            foreignWords[word] = 1
            if not hasForeignToken:
                for word in line.split():
                    if not word in freqWords: 
                        if word in unFreqWords:
                            if unFreqWords[word] == minOccurrence -1 :
                                freqWords[word] = 1
                                unFreqWords.pop(word)
                            else:
                                unFreqWords[word] += 1
                        else:
                            unFreqWords[word] = 1
                    else:
                        freqWords[word] +=1
                lines += 1
                data.extend([(random.random(),line)])
                    
        
print('Foreign tokens occurrences: %d' % foreignTokenCount)
print('Foreign tokens %d'% len(foreignWords))

with open(os.path.join(outdir,'nonLatin.txt'),'w+',encoding='utf8') as f:
    f.write('Non latin tokens occurrences: %d \n' % foreignTokenCount)
    f.write('Non latin tokens %d \n'% len(foreignWords))
    xs = [(k,foreignWords[k]) for k in sorted(foreignWords, key=foreignWords.get, reverse=True)]
    for x in xs:
        f.write(x[0] +' '+str(x[1])+'\n')

data.sort()

print('Frequent word occurrences: %d' % sum(freqWords.values()))
print('Frequent words: %d' % len(freqWords))

with open(os.path.join(outdir,'freqWords.txt'),'w+',encoding='utf8') as f:
    f.write('Frequent word occurrences: %d\n' % sum(freqWords.values()))
    f.write('Frequent words: %d\n' % len(freqWords))
    xs = [(k,freqWords[k]) for k in sorted(freqWords, key=freqWords.get, reverse=True)]
    for x in xs:
        f.write(x[0]+' '+str(x[1])+'\n')

tokensPerFile = math.trunc(trainTokens / splitFiles)

lineCount = 0
vocab = dict()

outTrainTokens,lineCount,vocab,outTrainLines = writeShuffledFile(trainTokens,tokensPerFile,data,os.path.join(outdir,'training'),trainPrefix,lineCount,vocab)

print('Training tokens: %d' % outTrainTokens )
print('Training lines: %d' % outTrainLines)

tokensPerFile = math.trunc(testTokens/splitFiles)
outTestTokens,lineCount,vocab,outTestLines = writeShuffledFile(testTokens,tokensPerFile,data,os.path.join(outdir,'testing'),testPrefix,lineCount,vocab)

print('Testing tokens: %d' % outTestTokens)
print('Testing lines: %d' % outTestLines)
s = [(k,vocab[k]) for k in sorted(vocab, key=vocab.get, reverse=True)]
print('Words in vocabulary: %d' % len(s))

voc = [("</S>",0),("<S>",0),("<UNK>",0)]
voc.extend(s)
with open(os.path.join(outdir,'vocab-freqs.txt'),'w+',encoding='utf8') as f:
    for x in voc:
        f.write(x[0] +' '+str(x[1])+ "\n")

with open(os.path.join(outdir,'vocab.txt'),'w+',encoding='utf8') as f:
    for x in voc:
        f.write(x[0] +"\n")


