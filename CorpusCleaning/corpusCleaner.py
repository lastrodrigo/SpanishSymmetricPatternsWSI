import argparse
import os
import unicodedata as ud
import random
import math

latin_letters = {}


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
                line = data[lineCount]
                f.write(line+'\n')
                lines += 1
                tokenCount += len(line.split())
                for word in line.split():
                    vocab = word_count(word,vocab) 
                totalTokenCount += len(line.split())
                lineCount += 1
    return totalTokenCount,lineCount,vocab,lines

parser = argparse.ArgumentParser(description='Clean and shuffle corpus')
parser.add_argument('path',metavar='Path to corpus',type=str, nargs=1)
parser.add_argument('--trainTokens',default=0.8)
parser.add_argument('--testTokens',default=0.2)
parser.add_argument('--splitFiles',default=100)
parser.add_argument('--trainPrefix',default='trainingCorpus')
parser.add_argument('--testPrefix',default='testingCorpus')
parser.add_argument('--minOccurrence',default=0)
parser.add_argument('--vocabOccurrences')
parser.add_argument('--maxSentenceLength',default=62)
parser.add_argument('--vocabSize',default = 16384)
parser.add_argument('--maxTokens',default = 1000000000)

args = parser.parse_args()

trainPrefix = args.trainPrefix
testPrefix = args.testPrefix
trainTokens = args.trainTokens
testTokens= args.testTokens
splitFiles = args.splitFiles
minOccurrence = args.minOccurrence
maxSentenceLength = args.maxSentenceLength
vocabSize = args.vocabSize - 3
maxTokens = args.maxTokens

path = args.path[0]
validWords = dict()
if args.vocabOccurrences is not None:

    vocabFile = args.vocabOccurrences
    print('Processing vocabulary file: %s' % vocabFile)
    with open(vocabFile,'r',encoding='utf8') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            word,occurrences = line.split()
            if int(occurrences) >= minOccurrence and len(validWords) < vocabSize:
                validWords[word] = int(occurrences)

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
validTokenCount = 0
lines = 0
data = []
for file in files:
    fileCount += 1
    filePath = os.path.join(path,file)
    print("Processing "+filePath+(" (%d in %d)" % (fileCount, len(files))))
    with open(filePath,encoding="utf8") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            if len(line.split()) <= maxSentenceLength and validTokenCount < maxTokens:
                hasInvalidWord = False
                for word in line.split():
                    if not word in validWords: 
                        hasInvalidWord = True
                if not hasInvalidWord:
                    lines += 1
                    data.append(line)
                    validTokenCount += len(line.split())

random.shuffle(data)

print('High occurrence words: %d' % len(validWords))
print('Valid lines: %d' % lines)
print('Total valid tokens: %d' % validTokenCount)
#with open(os.path.join(outdir,'freqWords.txt'),'w+',encoding='utf8') as f:
#    f.write('High occurrence words: %d\n' % len(validWords))
#    f.write('Valid lines: %d\n' % lines)

tokensPerFile = math.trunc(validTokenCount * trainTokens / splitFiles)

lineCount = 0
vocab = dict()

outTrainTokens,lineCount,vocab,outTrainLines = writeShuffledFile(tokensPerFile * splitFiles,tokensPerFile,data,os.path.join(outdir,'training'),trainPrefix,lineCount,vocab)

print('Training tokens: %d' % outTrainTokens )
print('Training lines: %d' % outTrainLines)

tokensPerFile = math.trunc(validTokenCount * testTokens / splitFiles)
outTestTokens,lineCount,vocab,outTestLines = writeShuffledFile(tokensPerFile * splitFiles,tokensPerFile,data,os.path.join(outdir,'testing'),testPrefix,lineCount,vocab)

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


