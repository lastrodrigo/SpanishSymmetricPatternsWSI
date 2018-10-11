import argparse
import os
import math



parser = argparse.ArgumentParser(description='Clean corpus')
parser.add_argument('path',metavar='Path to corpus',type=str, nargs=1)
parser.add_argument('--splitFiles',default=100)
parser.add_argument('--threshold',default=5)
parser.add_argument('--outputPrefix',default='corpus')
args = parser.parse_args()
splitFiles = args.splitFiles
threshold = args.threshold
outputPrefix = args.outputPrefix
path = args.path[0]
if os.path.isfile(path):
    files = [path]

elif os.path.isdir(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

else:
    print("No files found")

outdir = 'out-lowOccurrenceCleaner'
if not os.path.exists(outdir) or not os.path.isdir(outdir):
    os.makedirs(outdir)

fileCount = 0
freqWords = []
unFreqWords = dict()
data = []
lines = 0
for file in files:
    fileCount += 1
    filePath = os.path.join(path,file)
    print("Processing "+filePath+(" (%d in %d)" % (fileCount, len(files))))
    with open(filePath,encoding="utf8") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            lines += 1
            for word in line.split():
                if not word in freqWords: 
                    if word in unFreqWords:
                        if unFreqWords[word] = threshold:
                            freqWords.extend(word)
                            unFreqWords.pop(word)
            data.extend(line)

print('Low occurrence (<= %d) tokens occurrences: %d' % sum(unFreqWords.values))
print('Low occurrence tokens %d'% len(unFreqWords))

with open(os.path.join(outdir,'lowOccur.txt'),'w+',encoding='utf8') as f:
    f.write('Low occurrence (<= %d) tokens occurrences: %d \n' % sum(unFreqWords.values))
    f.write('Low occurrence tokens %d \n'% len(unFreqWords))
    xs = [(k,unFreqWords[k]) for k in sorted(unFreqWords, key=unFreqWords.get, reverse=True)]
    for x in xs:
        f.write(x[0] +' '+str(x[1])+'\n')

linesPerFile = math.trunc(lines / splitFiles)
fileCount = 0
totalLineCount = 0
fileLineCount = 0
lineIterator = 0
while lineIterator < len(data):
    fileCount += 1
    fileLineCount = 0
    fileName = os.path.join(outdir,prefix+'.'+str(fileCount)+'.noLowOccur' )
    with open(fileName,'w+',encoding='utf8') as f: 
        print('Processsing output %s' % fileName)
        while lineIterator < len(data) and fileLineCount <= linesPerFile:
            hasUnFreqWord = False
            for word in data[lineIterator].split():
                if not word in freqWords:
                    hasUnFreqWord = True

            if not hasUnFreqWord:        
                f.write(data[lineIterator]+'\n')
                fileLineCount += 1
                totalLineCount += 1
                
            lineIterator += 1
             
    
