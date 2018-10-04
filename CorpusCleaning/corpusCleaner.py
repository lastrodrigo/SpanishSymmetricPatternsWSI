import argparse
import os
import unicodedata as ud

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

parser = argparse.ArgumentParser(description='Clean corpus')
parser.add_argument('path',metavar='Path to corpus',type=str, nargs=1)
parser.add_argument('--foreignTokens',action='store_true',)

args = parser.parse_args()

path = args.path[0]
print(path)

if os.path.isfile(path):
    files = [path]

elif os.path.isdir(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

else:
    print("No files found")

fileCount = 0
foreignTokenCount = 0
foreignWords = dict()
for file in files:
    fileCount += 1
    filePath = os.path.join(path,file)
    print("Processing "+filePath+(" (%d in %d)" % (fileCount, len(files))))
    with open(filePath,encoding="utf8") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
                for word in line.split():
                    if not args.foreignTokens: 
                        
                        if not only_roman_chars(word):
                            foreignTokenCount += 1
                            if  word in foreignWords:
                                foreignWords[word] += 1
                            else:
                                foreignWords[word] = 1
        

if not args.foreignTokens:
    print('Foreign tokens occurrences: %d' % foreignTokenCount)
    print('Foreign tokens %d'% len(foreignWords))

with open('foreign.txt','w+',encoding='utf8') as f:
    f.write('Foreign tokens occurrences: %d \n' % foreignTokenCount)
    f.write('Foreign tokens %d \n'% len(foreignWords))
    xs = [(k,foreignWords[k]) for k in sorted(foreignWords, key=foreignWords.get, reverse=True)]
    for x in xs:
        f.write(x[0] +' '+str(x[1])+'\n')

