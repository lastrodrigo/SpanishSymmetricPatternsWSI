# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:53:57 2018

@author: rlastra

Creates a vocabulary file in the format required by bilm-tf from a clean corpus 
consisting of 1 or many files formed by 1 sentence per line.
"""
import sys
import datetime
import os
import argparse


def  word_count(str,counts,file):
    """Adds 1 to the occurrences of word in str in the dictionary counts"""
    words = str.split()
    for word in words:
        
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

parser = argparse.ArgumentParser(description='Count word occurrences')
parser.add_argument('path',metavar='Path to corpus',type=str, nargs=1)
args = parser.parse_args()
fname = args.path[0]

if os.path.isfile(fname):
    files = [fname]
elif os.path.isdir(fname):
    files = [f for f in os.listdir(fname) if os.path.isfile(os.path.join(fname, f))]
else:
    print("No files found")
counts = dict()
sentences = 0
count = 0
for file in files:    
    count += 1
    print("Processing "+os.path.join(fname,file)+(" (%d in %d)" % (count, len(files))))
    with open(os.path.join(fname,file),encoding="utf8") as f:
        content = f.readlines()
    
        content = [x.strip() for x in content]
        
        for x in content:
            counts = word_count(x,counts,file)
        sentences += len(content)
    print("Done!")

print("\n\n\nTOTALS:")        
print(str(sentences)+" sentences processed")
print(str(len(counts))+" distinct words")
print(str(sum(counts.values()))+" tokens")

s = [(k,counts[k]) for k in sorted(counts, key=counts.get, reverse=True)]
print("Top 10 words: ")
print(s[0:10])

now = datetime.datetime.now()
path = "vocab-freq.txt"
with open(path,"w+",encoding="utf8") as f:
    for x in s:
        f.write(x[0] +' '+str(x[1])+ "\n")
