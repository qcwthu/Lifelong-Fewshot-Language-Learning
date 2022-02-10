import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import pdb
import numpy as np
import time
import random
import re
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def ifinclude(str1,str2):
    #name.lower() in linelist[0].lower():
    str1list = str1.lower().split(' ')  ####name
    str2list = str2.lower().split(' ')  ####linelist
    ifin = False
    for i in range(0,len(str2list)):
        if str2list[i] == str1list[0]:
            ifin = True
            for j in range(1,len(str1list)):
                if str2list[i+j] != str1list[j]:
                    ifin = False
                    break
            if ifin == True:
                break
        else:
            continue
    return ifin




def handlefile(inputfile,outputfile,allnumber,trainnumber):
    f = open(inputfile,'r')
    allres = {}
    alltype = []
    for key in allnumber.keys():
        alltype.append(key)
    insen = 0
    allin = {}
    notinsen = 0
    allnotin = {}
    while True:
        line = f.readline().strip()
        if not line:
            break
        linelist = line.split("__ans__")
        if len(linelist) != 2:
            continue
        entitylist = linelist[1]
        if entitylist == 'end':
            continue
        if ';' not in entitylist:
            continue
        allentity = entitylist.split(";")
        if len(allentity) != 2:
            continue
        firstentity = allentity[0]
        #print(firstentity)
        if '!' not in firstentity:
            continue
        splitent = firstentity.split('!')
        if len(splitent) != 2:
            continue
        thistype = splitent[1].strip()
        #print(thistype)
        if thistype not in alltype:
            continue

        #print(linelist[0] + '\t' + linelist[1])
        name = linelist[1].split(";")[0].split("!")[0].strip(' ')
        entype = linelist[1].split(";")[0].split("!")[1].strip(' ')
        whole = name + " ! " + entype + " ;"
        #print(name)
        #####some filters
        thissen = linelist[0]
        ####length
        # senlist = thissen.split(' ')
        # if len(senlist) <= 3:
        #     continue
        # digitnum = 0
        # for one in senlist:
        #     if re.search(r'\d', one):
        #         digitnum += 1
        # if len(senlist) - digitnum < 1:
        #     continue

        #ifin = ifinclude(name,linelist[0])
        #if ifin:
        if name.lower() in linelist[0].lower():
            length = len(name)
            startindex = linelist[0].lower().find(name.lower())
            endindex = startindex + length
            toreplace = linelist[0][startindex:endindex]
            #newsen = linelist[0]
            newsen = linelist[0].replace(toreplace,name)
            if thistype not in allin:
                #allin[thistype] = [linelist[0] + '\t' + linelist[1]]
                allin[thistype] = {}
                if whole not in allin[thistype]:
                    insen += 1
                    allin[thistype][whole] = [newsen]
                #else:
                #    allin[thistype][whole].append(linelist[0])
            else:
                #allin[thistype].append(linelist[0] + '\t' + linelist[1])
                if whole not in allin[thistype]:
                    insen += 1
                    allin[thistype][whole] = [newsen]
                #else:
                #    allin[thistype][whole].append(linelist[0])
        else:
            ########some filter
            ##ensure the entity has similar words in sen
            # if name.lower() in linelist[0].lower():
            #     ###thisone will be used
            #     str1list = name.lower().split(' ')  ####name
            #     nolowlist = name.split(' ')
            #     str2list = linelist[0].lower().split(' ')  ####linelist
            #     ifin = False
            #     touselist = linelist[0].split(' ')
            #     for i in range(0, len(str2list)):
            #         if str1list[0] in str2list[i]:
            #             touselist[i] = nolowlist[0]
            #             for j in range(1,len(str1list)):
            #                 touselist[i+j] = nolowlist[j]
            #         else:
            #             continue
            #     newsen = ' '.join(touselist)
            # else:
            #     ####whether first similar 0.75  5
            #     str1list = name.lower().split(' ')
            #     tousestr = str1list[0]
            #     str2list = linelist[0].lower().split(' ')
            #     ifhave = 0
            #     index = -1
            #     for j in range(0,len(str2list)):
            #         thistoken = str2list[j]
            #         samenum = 0
            #         for k in range(min(len(tousestr),len(thistoken))):
            #             if tousestr[k] == thistoken[k]:
            #                 samenum += 1
            #             else:
            #                 break
            #         if min(len(tousestr),len(thistoken)) == 0:
            #             continue
            #         if samenum >= 5 or float(samenum) / float(min(len(tousestr),len(thistoken))) >= 0.75:
            #             ifhave = 1
            #             index = j
            #             break
            #     if not ifhave:
            #         continue
            #     else:
            #         ###replace
            #         newlinelist = linelist[0].split()[0:index] + name.split(' ') + linelist[0].split()[index+1:]
            #         newsen = " ".join(newlinelist)

            if thistype not in allnotin:
                #allnotin[thistype] = [linelist[0] + '\t' + linelist[1]]
                allnotin[thistype] = {}
                if whole not in allnotin[thistype]:
                    notinsen += 1
                    newsen = linelist[0] + " " + name
                    allnotin[thistype][whole] = [newsen]
                #else:
                #    allnotin[thistype][whole].append(linelist[0])
            else:
                #allnotin[thistype].append(linelist[0] + '\t' + linelist[1])
                if whole not in allnotin[thistype]:
                    notinsen += 1
                    newsen = linelist[0] + " " + name
                    allnotin[thistype][whole] = [newsen]
                #else:
                #    allnotin[thistype][whole].append(linelist[0])
    f.close()
    print(insen)
    print(notinsen)
    # for key in allin:
    #     print(key+"\t"+str(len(allin[key])))
    # for key in allnotin:
    #     print(key+"\t"+str(len(allnotin[key])))
    # for key in allin:
    #     for one in allin[key]:
    #         for aa in allin[key][one]:
    #            print(aa+"  "+one)
    # for key in allnotin:
    #     for one in allnotin[key]:
    #         for aa in allnotin[key][one]:
    #             print(aa + "  " + one)
    finalres = {}
    fall = open("allgenerate",'w')
    for key in allnumber.keys():
        finalres[key] = []
    for key in allin:
        for one in allin[key]:
            for aa in allin[key][one]:
                finalres[key].append(aa+"\t"+one)
                fall.write(aa+"\t"+one+'\n')
    for key in allnotin:
        for one in allnotin[key]:
            for aa in allnotin[key][one]:
                finalres[key].append(aa+"\t"+one)
                fall.write(aa + "\t" + one + '\n')
    fall.close()
    #for key in finalres.keys():
    #    print(len(finalres[key]))
    sampleres = []
    trainres = []
    validres = []

    for key in finalres.keys():
        thissample = random.sample(finalres[key],allnumber[key])
        #print(thissample)
        sampleres.extend(thissample)
        ####divide to train and valid
        thistrainnum = trainnumber[key]
        indexlist = [i for i in range(allnumber[key])]
        #print(indexlist)
        trainuse = random.sample(indexlist,thistrainnum)
        #print(trainuse)
        for j in range(allnumber[key]):
            if j in trainuse:
                trainres.append(thissample[j])
            else:
                validres.append(thissample[j])
        #print(trainres)
        #print(validres)
    #print(sampleres)
    fo = open(outputfile, 'w')
    for one in sampleres:
        fo.write(one+"\n")
    fo.close()
    fot = open('train_mem.txt', 'w')
    for one in trainres:
        fot.write(one+"\n")
    fot.close()
    fov = open('valid_mem.txt', 'w')
    for one in validres:
        fov.write(one + "\n")
    fov.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")

    parser.add_argument("--model", dest="model", type=str,
                        default="T5", help="{T5}")
    parser.add_argument("--seed", dest="seed", type=int,
                        default=160, help="seed for network")

    args = parser.parse_args()

    seed_everything(args)

    if args.model == "T5":
        #seed 100
        #train: person:10 location:12 org:6 mix:7
        #valid: person:16 location:12 org:11 mix:8
        print("right!")
        # allnumber = {'org': 17, 'location': 24, 'person': 26, 'mix': 15}
        # trainnumber = {'org': 6, 'location': 12, 'person': 10, 'mix': 7}
        # allnumber = {'org':15,'location':14,'person':11,'mix':9}
        # trainnumber = {'org':7,'location':8,'person':5,'mix':4}
        allnumber = {'org': 16, 'location': 21, 'person': 20, 'mix': 16}
        trainnumber = {'org': 7, 'location': 10, 'person': 11, 'mix': 6}
        handlefile("pseudosamples", "allselect", allnumber, trainnumber)
    else:
        raise Exception("No such model! Please make sure that `model` takes the value in {T5}")


