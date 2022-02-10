import torch
import os
import numpy as np
import random
import csv
import pickle
def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def getfewshot(inpath,outpath,thislabel,fewshotnum):
    ###read from inpath
    print(thislabel)
    intrain = inpath + "/train.csv"
    intest = inpath + "/test.csv"
    alllabel = []
    trainresult = {}
    f = open(intrain, 'r')
    reader = csv.reader(f)
    for item in reader:
        if thislabel[int(item[0]) - 1] not in alllabel:
            alllabel.append(thislabel[int(item[0]) - 1])
        content = ""
        for i in range(1,len(item)):
            if i == 1:
                content = content + item[i].replace("\t"," ")
            else:
                content = content + " " + item[i].replace("\t"," ")
        if thislabel[int(item[0]) - 1] not in trainresult:
            trainresult[thislabel[int(item[0]) - 1]] = [content]
        else:
            trainresult[thislabel[int(item[0]) - 1]].append(content)
    f.close()
    testresult = {}
    f = open(intest, 'r')
    reader = csv.reader(f)
    for item in reader:
        content = ""
        for i in range(1,len(item)):
            if i == 1:
                content = content + item[i].replace("\t"," ")
            else:
                content = content + " " + item[i].replace("\t"," ")
        if thislabel[int(item[0]) - 1] not in testresult:
            testresult[thislabel[int(item[0]) - 1]] = [content]
        else:
            testresult[thislabel[int(item[0]) - 1]].append(content)
    f.close()
    fewtrainname = outpath + "/train.txt"
    fewvalidname = outpath + "/valid.txt"
    fewtestname = outpath + "/test.txt"

    tousetres = {}
    for key in trainresult.keys():
        if 2 * fewshotnum < len(trainresult[key]):
            thisres = random.sample(trainresult[key], 2 * fewshotnum)
        else:
            thisres = trainresult[key]
        tousetres[key] = thisres

    sampletestres = {}
    for key in testresult.keys():
        sampletestnum = 1000
        if sampletestnum < len(testresult[key]):
            thisres = random.sample(testresult[key], sampletestnum)
        else:
            thisres = testresult[key]
        sampletestres[key] = thisres

    tousetrainres = {}
    tousevalidres = {}
    for key in tousetres.keys():
        allres = tousetres[key]
        fortrain = allres[0:fewshotnum]
        forvalid = allres[fewshotnum:2 * fewshotnum]
        tousetrainres[key] = fortrain
        tousevalidres[key] = forvalid
    f = open(fewtrainname,'w')
    for key in tousetrainres.keys():
        for one in tousetrainres[key]:
            f.write(one+"\t"+key + "\n")
    f.close()

    f = open(fewvalidname, 'w')
    for key in tousevalidres.keys():
        for one in tousevalidres[key]:
            f.write(one + "\t" + key + "\n")
    f.close()

    ####test
    f = open(fewtestname, 'w')
    for key in sampletestres.keys():
        for one in sampletestres[key]:
            f.write(one + "\t" + key + "\n")
    f.close()

def getpromptembedding(model,tokenizer,promptnumber,taskname,labellist):
    t5_embedding = model.model.get_input_embeddings()
    promptinitembedding = torch.FloatTensor(promptnumber, t5_embedding.weight.size(1))
    startindex = 0
    alllabel = ["sentence classification"]
    alllabel.append(taskname)
    alllabel.extend(labellist)
    print(alllabel)
    for one in alllabel:
        encoderes = tokenizer.batch_encode_plus([one], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        embeddingres = t5_embedding(touse).clone().detach()
        if embeddingres.shape[0] > 1:
            embeddingres = torch.mean(embeddingres, 0, keepdim=True)
        promptinitembedding[startindex] = embeddingres
        startindex += 1
    fr = open('allnumber.pickle', 'rb')
    alltokens = pickle.load(fr)
    sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
    top5000 = []
    for one in sortedalltoken:
        if one[0] == 2:
            continue
        else:
            if len(top5000) < 5000:
                top5000.append(one)
            else:
                break
    vocab = tokenizer.get_vocab()
    randomtokennum = promptnumber - len(alllabel)
    touse = random.sample(top5000, randomtokennum)
    print(touse)
    for one in touse:
        promptinitembedding[startindex] = t5_embedding.weight[one[0]].clone().detach()
        startindex += 1
    return promptinitembedding

def getmemdata(alldatafile,memdatafile,memeveryclass):
    f = open(alldatafile, 'r')
    alldata = {}
    while True:
        line = f.readline().strip()
        if not line:
            break
        linelist = line.split('\t')
        if len(linelist) != 2:
            continue
        content = linelist[0]
        label = linelist[1]
        if label not in alldata:
            alldata[label] = [content]
        else:
            alldata[label].append(content)
    f.close()
    fo = open(memdatafile,'w')
    allmemdata = {}
    for key in alldata.keys():
        if memeveryclass < len(alldata[key]):
            thisres = random.sample(alldata[key], memeveryclass)
        else:
            thisres = alldata[key]
        allmemdata[key] = thisres
    for key in allmemdata.keys():
        for one in allmemdata[key]:
            fo.write(one + "\t" + key + "\n")
    fo.close()