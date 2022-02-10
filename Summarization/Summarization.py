
import argparse
import gc
gc.enable()

import time
import logging

from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from torch.cuda.amp import autocast as autocast
from torch.utils import data
from torch.utils.data import (
    SequentialSampler, RandomSampler
)

from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler
import pickle
from model import *
from dataset import *
from utils import *
from datasets import load_metric

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def get_dataloader(num_workers,dataset, batch_size, max_len, pad_id, sampler):
    collate_fn = SmartBatchingCollate(
        max_length=max_len,
        pad_token_id=pad_id
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def train(args, model, train_dataset,valid_dataset,onerun):
    # total step
    step_tot = (len(
        train_dataset) // args.gradient_accumulation_steps // args.batch_size_per_gpu // args.n_gpu) * args.max_epoch
    warmup_steps_total = step_tot * args.warmup_steps
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(
        train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader(args.num_workers, train_dataset, args.batch_size_per_gpu, args.max_length,
                                      train_dataset.tokenizer.pad_token_id,train_sampler)
    valid_dataloader = get_dataloader(args.num_workers, valid_dataset, args.valid_size_per_gpu, args.max_length,
                                      valid_dataset.tokenizer.pad_token_id,valid_sampler)

    base_optimizer_arguments = {"lr": args.lr, "clip_threshold": args.max_grad_norm, "decay_rate": -0.8,
                                "weight_decay": args.weight_decay,
                                "scale_parameter": False, "relative_step": False}
    optimizer = Adafactor
    optimizer = OSS(params=filter(lambda p: p.requires_grad, model.parameters()), optim=optimizer,
                    **base_optimizer_arguments)
    # distributed training
    model = ShardedDDP(model, optimizer)
    model.train()
    scaler = ShardedGradScaler()
    scheduler = None
    #scaler = None

    startepoch = 0
    Best_F1 = 0.0

    logger.info("Begin train...")
    logger.info("We will train model in %d steps" % step_tot)

    result_dict = {
        'epoch': [],
        'val_rouge1': [],
        'best_val_rouge1': Best_F1
    }
    global_step = 0
    lm_lambda = args.lm_lambda
    kd_lamda = args.kd_lamda
    for i in range(startepoch, startepoch + args.max_epoch):
        thisevalstep = args.eval_step
        logger.info(i)
        model.train()
        result_dict['epoch'] = i
        allloss = []
        alllmloss = []
        allkdloss = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),"ifmem": batch[8].to(args.device)}
            inputs_lm = {"input_ids": batch[4].to(args.device), "attention_mask": batch[5].to(args.device),
                         "target_ids": batch[6].to(args.device), "target_mask": batch[7].to(args.device)}
            if scaler is not None:
                with autocast():
                    loss, kdloss = model(inputs,ifcalpre=True)
                    lmloss = model(inputs_lm,ifcalpre=False) * lm_lambda
            else:
                loss, kdloss = model(inputs,ifcalpre=True)
                lmloss = model(inputs_lm,ifcalpre=False) * lm_lambda
            finalloss = loss + lmloss
            finalloss = finalloss * (1.0 - kd_lamda) + kdloss * kd_lamda
            if scaler is not None:
                scaler.scale(finalloss).backward()
            else:
                finalloss.backward()
            allloss.append(loss.item())
            alllmloss.append(lmloss.item())
            allkdloss.append(kdloss.item())

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.log_step == 0:
                    logger.info("step: %d, shcedule: %.3f, loss: %.6f, lmloss: %.6f, kdloss: %.6f" % (
                        global_step, global_step / step_tot, np.average(allloss), np.average(alllmloss),
                        np.average(allkdloss)))

                if args.local_rank in [0, -1] and global_step % thisevalstep == 0:
                    print("not eval!!!")
                    model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            if i >= 8:
                dooneeval(model,valid_dataloader,args,result_dict,optimizer,scaler,i,onerun)
                model.train()

        if args.train_sample:
            logger.info("sampling...")
            logger.info("sampled")

    torch.cuda.empty_cache()
    del model, optimizer, scheduler, scaler, train_dataloader, valid_dataloader,
    gc.collect()

def dooneeval(modeltoeval,valid_dataloader,args,result_dict,optimizer,scaler,i,onerun):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    logger.info("Do one eval!")
    allytrue = []
    allypred = []
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in enumerate(valid_dataloader):
            logger.info(step)
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen, target, preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    rouge = load_metric('rouge')
    rouge_score = rouge.compute(references=allytrue, predictions=allypred)
    logger.info('----Validation Results Summary----')
    logger.info(len(allypred))
    logger.info(rouge_score)
    logger.info("valid_rouge1: %f", rouge_score["rouge1"].mid.fmeasure)
    logger.info("valid_rouge2: %f", rouge_score["rouge2"].mid.fmeasure)
    logger.info("valid_rougeL: %f", rouge_score["rougeL"].mid.fmeasure)

    result_dict['val_rouge1'].append(rouge_score["rouge1"].mid.fmeasure)
    if result_dict['val_rouge1'][-1] > result_dict['best_val_rouge1']:
        logger.info("{} epoch, best epoch was updated! val_rouge1: {: >4.5f}".format(i, result_dict['val_rouge1'][-1]))
        result_dict["best_val_rouge1"] = result_dict['val_rouge1'][-1]
        if not os.path.exists(args.tosavepath):
            os.mkdir(args.tosavepath)
        if not os.path.exists(args.tosavepath + "/" + args.taskfold):
            os.mkdir(args.tosavepath + "/" + args.taskfold)
        if not os.path.exists(args.tosavepath + "/" + args.taskfold + "/" + str(onerun)):
            os.mkdir(args.tosavepath + "/" + args.taskfold + "/" + str(onerun))
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": model_to_save.promptembedding
        }
        torch.save(ckpt, os.path.join(args.tosavepath + "/" + args.taskfold + "/" + str(onerun), "bestckpt"))

def test(args, test_dataset, onerun):

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = get_dataloader(args.num_workers, test_dataset, args.test_size_per_gpu, args.max_length,
                                      test_dataset.tokenizer.pad_token_id,test_sampler)

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
    model = T5forSummarization(args, t5model, test_dataset.tokenizer)
    allckpt = torch.load(args.tosavepath +  "/" + args.taskfold + "/" + str(onerun) + "/bestckpt")
    model.promptnumber = allckpt["promptnumber"]
    model.promptembedding = allckpt["promptembedding"]
    logger.info("load finished!")

    model.to(args.device)
    model.eval()
    scaler = ShardedGradScaler()
    #scaler = None
    allytrue = []
    allypred = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen,target,preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    rouge = load_metric('rouge')
    rouge_score = rouge.compute(references=allytrue, predictions=allypred)
    logger.info('----Test Results Summary----')
    logger.info(len(allypred))
    logger.info(rouge_score)
    logger.info("test_rouge1: %f", rouge_score["rouge1"].mid.fmeasure)
    logger.info("test_rouge2: %f", rouge_score["rouge2"].mid.fmeasure)
    logger.info("test_rougeL: %f", rouge_score["rougeL"].mid.fmeasure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="4", help="gpu id")

    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--lm_lambda", dest="lm_lambda", type=float,
                        default=0.25, help='language model loss lambda')
    parser.add_argument("--kd_lamda", dest="kd_lamda", type=float,
                        default=0.05, help='kd loss lambda')
    parser.add_argument("--startindex", dest="startindex", type=int,
                        default=0, help="start index")
    parser.add_argument("--taskindex", dest="taskindex", type=int,
                        default=0, help="task index")
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int,
                        default=16, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu", dest="valid_size_per_gpu", type=int,
                        default=24, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu", dest="test_size_per_gpu", type=int,
                        default=24, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                        default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int,
                        default=5, help="max epoch number")
    parser.add_argument("--num_workers", dest="num_workers", type=int,
                        default=4, help="dataloader num_workers")

    parser.add_argument("--save_step", dest="save_step", type=int,
                        default=100000, help="step to save")
    parser.add_argument("--log_step", dest="log_step", type=int,
                        default=1, help="how many steps to log")
    parser.add_argument("--eval_step", dest="eval_step", type=int,
                        default=100, help="how many steps to eval")

    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="t5_ckpt", help="ckpt dir to save")
    parser.add_argument("--tosavepath", dest="tosavepath", type=str,
                        default="t5_sum_ckpt", help="ckpt dir to save")
    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")


    parser.add_argument("--model", dest="model", type=str,
                        default="T5NER", help="{T5NER}")
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="t5-base", help="{t5-base,google/t5-v1_1-base}")
    parser.add_argument("--train_file_name", dest="train_file_name", type=str,
                        default="data_conll/newtrain.txt", help="train data file path")
    parser.add_argument("--valid_file_name", dest="valid_file_name", type=str,
                        default="data_conll/newvalid.txt", help="valid data file path")
    parser.add_argument("--test_file_name", dest="test_file_name", type=str,
                        default="data_conll/newtest.txt", help="test data file path")
    parser.add_argument("--train_sample", action="store_true",
                        help="dynamic sample or not")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=128, help="max sentence length")

    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default = 1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=float,
                        default=0.1, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1.0, help="max grad norm")

    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")

    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=0, help="whether to use lm_adapted model")
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default="../t5_ckpt_1_0622_bak/t5_ckpt/ckpt_of_step_100000",
                        help="The path of lm_adapted model")
    parser.add_argument("--cache_path", dest="cache_path", type=str,
                        default="/data/qin/cache/",
                        help="The path of huggingface cache")
    parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                        default=100, help="The number of prompt")
    parser.add_argument("--ifckpt_onlymodel", dest="ifckpt_onlymodel", type=int,
                        default=1, help="If ckpt only contains model. Default: True, only contains model")
    args = parser.parse_args()

    # print args
    #####handle startindex==1 and taskindex==1
    # if args.startindex == 1 and args.taskindex == 1:
    #     args.batch_size_per_gpu = 1
    #     args.gradient_accumulation_steps = 8
    print(args)
    # set cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    args.n_gpu = len(args.cuda.split(","))
    initialseed = args.seed
    seed_everything(args)

    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("./log"):
            os.mkdir("./log")
        with open("./log/trainner_log", 'a+') as f:
            f.write(str(time.ctime()) + "\n")
            f.write(str(args) + "\n")
            f.write("----------------------------------------------------------------------------\n")

    runtimes = 3
    alltaskfold = ["cnndm", "wikihow", "xsum"]
    alltaskname = ["cnn daily mail ", "wiki how", "extreme summarization"]
    allgentasktoken = ["summerizationcnndm", "summerizationwikihow", "summerizationxsum"]
    tasknum = len(alltaskfold)
    dataprefix = "./data/"
    fewshotnum = 16
    if args.local_rank != -1:
        torch.distributed.barrier()
    startindex = args.startindex
    for onerun in range(startindex, startindex + 1):
        logger.info(onerun)
        args.seed = initialseed + onerun * 100
        seed_everything(args)
        logger.info("new seed %s", args.seed)
        allindex = [i for i in range(tasknum)]
        random.shuffle(allindex)
        newtaskname = [alltaskname[i] for i in allindex]
        print(newtaskname)
        newtaskfold = [alltaskfold[i] for i in allindex]
        print(newtaskfold)
        newtgentasktokens = [allgentasktoken[i] for i in allindex]
        print(newtgentasktokens)
        getnewfew = False
        if getnewfew:
            for j in range(len(newtaskfold)):
                onefold = newtaskfold[j]
                if not os.path.exists(dataprefix+onefold+"/"+str(onerun)+"_"+str(args.seed)):
                    os.mkdir(dataprefix+onefold+"/"+str(onerun)+"_"+str(args.seed))
                thispath = dataprefix+onefold+"/"+str(onerun)+"_"+str(args.seed)
                logger.info(thispath)
                getfewshot(dataprefix+onefold,thispath,fewshotnum)
        globaltokenizer = None
        newfilefolder = "newdata"
        memdatafolder = "memdata"
        if not os.path.exists(newfilefolder):
            os.mkdir(newfilefolder)
        if not os.path.exists(memdatafolder):
            os.mkdir(memdatafolder)
        tostart = args.taskindex
        for j in range(tostart, tostart + 1):
            thistaskname = newtaskname[j]
            thistaskfold = newtaskfold[j]
            args.taskfold = thistaskfold
            t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
            tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
            for gg in range(0, tostart + 1):
                gentasktoken = newtgentasktokens[gg]
                tokenizer.add_tokens(gentasktoken)
                # print(len(tokenizer))
                logger.info(
                    'gen token = {} , gen token id = {}'.format(gentasktoken,
                                                                tokenizer.convert_tokens_to_ids(gentasktoken)))
            answertoken = "__ans__"
            special_tokens = {"ans_token": answertoken}
            tokenizer.add_tokens(list(special_tokens.values()))
            special_token_ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in special_tokens.items()}

            globaltokenizer = tokenizer
            model = T5forSummarization(args, t5model, tokenizer)
            if j == 0:
                promptnumber = args.prompt_number
                promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname)
            else:
                logger.info("use previous prompt")
                logger.info("Previous ckpt fold name: %s", newtaskfold[j - 1])
                promptckpt = torch.load(args.tosavepath + "/" + newtaskfold[j - 1] + "/" + str(onerun) + "/bestckpt")
                promptnumber = args.prompt_number
                promptnumber_ckpt = promptckpt['promptnumber']
                assert promptnumber == promptnumber_ckpt
                promptembedding = promptckpt['promptembedding']
                print(promptembedding.shape)

            model.set_prompt_embedding(promptnumber, promptembedding)
            model.to(args.device)

            if j == 0:
                thistrainfilename = dataprefix + thistaskfold +"/" +str(onerun)+"_"+str(args.seed) + "/train.txt"
                thisvalidfilename = dataprefix + thistaskfold +"/" +str(onerun)+"_"+str(args.seed) + "/valid.txt"
                thistestfilename = dataprefix + thistaskfold +"/" +str(onerun)+"_"+str(args.seed) + "/test.txt"
                if not os.path.exists(newfilefolder + "/" + thistaskfold):
                    os.mkdir(newfilefolder + "/" + thistaskfold)
                newtrainfile = newfilefolder + "/" + thistaskfold + "/" + "train.txt"
                newvalidfile = newfilefolder + "/" + thistaskfold + "/" + "valid.txt"
                newtestfile = newfilefolder + "/" + thistaskfold + "/" + "test.txt"
                f = open(newtrainfile, 'w')
                for line in open(thistrainfilename, 'r'):
                    f.write(str(j) + "\t" + line)
                f.close()
                f = open(newvalidfile, 'w')
                for line in open(thisvalidfilename, 'r'):
                    f.write(str(j) + "\t" + line)
                f.close()
                f = open(newtestfile, 'w')
                for line in open(thistestfilename, 'r'):
                    f.write(str(j) + "\t" + line)
                f.close()
                args.train_file_name = newtrainfile
                args.valid_file_name = newvalidfile
                args.test_file_name = newtestfile
            else:
                logger.info("get all test data")
                allpretest = []
                for kk in range(0, j + 1):
                    onepretaskfold = newtaskfold[kk]
                    thistestfilename = dataprefix + onepretaskfold + "/" + str(onerun) + "_" + str(
                        args.seed) + "/test.txt"
                    allpretest.append(thistestfilename)
                if not os.path.exists(newfilefolder + "/" + thistaskfold):
                    os.mkdir(newfilefolder + "/" + thistaskfold)
                newtestfile = newfilefolder + "/" + thistaskfold + "/" + "test.txt"
                f = open(newtestfile, 'w')
                for aa in range(len(allpretest)):
                    oneprefile = allpretest[aa]
                    for line in open(oneprefile, 'r'):
                        f.write(str(aa) + "\t" + line)
                f.close()

                logger.info("generate pseodu samples for previous tasks")
                memnumforeveryclass = 4
                scalerpre = ShardedGradScaler()
                for aa in range(0, j):
                    onepremempath = memdatafolder + "/" + newtaskfold[aa]
                    if not os.path.exists(onepremempath):
                        os.mkdir(onepremempath)
                    logger.info("generate pseudo samples for task: %s", newtaskname[aa])
                    tousetestfile = dataprefix + newtaskfold[aa] + "/" + str(onerun) + "_" + str(args.seed) + "/test.txt"
                    tempfile = 'temptest.txt'
                    f = open(tempfile, 'w')
                    for line in open(tousetestfile, 'r'):
                        f.write(str(aa) + "\t" + line)
                    f.close()

                    pretestdataset = T5SummarizationDataset(tempfile, args.max_length, tokenizer, newtgentasktokens, answertoken, aa)
                    pretestsampler = SequentialSampler(pretestdataset)
                    pretestdataloader = get_dataloader(args.num_workers, pretestdataset, 32, args.max_length,
                                                     pretestdataset.tokenizer.pad_token_id, pretestsampler)
                    model.eval()
                    allsampleonetask = []
                    with torch.no_grad():
                        for step, batch in enumerate(pretestdataloader):
                            logger.info(step)
                            if step > 2:
                                break
                            inputs_lm = {"input_ids": batch[4].to(args.device),
                                         "attention_mask": batch[5].to(args.device),
                                         "target_ids": batch[6].to(args.device),
                                         "target_mask": batch[7].to(args.device)}
                            if scalerpre is not None:
                                with autocast():
                                    sen, target, preds = model._generative_samples(inputs_lm)
                            else:
                                sen, target, preds = model._generative_samples(inputs_lm)
                            thisdatanum = len(preds)
                            ifoldmethod = False
                            for ll in range(thisdatanum):
                                if ifoldmethod:
                                    samplelist = preds[ll].split(answertoken)
                                    if len(samplelist) != 2:
                                        continue
                                    onesen = samplelist[0].strip(' ')
                                    onesum = samplelist[1].strip(' ')
                                    onepseudodata = onesen + "\t" + onesum
                                    print(onepseudodata)
                                    allsampleonetask.append(onepseudodata)
                                else:
                                    samplelist = preds[ll].split(answertoken)
                                    onesen = samplelist[0].strip(' ')
                                    if len(onesen.split(' ')) >= 256:
                                        onesum = ""
                                        num = 0
                                        allwords = onesen.split(' ')
                                        for oneword in allwords:
                                            if len(oneword) == 0:
                                                continue
                                            onesum = onesum + oneword + " "
                                            if '.' == oneword[-1]:
                                                num += 1
                                            if num >= 3:
                                                break
                                        onepseudodata = onesen + "\t" + onesum.strip(' ')
                                        allsampleonetask.append(onepseudodata)
                    model.train()
                    print(len(allsampleonetask))
                    if 2 * memnumforeveryclass < len(allsampleonetask):
                        thisres = random.sample(allsampleonetask, 2 * memnumforeveryclass)
                    else:
                        thisres = allsampleonetask
                    allchoosedsample = thisres
                    onetrainres = allchoosedsample[0:memnumforeveryclass]
                    onevalidres = allchoosedsample[memnumforeveryclass:]
                    fewtrainname = memdatafolder + "/" + newtaskfold[aa] + "/train_mem.txt"
                    fewvalidname = memdatafolder + "/" + newtaskfold[aa] + "/valid_mem.txt"

                    f = open(fewtrainname, 'w')
                    for one in onetrainres:
                        f.write(one + "\n")
                    f.close()

                    f = open(fewvalidname, 'w')
                    for one in onevalidres:
                        f.write(one + "\n")
                    f.close()

                del scalerpre
                logger.info("finish generate pseudo samples")

                newtrainfile = newfilefolder + "/" + thistaskfold + "/" + "train.txt"
                newvalidfile = newfilefolder + "/" + thistaskfold + "/" + "valid.txt"
                thistrainfilename = dataprefix + thistaskfold + "/" + str(onerun) + "_" + str(args.seed) + "/train.txt"
                thisvalidfilename = dataprefix + thistaskfold + "/" + str(onerun) + "_" + str(args.seed) + "/valid.txt"
                alltrainmem = []
                allvalidmem = []
                for aa in range(0,j):
                    onepremempath = memdatafolder + "/" + newtaskfold[aa] + "/"
                    onepretrain = onepremempath + "train_mem.txt"
                    oneprevalid = onepremempath + "valid_mem.txt"
                    alltrainmem.append(onepretrain)
                    allvalidmem.append(oneprevalid)
                f = open(newtrainfile, 'w')
                for aa in range(len(alltrainmem)):
                    oneprefile = alltrainmem[aa]
                    for line in open(oneprefile, 'r'):
                        f.write(str(aa) + "\t" + line)
                for line in open(thistrainfilename, 'r'):
                    f.write(str(j) + "\t" + line)
                f.close()
                f = open(newvalidfile, 'w')
                for aa in range(len(allvalidmem)):
                    oneprefile = allvalidmem[aa]
                    for line in open(oneprefile, 'r'):
                        f.write(str(aa) + "\t" + line)
                for line in open(thisvalidfilename, 'r'):
                    f.write(str(j) + "\t" + line)
                f.close()
                args.train_file_name = newtrainfile
                args.valid_file_name = newvalidfile
                args.test_file_name = newtestfile

            train_dataset = T5SummarizationDataset(args.train_file_name, args.max_length, tokenizer, newtgentasktokens, answertoken, j)
            valid_dataset = T5SummarizationDataset(args.valid_file_name, args.max_length, tokenizer, newtgentasktokens, answertoken, j)
            test_dataset = T5SummarizationDataset(args.test_file_name, args.max_length, tokenizer, newtgentasktokens, answertoken, j)

            logger.info("Finish prepare model and dataset")
            logger.info("Start training")

            train(args, model, train_dataset, valid_dataset, onerun)
            logger.info("Finish training")

            if args.local_rank in [0, -1]:
                logger.info("Start testing")
                logger.info("Testing...")
                test(args, test_dataset, onerun)
                logger.info("Finish testing!")

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()








