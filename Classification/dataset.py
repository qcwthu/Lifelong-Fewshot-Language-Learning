import sys
sys.path.append("../..")

import torch
from torch.utils.data import Dataset

class T5SenClassifyDatasetConll(Dataset):
    def __init__(self, filename, maxlen, tokenizer, newtgentasktokens, answertoken, currenttasknumber):
        super(T5SenClassifyDatasetConll, self).__init__()
        self.filename = filename
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.data = []
        self.currenttasknumber = currenttasknumber
        self.gentasktoken = newtgentasktokens
        self.answertoken = answertoken
        self.data,self.lmdata,self.ifmem = self.getalldata(self.filename)
        self.num_entries = len(self.data)

    def getalldata(self,filename):
        f = open(filename,'r')
        alldata = []
        alllmdata = []
        ifmem = []
        while True:
            oneline = f.readline().strip()
            if not oneline:
                break
            linelist = oneline.split("\t")
            if len(linelist) != 3:
                print(oneline)
                print(linelist)
            typeindex = int(linelist[0])
            if typeindex < self.currenttasknumber:
                ifmem.append(0)
            else:
                ifmem.append(1)
            onedata = []
            onedata.append(linelist[1])
            onedata.append(linelist[2])
            alldata.append(onedata)
            onelmdata = []
            onelmdata.append(self.gentasktoken[typeindex])
            onelmdata.append(linelist[1] + " " + self.answertoken + " " + linelist[2])
            alllmdata.append(onelmdata)
        f.close()
        return alldata,alllmdata,ifmem

    def __getitem__(self, idx):
        inputdata = self.data[idx][0]
        targetdata = self.data[idx][1]
        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")

        inputlmdata = self.lmdata[idx][0]
        targetlmdata = self.lmdata[idx][1]

        inputlmres = self.tokenizer.batch_encode_plus([inputlmdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        targetlmres = self.tokenizer.batch_encode_plus([targetlmdata], padding=False, max_length=self.maxlen * 2, truncation=True, return_tensors="pt")

        thisifmem = self.ifmem[idx]
        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze(), inputlmres["input_ids"].squeeze(), targetlmres["input_ids"].squeeze(), thisifmem

    def __len__(self):
        return self.num_entries


class SmartBatchingCollate:
    def __init__(self, max_length, pad_token_id):
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):

        sequences, targets, lmseq, lmtar, thisifmem = list(zip(*batch))

        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        target_ids, target_mask = self.pad_target(targets, max_sequence_length=self._max_length, pad_token_id=self._pad_token_id)

        lminput_id, lm_att_mask = self.pad_sequence(lmseq, max_sequence_length=self._max_length, pad_token_id=self._pad_token_id)
        lmtar_id, lm_tar_mask = self.pad_target(lmtar, max_sequence_length=self._max_length * 2, pad_token_id=self._pad_token_id)

        ifinmem = torch.tensor(thisifmem)
        output = input_ids, attention_mask, target_ids, target_mask, lminput_id, lm_att_mask, lmtar_id, lm_tar_mask, ifinmem

        return output

    def pad_target(self, sequence_batch, max_sequence_length, pad_token_id):

        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            new_sequence = list(sequence[:max_len])
            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)
            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)
            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)
        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences,attention_masks


    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            new_sequence = list(sequence[:max_len])

            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences, attention_masks
