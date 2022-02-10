import os
import pdb
import sys
import torch
import torch.nn as nn
from torch.nn.functional import kl_div
from torch.nn import Softmax

class T5forSummarization(nn.Module):
    def __init__(self, args, model, tokenizer):
        super(T5forSummarization, self).__init__()
        self.args = args
        self.model = model
        ### load ckpt
        if args.use_lm_adapted == 1:
            print("use lm adapted model!")
            t5ckpt = torch.load(args.lm_adapted_path)
            if args.ifckpt_onlymodel == 1:
                self.model.load_state_dict(t5ckpt)
            else:
                self.model.load_state_dict(t5ckpt['t5-large-prefixlm'])
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        self.tokenizer = tokenizer
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.promptnumber = 0
        self.promptembedding = None
        self.softmax = Softmax(dim=2)

    def set_prompt_embedding(self,promptnumber,promptembedding):
        self.promptnumber = promptnumber
        self.promptembedding = nn.parameter.Parameter(promptembedding)
        self.promptembedding_kd = nn.parameter.Parameter(promptembedding)
        self.promptembedding_kd.requires_grad = False

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((attention_mask.shape[0],self.promptnumber),1).to(self.args.device)
        all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)
        return self.model(
            inputs_embeds=allembedding,
            attention_mask=all_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True
        )

    def _step_pre(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        prompt_embed_repeat = self.promptembedding_kd.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((attention_mask.shape[0],self.promptnumber),1).to(self.args.device)
        all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)
        return self.model(
            inputs_embeds=allembedding,
            attention_mask=all_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True
        )

    def forward(self, batch, ifcalpre):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )
        kdloss = torch.tensor(0.0)
        if ifcalpre:
            ifinmem = batch["ifmem"]
            newindex = []
            for i in range(0, ifinmem.shape[0]):
                if ifinmem[i] == 0:
                    newindex.append(i)
            if newindex != []:
                outputs_pre = self._step_pre(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=lm_labels,
                    decoder_attention_mask=batch['target_mask']
                )
                thisdistri = outputs[1]
                predistri = outputs_pre[1]
                thisdistriuse = thisdistri[newindex]
                predistriuse = predistri[newindex]
                thisdistriuse_sm = self.softmax(thisdistriuse)
                predistriuse_sm = self.softmax(predistriuse)
                kdloss = kl_div(thisdistriuse_sm.log(), predistriuse_sm, reduction='sum')
                if torch.isinf(kdloss):
                    kdloss = torch.tensor(0.0)

        loss = outputs[0]
        if not ifcalpre:
            return loss
        else:
            return loss,kdloss

    def _generative_step(self, batch):
        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((batch["attention_mask"].shape[0], self.promptnumber), 1).to(self.args.device)
        all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            max_length=128,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])
        return input,target,preds

    def _generative_samples(self, batch):
        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((batch["attention_mask"].shape[0], self.promptnumber), 1).to(self.args.device)
        all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )

        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            max_length=self.args.max_length,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=True,
            top_k = 64,
            num_return_sequences=3
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])
        return input,target,preds


    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
