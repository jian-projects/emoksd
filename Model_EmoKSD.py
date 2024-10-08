import torch, os, json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from transformers import logging
logging.set_verbosity_error()

from utils_model import *
from utils_rnn import DynamicLSTM
# config 中已经添加路径了
from data_loader import ERCDataset_Multi

max_seq_lens = {'meld': 127, 'iec': 511, 'emn': 127}

class ERCDataset_EmoKSD(ERCDataset_Multi):
    """
    每个utt, 拼接之前若干个utt(带mask表emo), 作为一个样本
    通过 mlm 预测mask位置的单词, 来学习情绪流
    通过 lstm 聚合历史utt的信息, 来学习上下文
    """

    def label_change(self):
        new_labels, new_labels_dict = json.load(open(self.data_dir + 'label_change', 'r')), {}
        for lab, n_lab in new_labels:
            new_labels_dict[lab] = n_lab
        self.tokenizer_['labels']['e2l'] = new_labels_dict

    def prompt_utterance(self, dataset):
        for stage, convs in dataset.items():
            for conv in convs:
                spks, txts = conv['speakers'], conv['texts']
                emos = [self.tokenizer_['labels']['e2l'][emo] if emo else 'none' for emo in conv['emotions']] # 转换 emo
                emos_lab_id = [self.tokenizer_['labels']['ltoi'][emo] if emo else -1 for emo in conv['emotions']] # 获取 emo_id
                emos_token_id = [self.tokenizer.encode(e)[1] for e in emos]
                assert len(emos_lab_id) == len(emos_token_id)
                prompts_t = [f"{spk}: {txt} {self.tokenizer.sep_token} {spk} expresses {emo} {self.tokenizer.sep_token}"
                           for spk, txt, emo in zip(spks, txts, emos)]
                prompts = [f"{spk}: {txt} {self.tokenizer.sep_token} {spk} expresses {self.tokenizer.mask_token} {self.tokenizer.sep_token}"
                           for spk, txt in zip(spks, txts)]

                embeddings_t = self.tokenizer(prompts_t, padding=True, add_special_tokens=False, return_tensors='pt')
                embeddings = self.tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt')
                
                conv['new_emos'] = emos
                conv['emos_lab_id'] = emos_lab_id
                conv['emos_token_id'] = emos_token_id
                conv['prompts'] = prompts
                conv['embeddings'] = embeddings
                conv['prompts_t'] = prompts_t
                conv['embeddings_t'] = embeddings_t

                # 记录相关信息
                self.info['num_conv_speaker'][stage].append(len(set(spks))) # conv speaker number
                self.info['num_conv_utt'][stage].append(len(spks)) # conv utterance number
                self.info['num_conv_utt_token'][stage].append(embeddings.attention_mask.sum(dim=1).tolist()) # conv utterance token number

            self.datas[stage] = convs

    def extend_sample(self, dataset, mode='online'):
        for stage, convs in dataset.items():
            samples = []
            for conv in tqdm(convs):
                conv_input_ids, conv_attention_mask = conv['embeddings'].input_ids, conv['embeddings'].attention_mask
                conv_input_ids_t, conv_attention_mask_t = conv['embeddings_t'].input_ids, conv['embeddings_t'].attention_mask
                for ui, (emo_lab_id, emo_token_id) in enumerate(zip(conv['emos_lab_id'], conv['emos_token_id'])):
                    ## 一前一后 交替拼接，标注当前位置
                    cur_mask = [1] # 定位当前 utterance 位置
                    emo_flow_token_ids, emo_flow_token_label = [emo_token_id], [emo_lab_id]
                    input_ids_ext = conv_input_ids[ui][0:conv_attention_mask[ui].sum()].tolist()[-self.max_seq_len:]
                    input_ids_ext_t = conv_input_ids_t[ui][0:conv_attention_mask_t[ui].sum()].tolist()[-self.max_seq_len:]
                    for i in range(1,len(conv['emotions'])):
                        if ui-i >=0:
                            tmp = conv_input_ids[ui-i][0:conv_attention_mask[ui-i].sum()].tolist()
                            if len(input_ids_ext) + len(tmp) <= self.max_seq_len:
                                cur_mask = [1] + cur_mask
                                input_ids_ext = tmp + input_ids_ext
                                input_ids_ext_t = conv_input_ids_t[ui-i][0:conv_attention_mask_t[ui-i].sum()].tolist() + input_ids_ext_t
                                emo_flow_token_ids = [conv['emos_token_id'][ui-i]] + emo_flow_token_ids
                                emo_flow_token_label = [conv['emos_lab_id'][ui-i]] + emo_flow_token_label
                            else: break
                    
                    input_ids_ext = torch.tensor([self.tokenizer.cls_token_id] + input_ids_ext) # 增加 cls token
                    input_ids_ext_t = torch.tensor([self.tokenizer.cls_token_id] + input_ids_ext_t)

                    label_category = self.tokenizer_['labels']['ltoi'][conv['emotions'][ui]] if conv['emotions'][ui] else -1
                    if label_category == -1: continue
                    sample = {
                        'index':    len(samples),
                        'text':     conv['texts'][ui],
                        'speaker':  conv['speakers'][ui],
                        'emotion':  conv['emotions'][ui],
                        'prompt':   conv['prompts'][ui],
                        'prompt_t': conv['prompts_t'][ui],
                        'input_ids':   input_ids_ext, 
                        'input_ids_t': input_ids_ext_t, 
                        'attention_mask':   torch.ones_like(input_ids_ext),
                        'attention_mask_t': torch.ones_like(input_ids_ext_t),
                        'label': label_category, 
                        'cur_mask': torch.tensor(cur_mask), 
                        'emo_flow_token_ids': torch.tensor(emo_flow_token_ids), 
                        'emo_flow_token_label': torch.tensor(emo_flow_token_label),
                    }
                    samples.append(sample)

                    # 统计一下信息
                    if conv['emotions'][ui] not in self.info['emotion_category']:
                        self.info['emotion_category'][conv['emotions'][ui]] = 0
                        self.info['emotion_category'][conv['emotions'][ui]] += 1

            # 记录相关信息
            self.info['num_samp'][stage] = len(samples)
            self.datas[stage] = samples

    def setup(self, tokenizer, max_seq_len=128):
        self.tokenizer, self.max_seq_len = tokenizer, max_seq_len
        self.label_change() # 需要改变label or 将label加进字典, 使tokenizer后只有1位数字
        self.prompt_utterance(self.datas) # 给 utterance 增加 prompt
        self.extend_sample(self.datas) # 扩充 utterance, 构建样本

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col or 'flow' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            elif 'ret' in col:
                if self.flow_num: inputs[col] = torch.stack([sample[col][self.flow_num] for sample in samples])
                else: inputs[col] = torch.stack([sample[col][self.flow_num+1] for sample in samples])
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs


def config_for_model(args, scale='base'):
    scale = args.model['scale'] if 'scale' in args.model else scale
    args.model['plm'] = args.file['plm_dir'] + f"roberta-{scale}"
    
    args.model['data'] = f"{args.file['cache_dir']}{args.model['name']}.{scale}"

    return args
             
def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    
    ## 2. 导入数据
    max_seq_len = args.model['max_seq_len'] if 'max_seq_len' in args.model else max_seq_lens[args.train['tasks'][-1]]
    data_path = args.model['data']
    if os.path.exists(data_path):
        dataset = torch.load(data_path)
    else:
        data_dir = f"{args.file['data_dir']}{args.train['tasks'][-1]}/"
        dataset = ERCDataset_EmoKSD(data_dir, args.train['batch_size'])
        tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])      
        dataset.setup(tokenizer, max_seq_len=max_seq_len)
        # torch.save(dataset, data_path)
    
    dataset.batch_cols = {
        'index': -1,
        'label': -1,
        'input_ids': dataset.tokenizer.pad_token_id, 
        'input_ids_t': dataset.tokenizer.pad_token_id, 
        'attention_mask': 0, 
        'attention_mask_t': 0, 
        # 'cur_mask': 0,
        # 'emo_flow_token_ids': -1,   # token id
        # 'emo_flow_token_label': -1, # token category
    }

    model = EmoKSD(
        args=args,
        dataset=dataset,
        plm=args.model['plm'],
    )
    return model, dataset


class EmoKSD(ModelForClassification):
    def __init__(self, args, dataset, plm=None):
        super().__init__() # 能继承 ModelForClassification 的属性
        self.args = args
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.mask_token_id = dataset.tokenizer.mask_token_id

        if args.model['use_adapter']:
            from utils_adapter import auto_load_adapter
            self.plm_model = auto_load_adapter(args, plm=plm if plm is not None else args.model['plm'])
        else: self.plm_model = AutoModel.from_pretrained(plm if plm is not None else args.model['plm'])
        self.plm_pooler = PoolerAll(self.plm_model.config)  
        self.hidden_size = self.plm_model.config.hidden_size

        if self.args.model['use_lora']:
            peft_config = LoraConfig(inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1)
            self.plm_model = get_peft_model(self.plm_model, peft_config)

        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.loss_ce = CrossEntropyLoss(ignore_index=-1)

        self.weight = args.model['weight']

    def encode(self, inputs, stage='train'):
        outputs = {'student': None, 'teacher': None }
        if self.args.model['use_lora']:
            encode_model = self.plm_model.base_model
        else: encode_model = self.plm_model
        
        # 1. encoding
        outputs['student'] = encode_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        if stage == 'train':
            with torch.no_grad():
                outputs['teacher'] = encode_model(
                    input_ids=inputs['input_ids_t'],
                    attention_mask=inputs['attention_mask_t'],
                    output_hidden_states=True,
                    return_dict=True
                )

        outputs['mask_token_bool'] = inputs['input_ids']==self.mask_token_id
        return outputs
    
        
    def forward(self, inputs, stage='train'):
        ## 1. encoding 
        outputs = self.encode(inputs, stage=stage)
        features = self.dropout(outputs['student'].pooler_output)
        logits = self.classifier(features)
        preds = torch.argmax(logits, dim=-1).cpu()
        loss = self.loss_ce(logits, inputs['label'])

        ## 2. constraints
        if stage=='train':
            ###################### Fine Grained ###############################
            mask_token_bool = outputs['mask_token_bool']
            mask_features = outputs['student'].last_hidden_state[mask_token_bool]
            emo_features = outputs['teacher'].last_hidden_state[mask_token_bool]
            loss_td = F.l1_loss(mask_features, emo_features)
            ###################################################################

            ###################### Coarse Grained ###############################
            mask_feature = outputs['student'].pooler_output
            emo_feature = outputs['teacher'].pooler_output
            loss_ud = F.l1_loss(mask_feature, emo_feature)
            ###################################################################

            # loss = (1-self.weight)*loss + self.weight*(loss_ud+loss_td)
            if 'weight_' in self.args.model:
                loss = loss + self.weight*(self.args.model['weight_'][0]*loss_ud+self.args.model['weight_'][1]*loss_td)
            else:
                loss = loss + self.weight*(loss_ud+loss_td)

        mask = inputs['label'] >= 0
        return {
            # 'fea': features,
            'loss':   loss if mask.sum() > 0 else torch.tensor(0.0).to(loss.device),
            'logits': logits,
            'preds':  preds[mask.cpu()],
            'labels': inputs['label'][mask],
        }
    
    def get_constraints(self, inputs, constraints=['ekd']):
        outputs, bz = {}, len(inputs['mask_token_ids'])
        mask_token_bool, mask_token_ids = inputs['mask_token_bool'], inputs['mask_token_ids']
        layer_mask_features = torch.stack(inputs['student_layer'])[self.layers][:, mask_token_bool]
        layer_emo_features = torch.stack(inputs['teacher_layer'])[self.layers][:, mask_token_bool]
        
        # 1.0 teacher emo 蒸馏到 student mask 上
        if 'ekd' in constraints:  
            loss_ekd = [F.l1_loss(m_fea, e_fea) for m_fea, e_fea in zip(layer_mask_features, layer_emo_features)]
            outputs['loss_ekd'] = sum([r*l for r,l in zip(self.l_rate, loss_ekd)])
        
        # 2.0 构建原型, 执行对比学习
        if 'eka' in constraints:
            emo_token_ids = mask_token_ids[mask_token_ids >= 0] # emo token ids for each layer
            existing_category = e_c = emo_token_ids.unique()
            layer_proto = torch.stack([layer_emo_features[:,emo_token_ids==c].mean(dim=1) for c in e_c]).transpose(0,1)

            e2p_label = torch.stack([e_c==l for l in emo_token_ids])
            loss_eka = [proto_cl(e, p, e2p_label) for e, p in zip(layer_mask_features, layer_proto)]
            outputs['loss_eka'] = sum([r*l for r,l in zip(self.l_rate, loss_eka)])

        # 3.0 稳固情绪流, 增强分类器稳定性
        if 'ekp' in constraints:
            mask_token_label_pad = inputs['mask_token_label']
            mask_token_label = mask_token_label_pad[mask_token_ids>=0] # 展开所有sample中存在的 mask token 的标签 
            assert len(mask_token_label) == layer_emo_features.shape[1]
            layer_logits = [torch.cat([self.classifier(self.dropout(c)) for c in torch.split(e, bz)]) for e in layer_emo_features]
            loss_ekp = [self.loss_ce(l, mask_token_label) for l in layer_logits]
            outputs['loss_ekp'] = sum([r*l for r,l in zip(self.l_rate, loss_ekp)])

        return outputs
    

def proto_cl(embedding, proto, label, temp=1.0):
    """
    embeddings: 目标向量
    protos：原型向量
    labels：目标相对于原型向量的label, 即目标向量与哪个原型相似
    """
    cosine_sim = F.cosine_similarity(embedding.unsqueeze(1), proto.unsqueeze(0), dim=-1) / temp
    cosine_sim_exp = torch.exp(cosine_sim)

    # 每个 embedding, 最接近其所属原型
    loss = -torch.cat([torch.log(sim[lab]/sum(sim)) for sim, lab in zip(cosine_sim_exp, label)]).mean()
    
    return loss