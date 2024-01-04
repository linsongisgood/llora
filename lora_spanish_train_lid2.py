# coding=gbk
import os
import re
import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import WEIGHTS_NAME, CONFIG_NAME
from models import *


sentences1 = []
sentences1_tmp = []
sentences_h = []
sentences = []
sentences11 = []
tags = []
tags_t1 = []
tags_t2 = []
tags1 = []
labels = []
with open("./semeval2020/spanglish/1/train.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip('\n')
        a = re.split(r"\s+", line, maxsplit=1)
        x = a[0]
        labels.append(int(x))

with open("./semeval2020/spanglish/1/train.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip('\n')
        a = re.split(r"\s+", line, maxsplit=1)
        y = a[1]
        sentences1_tmp.append(y)
with open("./semeval2020/spanglish/3/train_span.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        a = re.split(r"\s+", line, maxsplit=1)
        y = a[1]
        sentences_h.append(y)
for i, j in zip(sentences1_tmp, sentences_h):
    tmp = i + '. ' + j

    sentences1.append(tmp)

sentences21 = []
sentences2 = []
sentences2_t1 = []
sentences2_t2 = []
sentences22 = []
tags2 = []
tags2_t1 = []
tags2_t2 = []
tags22 = []
labels2 = []
with open("./semeval2020/spanglish/1/dev.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip('\n')
        a = re.split(r"\s+", line, maxsplit=1)
        x = a[0]
        labels2.append(int(x))

with open("./semeval2020/spanglish/1/dev.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip('\n')
        a = re.split(r"\s+", line, maxsplit=1)
        x = a[1]
        sentences2_t1.append(x)
with open("./semeval2020/spanglish/3/dev_span.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        a = re.split(r"\s+", line, maxsplit=1)
        y = a[1]
        sentences2_t2.append(y)
for i, j in zip(sentences2_t1, sentences2_t2):
    tmp = i + '. ' + j

    sentences21.append(tmp)


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../../../root/autodl-tmp/xlm-roberta-large")

input_ids = []
attention_masks = []
input_ids11 = []
attention_masks11 = []
input_ids2 = []
attention_masks2 = []
input_ids22 = []
attention_masks22 = []

for sent in sentences1:
    encoded_dict = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=120,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # �ѱ���ľ��Ӽ���list.
    input_ids.append(encoded_dict['input_ids'])
    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])
from language_tag import lang_tag
input_ids11 = torch.tensor(lang_tag(sentences1))

for sent in sentences21:
    encoded_dict2 = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=120,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # �ѱ���ľ��Ӽ���list.
    input_ids2.append(encoded_dict2['input_ids'])
    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks2.append(encoded_dict2['attention_mask'])
input_ids22 = torch.tensor(lang_tag(sentences21))
# ��lists תΪ tensors.

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

input_ids2 = torch.cat(input_ids2, dim=0)
attention_masks2 = torch.cat(attention_masks2, dim=0)
labels2 = torch.tensor(labels2)

from torch.utils.data import TensorDataset
# ��input ���� TensorDataset��
dataset = TensorDataset( input_ids, attention_masks,input_ids11,labels)
val_dataset = TensorDataset(input_ids2, attention_masks2,input_ids22, labels2)
from torch.utils.data import DataLoader, RandomSampler

batch_size = 32
train_dataloader = DataLoader(
    dataset,  # ѵ������.
    sampler=RandomSampler(dataset),  # ����˳��
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    val_dataset,  # ��֤����.
    # sampler = RandomSampler(val_dataset), # ����˳��
    batch_size=batch_size
)
from transformers import AdamW, AutoConfig
config = AutoConfig.from_pretrained("../../../root/autodl-tmp/xlm-roberta-large", trust_remote_code=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = XLMRobertaForSequenceClassification.from_pretrained("../../../root/autodl-tmp/xlm-roberta-large",
                          return_dict=True,
                          num_labels=3,
                          output_hidden_states=True )

peft_config = PLoraConfig(task_type="SEQ_CLS",
                          inference_mode=False,
                          r=512,
                          lora_alpha=1024,
                          target_modules= ['query',"key"],
                          #modules_to_save = ['query','value',"key"],
                          lora_dropout=0.1,
                          num_virtual_users=3,
                          user_token_dim=1024)
model = get_peft_model(model, peft_config)
def set_non_lora_parameters_trainable(model):
    for name, param in model.named_parameters():
        if 'lora' not in name:  # �ų� Lora ��
            param.requires_grad = True
set_non_lora_parameters_trainable(model)
for name, param in model.named_parameters():
    print(name, param.requires_grad)

model.to('cuda:0')
#peft_config = LoraConfig(task_type="SEQ_CLS", target_modules=["query_key_value", "dense_h_to_4h", "dense_4h_to_h", "dense"], inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1)
#peft_config = LoraConfig(task_type="SEQ_CLS", target_modules=['query', 'value',"dense"], inference_mode=False, r=128,lora_alpha=256,lora_dropout=0.1)

print(model)
model.print_trainable_parameters()
# AdamW ��һ�� huggingface library ���࣬'W' ��'Weight Decay fix"����˼��
optimizer = AdamW(model.parameters(),
                  lr=1e-5,  # args.learning_rate - Ĭ���� 5e-5
                  eps=1e-8  # args.adam_epsilon  - Ĭ���� 1e-8�� ��Ϊ�˷�ֹ˥���ʷ�ĸ����0
                  )

from transformers import get_linear_schedule_with_warmup

# bert �Ƽ� epochs ��2��4֮��Ϊ�á�
epochs = 10

# training steps ������: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# ��� learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=50,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


def flat_accuracy(preds,labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # ���� hh:mm:ss ��ʽ��ʱ��
    return str(datetime.timedelta(seconds=elapsed_rounded))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


output_dir = "./binary"
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

# �����������.
'''seed_val = 62
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)'''

# ��¼training ,validation loss ,validation accuracy and timings.
training_stats = []

# ������ʱ��.
total_t0 = time.time()
best_val_accuracy = 0

for epoch_i in range(0, epochs):
    print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

    # ��¼ÿ�� epoch ���õ�ʱ��
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        # ÿ��40��batch ���һ������ʱ��.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_input_ids1 = batch[2].to(device)
        b_labels = batch[3].to(device)
        # ����ݶ�
        model.zero_grad()
        # forward
        outputs = model(input_ids = b_input_ids,attention_mask = b_input_mask,
                        p = b_input_ids1,
                        labels = b_labels
                             )
        loss, logits = outputs[:2]
        total_train_loss += loss.item()
        # backward ���� gradients.
        loss.backward()
        # ��ȥ����1 ���ݶȣ�������Ϊ 1.0, �Է��ݶȱ�ը.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # ����ģ�Ͳ���
        optimizer.step()
        # ���� learning rate.
        scheduler.step()
        logit = logits.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()
        # ����training ���ӵ�׼ȷ��.
        total_train_accuracy += flat_accuracy(logit, label_id)

        # ����batches��ƽ����ʧ.
    avg_train_loss = total_train_loss / len(train_dataloader)
    # ����ѵ��ʱ��.
    training_time = format_time(time.time() - t0)

    # ѵ������׼ȷ��.
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    #print("  ѵ��׼ȷ��: {0:.2f}".format(avg_train_accuracy))
    print("  ƽ��ѵ����ʧ loss: {0:.2f}".format(avg_train_loss))
    print("  ѵ��ʱ��: {:}".format(training_time))


    t0 = time.time()
    # ���� model Ϊvaluation ״̬����valuation״̬ dropout layers ��dropout rate�᲻ͬ
    model.eval()
    # ���ò���
    labels_pred1 = []
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_input_ids1 = batch[2].to(device)
        b_labels = batch[3].to(device)

        # ��valuation ״̬��������Ȩֵ�����ı����ͼ
        with torch.no_grad():
             outputs2 = model(input_ids = b_input_ids,attention_mask = b_input_mask,
                              p = b_input_ids1,
                              labels = b_labels
                             )
        loss2, logits2 = outputs2[:2]
        # ���� validation loss.
        total_eval_loss += loss2.item()
       # b = torch.tensor([0.1, 0.8, 0.1]).cuda()
        #logits = torch.mul(logits2, b)
        logit = logits2.detach().cpu().numpy()

        label_id = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logit, axis=1).flatten()
        labels_pred1 = np.append(labels_pred1, pred_flat)
        # ���� validation ���ӵ�׼ȷ��.
        total_eval_accuracy += flat_accuracy(logit, label_id)

    # ���� validation ��׼ȷ��.
    print(classification_report(labels2, labels_pred1, digits=4))
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("")
    print("  ����׼ȷ��: {0:.2f}".format(avg_val_accuracy))

    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), output_model_file)

    # ����batches��ƽ����ʧ.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # ����validation ʱ��.
    validation_time = format_time(time.time() - t0)

    print("  ƽ��������ʧ Loss: {0:.2f}".format(avg_val_loss))
    print("  ����ʱ��: {:}".format(validation_time))


print("ѵ��һ������ {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

