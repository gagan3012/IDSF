from accelerate.utils import set_seed
from accelerate import Accelerator
import pandas as pd
import numpy as np
import os
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm.rich import tqdm
import evaluate

def read_file(file):
    with open(file, 'r') as f:
      lines = f.readlines()
      data = []
      for i in range(len(lines)):
        if not lines[i].startswith('#'):
          data.append(lines[i].strip('\n'))

    df = pd.DataFrame([x.split('\t') for x in data], columns=['index','tokens','intents','ner_tags'])

    df_list = np.split(df, df[df.eq(None).all(1)].index) 
    df = df.fillna(value=np.nan)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df_list = np.split(df, df[df.isnull().all(1)].index) 

    print(len(df_list))
    tokens = []
    intents = []
    ner_tags = []

    for df1 in tqdm(df_list):
      df1 = df1.set_index("index").dropna()
      # print(df, '\n')
      token = df1['tokens'].values.tolist()
      tokens.append(' '.join(token))
      ner_tag = df1['ner_tags'].values.tolist()
      ner_tags.append(' '.join(ner_tag))
      intent= df1['intents'].unique().tolist()
      intents.append(' '.join(intent))

    df2 = pd.DataFrame(list(zip(tokens,ner_tags,intents)),
               columns =['text', 'slots', 'intents'])
    
    df2['text'].replace('', np.nan, inplace=True)
    df2.dropna(subset=['text'], inplace=True)
    # df2['tokens'] = df2.tokens.apply(lambda x: x.split(' '))
    # df2['ner_tags'] = df2.ner_tags.apply(lambda x: x.split(' '))
    dataset = Dataset.from_pandas(df2)
    return dataset 

def data_builder(split):
    data= DatasetDict({
        "train":read_file(f"../sid4lr/xSID-0.4/{split}.train.conll"),
        "validation": read_file(f"../sid4lr/xSID-0.4/{split}.valid.conll"),
        "test": read_file(f"../sid4lr/xSID-0.4/{split}.test.conll"),
        "test_nap": read_file(f"../sid4lr/tgt_data/nap.valid.conll"),
        "test_de-st": read_file(f"../sid4lr/tgt_data/de-st.valid.conll"),
        "test_gsw": read_file(f"../sid4lr/tgt_data/gsw.valid.conll"),
    })

    return data

device = "cuda"
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

checkpoint_name = "financial_sentiment_analysis_lora_v1.pt"
text_column = "text"
label_column = "intents"
max_length = 128
lr = 1e-4
num_epochs = 3
batch_size = 16

dataset = data_builder("en")
# seqeval = evaluate.load('seqeval.py')

def compute_metric(pred,ref):
    predictions, references = [], []
    for i in range(len(pred)):
        predictions.append(pred[i].split())
        references.append(ref[i].split())
    return seqeval.compute(predictions, references)
    

# dataset = dataset.train_test_split(test_size=0.1)
# dataset["validation"] = dataset["test"]
# del dataset["test"]

# classes = dataset["train"].features["label"].names


def add_prefix_intent(example):
    example["text_intent"] = 'Intent: ' + example["text"]
    return example


def add_prefix_slots(example):
    example["text_slot"] = 'Slots: ' + example["text"]
    return example


dataset = dataset.map(
    add_prefix_intent,
    num_proc=1,
    desc="Adding prefix to intent",
)

dataset = dataset.map(
    add_prefix_slots,
    num_proc=1,
    desc="Adding prefix to slots",
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def preprocess_function(examples):
    #     print(examples)
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(
        inputs, max_length=max_length, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=10,
                       padding="max_length", truncation=True)
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    desc="Running tokenizer on dataset",
)

# training and evaluation


def train():
    set_seed(42)

    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                             inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
    accelerator = Accelerator(mixed_precision="fp16")
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    device = accelerator.device
    model = model.to(device)

    model, optimizer, train_dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, eval_dataloader
        )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(
                    outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(train_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(eval_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(
            f"{epoch}: {train_ppl} {train_epoch_loss} {eval_ppl} {eval_epoch_loss}")
