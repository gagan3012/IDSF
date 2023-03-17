from datasets import interleave_datasets
import gc
import json
import os
import sys
import threading
from glob import glob
import evaluate
import fire
import numpy as np
import pandas as pd
import psutil
import torch
from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from sidEval import getScores


def read_file(file):
    with open(file, 'r') as f:
      lines = f.readlines()
      data = []
      for i in range(len(lines)):
        if not lines[i].startswith('#'):
          data.append(lines[i].strip('\n'))

    df = pd.DataFrame([x.split('\t') for x in data], columns=[
                      'index', 'tokens', 'intents', 'ner_tags'])

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
      intent = df1['intents'].unique().tolist()
      intents.append(' '.join(intent))

    df2 = pd.DataFrame(list(zip(tokens, ner_tags, intents)),
                       columns=['text', 'slots', 'intents'])

    df2['text'].replace('', np.nan, inplace=True)
    df2.dropna(subset=['text'], inplace=True)
    # df2['tokens'] = df2.tokens.apply(lambda x: x.split(' '))
    # df2['ner_tags'] = df2.ner_tags.apply(lambda x: x.split(' '))
    dataset = Dataset.from_pandas(df2)
    return dataset


def data_builder(split):

    if os.path.exists(f"../sid4lr/xSID-0.4/hf_{split}"):
        data = load_from_disk(f"../sid4lr/xSID-0.4/hf_{split}")
        return data
    if split == "multi":
        train_data = glob("../sid4lr/xSID-0.4/*.projectedTrain.conll")+glob("./sid4lr/xSID-0.4/*.train.conll")
        dataset_train = concatenate_datasets([read_file(x) for x in train_data])
        valid_data = glob("../sid4lr/xSID-0.4/*.valid.conll")
        dataset_valid = concatenate_datasets([read_file(x) for x in valid_data])
        test_data = glob("../sid4lr/xSID-0.4/*.test.conll")
        dataset_test = concatenate_datasets([read_file(x) for x in test_data])

        data= DatasetDict({
            "train":dataset_train.shuffle(seed=42),
            "validation": dataset_valid.shuffle(seed=42),
            "test":dataset_test.shuffle(seed=42),
            "test_nap": read_file("../sid4lr/tgt_data/nap.valid.conll"),
            "test_de_st": read_file("../sid4lr/tgt_data/de-st.valid.conll"),
            "test_gsw": read_file("../sid4lr/tgt_data/gsw.valid.conll"),
        })
        data.save_to_disk(f"../sid4lr/xSID-0.4/hf_{split}")
    elif split == "setting5":
        dataset_train = Dataset.from_pandas(pd.read_csv(
            "../sid4lr/xSID-0.4/dataset_fixed/setting5.csv"))
        data= DatasetDict({
            "train":dataset_train.shuffle(seed=42),
            "validation": read_file(f"../sid4lr/xSID-0.4/de.valid.conll"),
            "test": read_file(f"../sid4lr/xSID-0.4/de.test.conll"),
            "test_nap": read_file("../sid4lr/tgt_data/nap.valid.conll"),
            "test_de_st": read_file("../sid4lr/tgt_data/de-st.valid.conll"),
            "test_gsw": read_file("../sid4lr/tgt_data/gsw.valid.conll"),
        })
        data.save_to_disk(f"../sid4lr/xSID-0.4/hf_{split}")
    elif split == "setting6":
        dataset_train = Dataset.from_pandas(pd.read_csv(
            "../sid4lr/xSID-0.4/dataset_fixed/setting6.csv"))
        data= DatasetDict({
            "train":dataset_train.shuffle(seed=42),
            "validation": read_file(f"../sid4lr/xSID-0.4/de.valid.conll"),
            "test": read_file(f"../sid4lr/xSID-0.4/de.test.conll"),
            "test_nap": read_file("../sid4lr/tgt_data/nap.valid.conll"),
            "test_de_st": read_file("../sid4lr/tgt_data/de-st.valid.conll"),
            "test_gsw": read_file("../sid4lr/tgt_data/gsw.valid.conll"),
        })
        data.save_to_disk(f"../sid4lr/xSID-0.4/hf_{split}")
    elif split == "setting7":
        dataset_train = Dataset.from_pandas(pd.read_csv(
            "../sid4lr/xSID-0.4/dataset_fixed/setting7.csv"))
        data= DatasetDict({
            "train":dataset_train.shuffle(seed=42),
            "validation": read_file(f"../sid4lr/xSID-0.4/it.valid.conll"),
            "test": read_file(f"../sid4lr/xSID-0.4/it.test.conll"),
            "test_nap": read_file("../sid4lr/tgt_data/nap.valid.conll"),
            "test_de_st": read_file("../sid4lr/tgt_data/de-st.valid.conll"),
            "test_gsw": read_file("../sid4lr/tgt_data/gsw.valid.conll"),
        })
        data.save_to_disk(f"../sid4lr/xSID-0.4/hf_{split}")
    elif split == "setting8":
        dataset_train = Dataset.from_pandas(pd.read_csv(
            "../sid4lr/xSID-0.4/dataset_fixed/setting8.csv"))
        data= DatasetDict({
            "train":dataset_train.shuffle(seed=42),
            "validation": read_file(f"../sid4lr/xSID-0.4/it.valid.conll"),
            "test": read_file(f"../sid4lr/xSID-0.4/it.test.conll"),
            "test_nap": read_file("../sid4lr/tgt_data/nap.valid.conll"),
            "test_de_st": read_file("../sid4lr/tgt_data/de-st.valid.conll"),
            "test_gsw": read_file("../sid4lr/tgt_data/gsw.valid.conll"),
        })
        data.save_to_disk(f"../sid4lr/xSID-0.4/hf_{split}")
    elif split == "setting9":
        dataset_train = Dataset.from_pandas(pd.read_csv(
            "../sid4lr/xSID-0.4/dataset_fixed/setting9.csv").dropna())
        data= DatasetDict({
            "train":dataset_train,
            "validation": read_file(f"../sid4lr/xSID-0.4/en.valid.conll"),
            "test": read_file(f"../sid4lr/xSID-0.4/en.test.conll"),
            "test_nap": read_file("../sid4lr/tgt_data/nap.valid.conll"),
            "test_de_st": read_file("../sid4lr/tgt_data/de-st.valid.conll"),
            "test_gsw": read_file("../sid4lr/tgt_data/gsw.valid.conll"),
        })
        data.save_to_disk(f"../sid4lr/xSID-0.4/hf_{split}_v2")
    else:
        if split in ['en']:
            data = DatasetDict({
            "train": read_file(f"../sid4lr/xSID-0.4/{split}.train.conll"),
            "validation": read_file(f"../sid4lr/xSID-0.4/{split}.valid.conll"),
            "test": read_file(f"../sid4lr/xSID-0.4/{split}.test.conll"),
            "test_nap": read_file("../sid4lr/tgt_data/nap.valid.conll"),
            "test_de_st": read_file("../sid4lr/tgt_data/de-st.valid.conll"),
            "test_gsw": read_file("../sid4lr/tgt_data/gsw.valid.conll"),
            })
        else:
            data = DatasetDict({
            "train": read_file(f"../sid4lr/xSID-0.4/{split}.projectedTrain.conll"),
            "validation": read_file(f"../sid4lr/xSID-0.4/{split}.valid.conll"),
            "test": read_file(f"../sid4lr/xSID-0.4/{split}.test.conll"),
            "test_nap": read_file("../sid4lr/tgt_data/nap.valid.conll"),
            "test_de_st": read_file("../sid4lr/tgt_data/de-st.valid.conll"),
            "test_gsw": read_file("../sid4lr/tgt_data/gsw.valid.conll"),
            })
        data.save_to_disk(f"../sid4lr/xSID-0.4/hf_{split}")
    

    return data

def levenshtein_distance(str1, str2):
    # TC: O(N^2)
    # SC: O(N^2)
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = np.empty((num_rows, num_cols))
    dp_matrix[0, :] = range(num_cols)
    dp_matrix[:, 0] = range(num_rows)

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[i, j] = dp_matrix[i - 1, j - 1]
            else:
                dp_matrix[i, j] = min(dp_matrix[i - 1, j - 1], dp_matrix[i - 1, j], dp_matrix[i, j - 1]) + 1

    return dp_matrix[num_rows - 1, num_cols - 1]


def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return classes[min_id]


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


def generate_label(input: str, target: str):
    inputs = input.replace(",", "").split(" ")
    # target = target.replace(" ",",")
    if str((target)).lower() == 'nan':
        bio_tagged = ['O']*len(inputs)
        return bio_tagged
    else:
        # print(target)
        try:
            target = target.replace(" ", ",")
        except AttributeError:
            target = target[0].replace(" ", ",")
        targets = [item.split(":", 1) for item in target.split(",")]
    bio_tagged = ['O']*len(inputs)
    prev_tag = "O"
    if len(targets) == 0:
      return bio_tagged
    targets = [x for x in targets if x != [''] and len(x) == 2]
    for tag, text in targets:
        try:
          index = inputs.index(text)
        except:
          continue
        if tag != "O" and prev_tag == "O":  # Begin NE
          bio_tagged[index] = "B-"+tag
          prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag:  # Inside NE
          bio_tagged[index] = "I-"+tag
          prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag:  # Adjacent NE
          bio_tagged[index] = "B-"+tag
          prev_tag = tag

    return " ".join(bio_tagged)


def generate_output(langauge, df_intents, model_id):
    for i in tqdm(range(len(df_intents))):
        df_intents['pred_slots'][i] = generate_label(
            df_intents['text'][i], df_intents['pred_slots'][i])
    string = ""
    for i, df in enumerate(np.array_split(df_intents, len(df_intents))):
        string += "# text: " + df_intents['text'][i] + "\n"
        string += "# intent: " + df_intents['pred_intents'][i] + "\n"
        df["text"] = df["text"].str.split(" ")
        df["pred_slots"] = df["pred_slots"].str.split(" ")
        df = df.apply(lambda x: x.explode() if x.name in [
                      'text', 'pred_slots'] else x)
        df = df[['text', 'pred_intents', 'pred_slots']]
        df.reset_index(inplace=True)
        dfAsString = df.to_csv(index=False, header=False, sep='\t')
        string += dfAsString + "\n"

    with open(f"/lustre07/scratch/gagan30/arocr/results/Test_{langauge}_{model_id}.out", "w") as text_file:
        text_file.write(string)

    return string


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

# seqeval = evaluate.load('seqeval.py')
# def compute_metric(pred,ref):
#     predictions, references = [], []
#     for i in range(len(pred)):
#         predictions.append(pred[i].split())
#         references.append(ref[i].split())
#     return seqeval.compute(predictions, references)

def main(model_name="mt0-xxl-mt", dataset_name="multi", label_column="intents", num_epochs=5, batch_size=64, do_test=True):
    accelerator = Accelerator()
    model_name_or_path = f"/lustre06/project/6005442/DataBank/IDSF/results/mt0-small_multi_intents"
    # dataset_name = "en"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = PeftConfig.from_pretrained(model_name_or_path)
    # peft_config = LoraConfig(
    #     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    # )
    text_column = "text"
    # label_column = "intents"
    lr = 3e-4
    # num_epochs = 5
    # batch_size = 4
    seed = 42
    # do_test = False
    set_seed(seed)
    max_length = 128    

    print(f"model_name: {model_name}")
    print(f"dataset_name: {dataset_name}")
    print(f"label_column: {label_column}")
    print(f"num_epochs: {num_epochs}")
    print(f"batch_size: {batch_size}")
    print(f"do_test: {do_test}")

    dataset = data_builder(dataset_name)

    if label_column == "intents":
        def add_prefix_intent(example):
            example["text"] = 'Intents: ' + example["text"]
            return example
        out_len=10
        dataset = dataset.map(
            add_prefix_intent,
            num_proc=1,
            desc="Adding prefix to intent",
        )

    if label_column == "slots":
        def add_prefix_slots(example):
            target = example['slots'].replace("B-","").replace("I-","").split()
            text = example['text'].split()
            test = "".join([m+":"+str(n)+" " for m,n in zip(target,text)])
            test_filter = filter(lambda x:x[0]!='O', test.split())
            example["slots"] = " ".join(test_filter).rstrip()
            example["text"] = 'Slots: ' + example["text"]
            return example
        out_len=115
        dataset = dataset.map(
            add_prefix_slots,
            num_proc=1,
            desc="Adding prefix to slots",
        )

    if label_column == "target":
        def add_prefix_intent(example):
            example["text"] = 'Intent: ' + example["text"]
            example['target'] = example['intents']
            return example
        out_len=109

        def add_prefix_slots(example):
            target = example['slots'].replace("B-", "").replace("I-", "").split()
            text = example['text'].split()
            test = "".join([m+":"+str(n)+" " for m, n in zip(target, text)])
            test_filter = filter(lambda x: x[0] != 'O', test.split())
            example["target"] = " ".join(test_filter).rstrip()
            example["text"] = 'Slots: ' + example["text"]
            return example


        dataset_slots = dataset.map(
            add_prefix_slots,
            num_proc=1,
            remove_columns=["slots", "intents"]
        )

        dataset_intents = dataset.map(
            add_prefix_intent,
            num_proc=1,
            remove_columns=['intents', 'slots']
        )

        dataset = DatasetDict()

        for (split1, x), (split2, y) in zip(dataset_slots.items(), dataset_intents.items()):
            dataset[split1] = interleave_datasets([x, y]).shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    def preprocess_function(examples):
        #     print(examples)
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(inputs, max_length=48, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=out_len, padding="max_length", truncation=True)
        labels = labels["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    f1 = evaluate.load("metrics/f1.py")

    def compute_metrics(test_pred, test_label):
        predictions = tokenizer(
            test_pred, truncation=True, max_length=20, padding="max_length").input_ids
        references = tokenizer(
              test_label, truncation=True, max_length=20, padding="max_length").input_ids
        f1_score = 0
        for i in range(len(predictions)):
            f1_score += f1.compute(
                predictions=predictions[i], references=references[i], average="weighted")['f1']
        return {"f1": f1_score / len(predictions)}


    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            # batched=True,
            num_proc=1,
            remove_columns=dataset["validation"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]
    test_nap_dataset = processed_datasets["test_nap"]
    test_de_st_dataset = processed_datasets["test_de_st"]
    test_gsw_dataset = processed_datasets["test_gsw"]

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    @find_executable_batch_size(starting_batch_size=batch_size)
    def inner_training_loop(batch_size):
        nonlocal accelerator # Ensure they can be used in our context
        accelerator.free_memory() # Free all lingering references
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, num_workers=4
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, num_workers=4)
        test_dataloader = DataLoader(
            test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, num_workers=4)
        test_nap_dataloader = DataLoader(
            test_nap_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, num_workers=4)
        test_de_st_dataloader = DataLoader(
            test_de_st_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, num_workers=4)
        test_gsw_dataloader = DataLoader(
            test_gsw_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, num_workers=4)

        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_name_or_path)
        model.print_trainable_parameters()

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # lr scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * num_epochs),
        )

        model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler, test_nap_dataloader, test_de_st_dataloader, test_gsw_dataloader  = accelerator.prepare(
            model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler, test_nap_dataloader, test_de_st_dataloader, test_gsw_dataloader
        )
        accelerator.print(model)
        os.makedirs(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}", exist_ok=True)

        is_ds_zero_3 = False
        if getattr(accelerator.state, "deepspeed_plugin", None):
            is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

        for epoch in range(num_epochs):
            with TorchTracemalloc() as tracemalloc:
                model.train()
                total_loss = 0
                for step, batch in enumerate(tqdm(train_dataloader)):
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
            accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
            accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
            accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
            accelerator.print(
                "GPU Total Peak Memory consumed during the train (max): {}".format(
                    tracemalloc.peaked + b2mb(tracemalloc.begin)
                )
            )

            accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
            accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
            accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
            accelerator.print(
                "CPU Total Peak Memory consumed during the train (max): {}".format(
                    tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
                )
            )
            train_epoch_loss = total_loss / len(eval_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

            model.eval()
            eval_preds = []
            with TorchTracemalloc() as tracemalloc:
                for _, batch in enumerate(tqdm(eval_dataloader)):
                    batch = {k: v for k, v in batch.items() if k != "labels"}
                    with torch.no_grad():
                        outputs = accelerator.unwrap_model(model).generate(
                            **batch, synced_gpus=is_ds_zero_3
                        )  # synced_gpus=True for DS-stage 3
                    preds = outputs.detach().cpu().numpy()
                    eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

            # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
            accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
            accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
            accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
            accelerator.print(
                "GPU Total Peak Memory consumed during the eval (max): {}".format(
                    tracemalloc.peaked + b2mb(tracemalloc.begin)
                )
            )

            accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
            accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
            accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
            accelerator.print(
                "CPU Total Peak Memory consumed during the eval (max): {}".format(
                    tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
                )
            )

            correct = 0
            total = 0
            for pred, true in zip(eval_preds, dataset["validation"][label_column]):
                if pred.strip() == true.strip():
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            results = {
                "epoch": epoch,
                "train_ppl": train_ppl.item(),
                "train_epoch_loss": train_epoch_loss.item(),
                "eval_accuracy":  accuracy,
                "eval_f1" : compute_metrics(eval_preds, dataset["validation"][label_column])['f1'],
            }
            accelerator.print(results)
            json.dump(results, open(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/eval_results.json", "w"))
            # accelerator.print(f"{eval_preds[:10]=}")
            # accelerator.print(f"{dataset['validation'][label_column][:10]=}")

            model.save_pretrained(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}")
            tokenizer.save_pretrained(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}")
            
            model.eval()
            test_preds = []
            for _, batch in enumerate(tqdm(test_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3
                    )  # synced_gpus=True for DS-stage 3
                test_preds.extend(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

            # test_preds_cleaned = []
            # for _, pred in enumerate(test_preds):
            #     test_preds_cleaned.append(get_closest_label(pred, classes))

            test_df = dataset["test"].to_pandas()
            test_df["text_labels_orig"] = test_preds
            # accelerator.print(seqeval.compute(test_df['text_labels_orig'], test_df[label_column]))
            correct = 0
            total = 0
            for pred, true in zip(test_preds, dataset["test"][label_column]):
                if pred.strip() == true.strip():
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            # accelerator.print(results)
            json.dump(results, open(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/results_test.json", "w"))
            accelerator.print(f"{accuracy=} on test set")
            # accelerator.print(f"{test_preds[:10]=}")
            # accelerator.print(f"{dataset['test'][label_column][:10]=}")

            pred_df = test_df

            os.makedirs(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}", exist_ok=True)
            pred_df.to_csv(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/predictions.csv", index=False)

            model.eval()
            test_preds = []
            for _, batch in enumerate(tqdm(test_gsw_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3
                    )  # synced_gpus=True for DS-stage 3
                test_preds.extend(tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True))

            # test_preds_cleaned = []
            # for _, pred in enumerate(test_preds):
            #     test_preds_cleaned.append(get_closest_label(pred, classes))

            test_df = dataset["test_gsw"].to_pandas()
            test_df["text_labels_orig"] = test_preds
            # accelerator.print(seqeval.compute(test_df['text_labels_orig'], test_df[label_column]))
            correct = 0
            total = 0
            for pred, true in zip(test_preds, dataset["test_gsw"][label_column]):
                if pred.strip() == true.strip():
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            results = {
                "test_f1": compute_metrics(test_preds, dataset["test_gsw"][label_column])["f1"],
                "test_accuracy":  accuracy,
                "results": getScores(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/")
            }
            # accelerator.print(results)
            json.dump(results, open(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/results_test_gsw.json", "w"))
            accelerator.print(f"{accuracy=} on GSW set")
            # accelerator.print(f"{test_preds[:10]=}")
            # accelerator.print(f"{dataset['test_gsw'][label_column][:10]=}")
            pred_df = test_df

            os.makedirs(
                f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}", exist_ok=True)
            pred_df.to_csv(
                f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/predictions_gsw.csv", index=False)

            model.eval()
            test_preds = []
            for _, batch in enumerate(tqdm(test_nap_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3
                    )  # synced_gpus=True for DS-stage 3
                test_preds.extend(tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True))

            # test_preds_cleaned = []
            # for _, pred in enumerate(test_preds):
            #     test_preds_cleaned.append(get_closest_label(pred, classes))

            test_df = dataset["test_nap"].to_pandas()
            test_df["text_labels_orig"] = test_preds
            # accelerator.print(seqeval.compute(test_df['text_labels_orig'], test_df[label_column]))
            correct = 0
            total = 0
            for pred, true in zip(test_preds, dataset["test_nap"][label_column]):
                if pred.strip() == true.strip():
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            results = {
                "test_f1": compute_metrics(test_preds, dataset["test_nap"][label_column])["f1"],
                "test_accuracy":  accuracy,
            }
            # accelerator.print(results)
            json.dump(results, open(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/results_test_nap.json", "w"))
            accelerator.print(f"{accuracy=} on NAP set")
            # accelerator.print(f"{test_preds[:10]=}")
            # accelerator.print(f"{dataset['test_gsw'][label_column][:10]=}")
            pred_df = test_df

            os.makedirs(
                f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}", exist_ok=True)
            pred_df.to_csv(
                f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/predictions_nap.csv", index=False)

            model.eval()
            test_preds = []
            for _, batch in enumerate(tqdm(test_de_st_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3
                    )  # synced_gpus=True for DS-stage 3
                test_preds.extend(tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True))

            # test_preds_cleaned = []
            # for _, pred in enumerate(test_preds):
            #     test_preds_cleaned.append(get_closest_label(pred, classes))

            test_df = dataset["test_de_st"].to_pandas()
            test_df["text_labels_orig"] = test_preds
            # accelerator.print(seqeval.compute(test_df['text_labels_orig'], test_df[label_column]))
            correct = 0
            total = 0
            for pred, true in zip(test_preds, dataset["test_de_st"][label_column]):
                if pred.strip() == true.strip():
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            results = {
                # "test_f1": compute_metrics(test_preds, dataset["test_de_st"][label_column])["f1"],
                "test_accuracy":  accuracy,
            }
            # accelerator.print(results)
            json.dump(results, open(f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/results_test_de_st.json", "w"))
            accelerator.print(f"{accuracy=} on DE-ST set")
            # accelerator.print(f"{test_preds[:10]=}")
            # accelerator.print(f"{dataset['test_gsw'][label_column][:10]=}")
            pred_df = test_df

            os.makedirs(
                f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}", exist_ok=True)
            pred_df.to_csv(
                f"/lustre07/scratch/gagan30/arocr/results/{model_name}_{dataset_name}_{label_column}/predictions_de_st.csv", index=False)

    inner_training_loop()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
