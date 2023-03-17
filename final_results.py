import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import torch
from datasets import Dataset, DatasetDict, load_from_disk
import os
from difflib import get_close_matches
import subprocess

tag_list = """B-alarm/alarm_modifier
B-album
B-artist
B-best_rating
B-condition_description
B-condition_temperature
B-cuisine
B-datetime
B-ecurring_datetime
B-entity_name
B-facility
B-genre
B-location
B-movie_name
B-movie_type
B-music_item
B-negation
B-news/type
B-object_location_type
B-object_name
B-object_part_of_series_type
B-object_select
B-object_type
B-party_size_description
B-party_size_number
B-playlist
B-rating_unit
B-rating_value
B-recurring_datetime
B-reference
B-reference-part
B-reminder/reminder_modifier
B-reminder/todo
B-restaurant_name
B-restaurant_type
B-served_dish
B-service
B-sort
B-timer/attributes
B-track
B-weather/attribute
B-weather/temperatureUnit
I-alarm/alarm_modifier
I-album
I-artist
I-best_rating
I-condition_description
I-condition_temperature
I-cuisine
I-datetime
I-ecurring_datetime
I-entity_name
I-facility
I-genre
I-location
I-movie_name
I-movie_type
I-music_item
I-negation
I-news/type
I-object_location_type
I-object_name
I-object_part_of_series_type
I-object_select
I-object_type
I-party_size_description
I-party_size_number
I-playlist
I-rating_unit
I-rating_value
I-recurring_datetime
I-reference
I-reminder/reminder_modifier
I-reminder/todo
I-restaurant_name
I-restaurant_type
I-served_dish
I-service
I-sort
I-timer/attributes
I-track
I-weather/attribute
I-weather/temperatureUnit
None
O
Orecurring_datetime""".replace("B-", "").replace("I-", "").replace("O", "").split("\n")

device = "cpu"

ipeft_model_id="../results/mt0-large_multi_intents"
itokenizer = AutoTokenizer.from_pretrained(ipeft_model_id)
iconfig = PeftConfig.from_pretrained(ipeft_model_id)
imodel = AutoModelForSeq2SeqLM.from_pretrained(iconfig.base_model_name_or_path, torch_dtype="auto", device_map="auto")
imodel = PeftModel.from_pretrained(imodel, ipeft_model_id)
imodel.eval()
imodel=imodel.to("cuda")

speft_model_id="../results/mt0-large_multi_slots"
stokenizer = AutoTokenizer.from_pretrained(speft_model_id)
sconfig = PeftConfig.from_pretrained(speft_model_id)
smodel = AutoModelForSeq2SeqLM.from_pretrained(
    sconfig.base_model_name_or_path, torch_dtype="auto", device_map="auto")
smodel = PeftModel.from_pretrained(smodel, speft_model_id)
smodel=smodel.to("cuda")

def get_intent(peft_model_id, text):
    input_ids = itokenizer("Intents: " + text, return_tensors='pt').to("cuda")
    imodel.eval()
    with torch.no_grad():
        outputs = imodel.generate(input_ids=input_ids["input_ids"], max_new_tokens=10)
        out= itokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    return out

def get_slots(peft_model_id, text):
    input_ids = stokenizer("Slots: " + text, return_tensors='pt').to("cuda")
    smodel.eval()
    with torch.no_grad():
        outputs = smodel.generate(input_ids=input_ids["input_ids"], max_new_tokens=115)
        out= stokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

    return out


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

def get_dataset():
    if os.path.exists(f"../sid4lr/xSID-0.4/hf_final-v3"):
        data = load_from_disk(f"../sid4lr/xSID-0.4/hf_final-v3")
    else:
        data= DatasetDict({
                "valid_nap": read_file("../sid4lr/tgt_data/nap.valid.conll"),
                "valid_de_st": read_file("../sid4lr/tgt_data/de-st.valid.conll"),
                "valid_gsw": read_file("../sid4lr/tgt_data/gsw.valid.conll"),
                "test_nap": read_file("../sid4lr/tgt_data/nap.test.conll-masked"),
                "test_de_st": read_file("../sid4lr/tgt_data/de-st.test.conll-masked"),
                "test_gsw": read_file("../sid4lr/tgt_data/gsw.test.conll-masked"),
            })
        data.save_to_disk(f"../sid4lr/xSID-0.4/hf_final-v3")
    return data

def get_predictions(dataset, peft_model_id,split):
    predictions_int = []
    intent_model = "../results/"+peft_model_id+"_multi_intents"
    for i in tqdm(range(len(dataset))):
        predictions_int.append(get_intent(intent_model, dataset[i]['text']))
    df = pd.DataFrame(predictions_int, columns=['pred_intents'])
    df['text'] = dataset['text']
    predictions_slot = []
    slot_model = "../results/"+peft_model_id+"_multi_slots"
    for i in tqdm(range(len(dataset))):
        predictions_slot.append(get_slots(slot_model, dataset[i]['text']))
    df['pred_slots'] = predictions_slot
    # df.to_csv('predictions.csv', index=False)
    generate_output(split,df,peft_model_id)
    return df
 
def find_sub_list(sl, l):
    results = []
    [results.append((match.start(), match.end())) for match in re.finditer(sl, l)]
    return results

def generate_label(input: str, target: str):
    inputs = input.replace(",","").split(" ")
    # target = target.replace(" ",",")
    if str((target)).lower() == 'nan':
        bio_tagged = ['O']*len(inputs)
        return bio_tagged
    else:
        # print(target)
        try:
            target = target.replace(" ",",")
        except AttributeError:
            target = target[0].replace(" ",",")
        targets = [item.split(":",1) for item in target.split(",")]
    bio_tagged = ['O']*len(inputs)
    prev_tag = "O"
    if len(targets)==0:
      return bio_tagged
    targets = [x for x in targets if x != [''] and len(x)==2]
    inputs = [x[1] for x in targets]
    for tag,text in targets:
        text = get_close_matches(text, inputs, n=1, cutoff=0.5)[0]
        index = inputs.index(text)
        print(text, index, tag)
        ## if first occurance of tag then add B- else I-
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged[index] ="B-"+tag
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged[index]="I-"+tag
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            bio_tagged[index]="B-"+tag
            prev_tag = tag
        
    return " ".join(bio_tagged)


def convert_to_slot_format(text,slots_gold,tag_list=tag_list):
    words = text.split()
    slots = ["O"]*len(words)
    slots_gold = slots_gold[0].replace(" ",",")
    targets=[item.split(":",1) for item in slots_gold.split(",")]
    targets = [x for x in targets if x != [''] and len(x) == 2]
    for i in range(len(targets)):
        sel_text = get_close_matches(targets[i][1], words, n=1, cutoff=0.5) # get the closest match
        if len(sel_text) == 0:
            sel_text = targets[i][1]
        else:
            sel_text = sel_text[0]
        if targets[i][0] in tag_list and sel_text in words:
            index = words.index(sel_text)
            if i == 0 or slots[index-1]=="O" or not slots[index-1].endswith(targets[i][0]):
                slots[index] = "B-" + targets[i][0]
            else:
                slots[index] = "I-" + targets[i][0]
    return " ".join(slots)

def generate_output(langauge, df_intents, model_id):
    for i in tqdm(range(len(df_intents))):
        df_intents['pred_slots'][i] = convert_to_slot_format(
            df_intents['text'][i], df_intents['pred_slots'][i])
    string=""
    for i, df in enumerate(np.array_split(df_intents, len(df_intents))):
        string += "# text: " + df_intents['text'][i] +"\n"
        string += "# intent: " + df_intents['pred_intents'][i] +"\n"
        df["text"]=df["text"].str.split(" ")
        df["pred_slots"]=df["pred_slots"].str.split(" ")
        df = df.apply(lambda x: x.explode() if x.name in ['text','pred_slots'] else x)
        df = df[['text','pred_intents','pred_slots']]
        df.reset_index(inplace=True)
        dfAsString = df.to_csv(index=False, header=False, sep='\t')
        string+= dfAsString + "\n"
    
    with open(f"../submission/UBC-DLNLP_{langauge}_{model_id}_v2.out", "w") as text_file:
        text_file.write(string)

    if langauge == "nap":
        getscore(f"../submission/UBC-DLNLP_{langauge}_{model_id}_v2.out", f"../sid4lr/tgt_data/{langauge}.valid.conll")
    elif langauge == "de_st":
        getscore(f"../submission/UBC-DLNLP_{langauge}_{model_id}_v2.out", f"../sid4lr/tgt_data/de-st.valid.conll")
    elif langauge == "gsw":
        getscore(f"../submission/UBC-DLNLP_{langauge}_{model_id}_v2.out", f"../sid4lr/tgt_data/{langauge}.valid.conll")

def getscore(pred_file, gold_file):
    print(pred_file)
    print(gold_file)

    # print("python3 ../sid4lr/sid4lr.py -p "+pred_file+" -g "+gold_file)
    process = subprocess.Popen(
        ["python3", "../sid4lr/scripts/sidEval.py",  pred_file, gold_file], stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    return output

if __name__ == "__main__":
    data = get_dataset()
    print(data["test_nap"][0])
    # get_predictions(data["test_nap"], "mt0-small", "nap")
    # get_predictions(data["test_de_st"], "mt0-small", "de_st")
    # get_predictions(data["test_gsw"], "mt0-small", "gsw")
    # get_predictions(data["valid_nap"], "mt0-large", "nap")
    # get_predictions(data["valid_de_st"], "mt0-large", "de_st")
    # get_predictions(data["valid_gsw"], "mt0-large", "gsw")
    # get_predictions(data["test_nap"], "mt0-base", "test_nap")
    # get_predictions(data["test_de_st"], "mt0-base", "test_de_st")
    # get_predictions(data["test_gsw"], "mt0-base", "-test_gsw")
    get_predictions(data["test_nap"], "mt0-large", "test_nap")
    get_predictions(data["test_de_st"], "mt0-large", "test_de_st")
    get_predictions(data["test_gsw"], "mt0-large", "test_gsw")