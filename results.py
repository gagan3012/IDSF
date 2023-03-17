import os
import pandas as pd
import json
import gspread
from gspread_dataframe import set_with_dataframe
from gspread_formatting import *

def get_results(split):
    models = ["mt0-small", "mt0-base", "mt0-large"]
    subsets = ["eval","test","DE_ST","GSW","NAP"]
    training_data = [split]
    training_type = ["intents"]

    df_list = []
    df = pd.DataFrame(columns=subsets, index=models)
    for model in models:
        for data in training_data:
            for subset in subsets:
                for t_type in training_type:
                    path = f"../results/{model}_{data}_{t_type}/results_{subset.lower()}.json"
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            results = json.load(f)
                            if results.get("test_accuracy"):
                                df.loc[model, subset] = results["test_accuracy"]
                            elif results.get("eval_accuracy"):
                                df.loc[model, subset] = results["eval_accuracy"]
                    elif os.path.exists(f"../results/{model}_{data}_{t_type}/{subset.lower()}_results.json"):
                        with open(f"../results/{model}_{data}_{t_type}/{subset.lower()}_results.json", "r") as f:
                            results = json.load(f)
                            df.loc[model, subset] = results["eval_accuracy"]
                    elif os.path.exists(f"../results/{model}_{data}_{t_type}/results_test_{subset.lower()}.json"):
                        with open(f"../results/{model}_{data}_{t_type}/results_test_{subset.lower()}.json", "r") as f:
                            results = json.load(f)
                            df.loc[model, subset] = results["test_accuracy"]

    # to_sheets(df,"Results MT0")
    df = df.reset_index(level=0)
    df = df.rename(columns={'index': f'{training_data[0]}'})
    df = df.rename_axis(f'index').reset_index()
    print(df)
    return df

def to_sheets(df_test, worksheet_title):
    # Open an existing spreadsheet
    gc = gspread.service_account()
    sh = gc.open_by_url(
        'https://docs.google.com/spreadsheets/d/1SFWNF-aNXKlLRhuTQVcBAXDfAuM0pEWCC2GabTaxWlk/edit#gid=0')

    # Read a worksheet and create it if it doesn't exist
    try:
        worksheet = sh.worksheet(worksheet_title)
    except gspread.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=worksheet_title, rows=100, cols=100)

    # Write a test DataFrame to the worksheet
    set_with_dataframe(worksheet, df_test)
    # format_sheet(worksheet)

if __name__ == "__main__":
    df0 = get_results("multi")
    df1 = get_results("en")
    df2 = get_results("de")
    df3 = get_results("it")
    df = pd.concat([df0, df1, df2, df3],ignore_index=True, axis=0)
    df['multi'].fillna(df['en'], inplace=True)
    df['multi'].fillna(df['de'], inplace=True)
    df['multi'].fillna(df['it'], inplace=True)
    df = df.drop(['en', 'de', 'it'], axis=1)
    print(df)
    to_sheets(df,"Results MT0")
