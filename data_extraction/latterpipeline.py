import pandas as pd
import numpy as np
import json
import re
import os
from data_extraction.utils import transform_row, group_company, fit_minmax, parse_filename


def latterpipeline(df):
    # Reading Files
    with open('linkedin_scrape.json', 'r') as file:
        linkedin_scrape = json.load(file)
    df_st = pd.read_csv('standard.csv')
    with open('sentiment_analysis_results.json', 'r') as file:
        sentiment = json.load(file)
    with open('esg_weights.json', 'r') as file:
        esg_weights = json.load(file)

    # Transform df
    df_t = df.apply(transform_row, axis=1)
    df_t['size'] = linkedin_scrape['company_size']
    df_t = df_t.apply(group_company, axis=1)

    quantitatives = list(df_t.columns[1:11]) + list(df_t.columns[12:27])
    qualitatives = list(df_t.columns[-10:-1])

    def normalize_qn1(row):
        signs = [-1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,0,0,-1,1,-1,-1,-1,-1,-1,1]
        cond = df_st[df_st['size'] == row['size']]
        for i,col in enumerate(quantitatives):
            if col not in ['Workforce Gender Ratios', 'Workforce Minority Ratios']:
                row[col] = float(row[col] - cond[col])*signs[i]
            else:
                row[col] = float(1 - (row[col] - cond[col])**2)
        return row
    def normalize_qn2(df):
        for col in quantitatives:
            tmp = df[col]
            tmp_max = np.max(df[col])
            tmp_min = np.min(df[col])
            df[col] = (2*(tmp - tmp_min) - tmp_max + tmp_min)/(tmp_max - tmp_min)
        return df

    def noramlize_quantitative(df):
        df_res = df.apply(normalize_qn1, axis=1)
        df_res = normalize_qn2(df_res)
        return df_res

    def restructure_ql(df, sentiment):
        df['filename'] = df_t['filename']
        for company, senti in sentiment.items():
            row = df[df['filename'] == company]
            for indi, values in senti.items():
                tmp = values['sentiment'].lower()
                if tmp == 'positive':
                    tmp = 1
                elif tmp == 'negative':
                    tmp = -1
                else:
                    tmp = 0
                row[indi] = tmp
            df[df['filename'] == company] = row
        df = df.drop(columns=['filename'], errors='ignore')
        return df

    df_quanti = noramlize_quantitative(df_t)
    df_quanti = df_quanti[['filename'] + quantitatives]
    df_quali = pd.DataFrame(np.nan, index=range(61), columns=qualitatives)
    df_quali = restructure_ql(df_quali, sentiment)
    df_norm = pd.concat([df_quanti, df_quali], axis=1).fillna(-0.2)
    df_norm['e_score'] = df_norm.iloc[:,[1,2,3,4,5,6,7,8,9,10,24,25,26,27,28,33,34]] @ esg_weights['E']['Coef'] + esg_weights['E']['Intercept']
    df_norm['s_score'] = df_norm.iloc[:,[11,12,13,14,15,16,17,29,30]] @ esg_weights['S']['Coef'] + esg_weights['S']['Intercept']
    df_norm['g_score'] = df_norm.iloc[:,[18,19,20,21,22,23,31,32]] @ esg_weights['G']['Coef'] + esg_weights['G']['Intercept']
    df_norm['esg_score'] = df_norm[['e_score', 's_score', 'g_score']] @ esg_weights['ESG']['Coef'] + esg_weights['ESG']['Intercept']
    df_scored = df_norm.apply(fit_minmax, axis=1)
    df_scored[['Industry', 'Company', 'Year']] = df_scored['filename'].apply(parse_filename)
    df_t[['Industry', 'Company', 'Year']] = df_t['filename'].apply(parse_filename)

    # Output File
    scored = '../data/final_output.csv'
    transformed = '../data/nonnormalized.csv'
    df_scored.to_csv(scored)
    df_t.to_csv(transformed)
