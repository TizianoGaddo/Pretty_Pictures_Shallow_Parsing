import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score
import re

DATA_DIR        = './results'        # path to the folder with the vlm answers
ANNOTATIONS_DIR = './annotations'    # path to the folder with the annotation csvs
OUTPUT_DIR      = './results'        # path to the folder in which to save the annotator's agreement results

metrics = ['ldd', 'wo']

# import the vlm answers (only for word order and ldd)
res_wo = pd.read_csv(f"{DATA_DIR}/res_wo.csv")
res_ldd = pd.read_csv(f"{DATA_DIR}/res_ldd.csv")
res_ldd['img_name']
res_wo = res_wo[['img_name','a1','a2','a3','ca3']]
res_ldd = res_ldd[['img_name','a1','a2','a3','a4','a5','a6']]

# clean the dataframes, convert answers to 1 for yes, 0 for no
res_wo[['a1','a2','a3','ca3']] = res_wo[['a1','a2','a3','ca3']].apply(
    lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x)
)
res_wo[['a1','a2','a3','ca3']] = res_wo[['a1','a2','a3','ca3']].apply(
    lambda col: col.map(
        lambda x: (
            1 if isinstance(x, str) and re.fullmatch(r"\W*yes\W*", x)
            else 0 if isinstance(x, str) and re.fullmatch(r"\W*no\W*", x)
            else x
        )
    )
)

res_ldd[['a1','a2','a3','a4','a5','a6']] = res_ldd[['a1','a2','a3','a4','a5','a6']].apply(
    lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x)
)
res_ldd[['a1','a2','a3','a4','a5','a6']] = res_ldd[['a1','a2','a3','a4','a5','a6']].apply(
    lambda col: col.map(
        lambda x: (
            1 if isinstance(x, str) and re.fullmatch(r"\W*yes\W*", x)
            else 0 if isinstance(x, str) and re.fullmatch(r"\W*no\W*", x)
            else x
        )
    )
)

# melt lo longform
res_wo = res_wo.melt(id_vars='img_name', var_name='answer', value_name='vlm')
res_ldd = res_ldd.melt(id_vars='img_name', var_name='answer', value_name='vlm')

# import the annotators csvs
for metric in metrics:
    ann_1 = pd.read_csv(f"{ANNOTATIONS_DIR}/{metric}_eval_1.csv")
    ann_2 = pd.read_csv(f"{ANNOTATIONS_DIR}/{metric}_eval_2.csv")
    ann_3 = pd.read_csv(f"{ANNOTATIONS_DIR}/{metric}_eval_3.csv")
    annotators = [ann_1, ann_2, ann_3]
    if metric == 'ldd':
        vlm_df = res_ldd.copy()
        value_vars_cols = ['a1','a2','a3','a4','a5','a6']
        prefix = 'LDD_'
    elif metric == 'wo':
        vlm_df = res_wo.copy()
        value_vars_cols = ['a1','a2','a3','ca3']
        prefix = 'WordOrder_'
    else:
        print('Error: metric')
        break
    for ann in annotators:
        ann["img_number"] = prefix + ann["img_number"].astype(str) + ".png"
    df = vlm_df[vlm_df['img_name'].isin(ann_1['img_number'])]
    for i, ann in enumerate(annotators):
        ann.rename(columns={'img_number': 'img_name'}, inplace=True)
        df_long = ann.melt(id_vars='img_name', value_vars=value_vars_cols, var_name='answer', value_name=f'ann_{i+1}') # melt to longform
        df = df.merge(
        df_long,
        on=["img_name", "answer"],
        how="inner"
        )
    cols = ['vlm', 'ann_1', 'ann_2', 'ann_3']
    df[cols] = df[cols].replace({"y": 1, "n": 0, "Y": 1, "N": 0})
    df = df.dropna(subset=cols)
    df[cols] = df[cols].astype(int)

    # Compute Fleiss' Kappa without VLM
    fleiss_table = []
    cols_ann = ["ann_1", "ann_2", "ann_3"]
    for _, row in df[cols_ann].iterrows():
        counts = row.value_counts()
        n_zeros = counts.get(0, 0)
        n_ones = counts.get(1, 0)
        fleiss_table.append([n_zeros, n_ones])
    fleiss_table = np.array(fleiss_table)
    kappa_without = fleiss_kappa(fleiss_table)

    # Compute Fleiss' Kappa with VLM
    fleiss_table = []
    for _, row in df[cols].iterrows():
        counts = row.value_counts()
        n_zeros = counts.get(0, 0)
        n_ones = counts.get(1, 0)
        fleiss_table.append([n_zeros, n_ones])
    fleiss_table = np.array(fleiss_table)
    kappa_with = fleiss_kappa(fleiss_table)
    
    with open(f'{OUTPUT_DIR}/annotator_agreement.txt', 'a') as f:
        f.write(f"Fleiss' Kappa ({metric}) without VLM: {kappa_without:.4f}\n")
        f.write(f"Fleiss' Kappa ({metric}) with VLM: {kappa_with:.4f}\n")

    # Compute Cohen's k between VLM and human majority
    df["human_majority"] = df[cols_ann].mean(axis=1).round().astype(int)
    kappa_majority = cohen_kappa_score(df["vlm"], df["human_majority"])

    with open(f'{OUTPUT_DIR}/annotators_agreement.txt', 'a') as f:
        f.write(f"Cohen's Kappa between VLM and human majority ({metric}): {kappa_majority:.4f}\n\n")