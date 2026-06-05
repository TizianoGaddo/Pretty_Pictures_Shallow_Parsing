import pandas as pd
import spacy
from collections import Counter
from scipy.stats import chi2_contingency
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

ANNOTATIONS_DIR = "./annotations"    # path to folder where annotations csvs are located
OUTPUT_DIR      = "./results"        # path to folder in which to save results of ambiguity analysis

df1 = pd.read_csv(f"{ANNOTATIONS_DIR}/amb_eval_1.csv")
df2 = pd.read_csv(f"{ANNOTATIONS_DIR}/amb_eval_2.csv")
df3 = pd.read_csv(f"{ANNOTATIONS_DIR}/amb_eval_3.csv")

ANNOTATION_COLUMN_NAME = 'meaning (1, 2, 0 for none, 3 for both)' # change to the name of the column containing the annotations in your csv

INDEXES = ['prompt', 'img_number'] # columns on which to merge the 3 dataframes

# Rename the annotations column to the code for the annotator
df1 = df1.rename(columns={ANNOTATION_COLUMN_NAME: 'ann1'})
df2 = df2.rename(columns={ANNOTATION_COLUMN_NAME: 'ann2'})
df3 = df3.rename(columns={ANNOTATION_COLUMN_NAME: 'ann3'})

# Merge the 3 annotators
df1_cols = INDEXES + ['meaning1', 'meaning2', 'ann1']

df = df1[df1_cols].merge(
        df2[INDEXES + ['ann2']], on=INDEXES
     ).merge(
        df3[INDEXES + ['ann3']], on=INDEXES
)

# load SpaCy parser
nlp = spacy.load("en_core_web_sm")

def get_spacy_prior(prompt):
    '''Returns the attachment preferred by SpaCy'''
    doc = nlp(prompt)
    for token in doc:
        if token.dep_ == "prep":
            if token.head.pos_ == "VERB":
                return 1  # 1 = VP (Meaning 1)
            elif token.head.pos_ in ["NOUN", "PROPN"]:
                return 2  # 2 = NP (Meaning 2)
    return -1

df['spacy_prior'] = df['prompt'].apply(get_spacy_prior)

# Majority voting for annotators scores 
def compute_gold_standard(row):
    votes = [row['ann1'], row['ann2'], row['ann3']]
    counts = Counter(votes)
    majority_vote, occurrecies = counts.most_common(1)[0]

    if occurrecies >= 2:
        return majority_vote
    else:
        return 3 # complete disagreement is treated as evidence for the presence of both meanings

df['Gold_Standard'] = df.apply(compute_gold_standard, axis=1)

# Annotators agreement (Fleiss Kappa)
fleiss_table = []
cols_ann = ["ann1", "ann2", "ann3"]
for _, row in df[cols_ann].iterrows():
    counts = row.value_counts()
    n_0 = counts.get(0, 0)
    n_1 = counts.get(1, 0)
    n_2 = counts.get(2, 0)
    n_3 = counts.get(3, 0)
    fleiss_table.append([n_0, n_1, n_2, n_3])
fleiss_table = np.array(fleiss_table)
kappa = fleiss_kappa(fleiss_table)
with open(f"{OUTPUT_DIR}/ambiguity_results.txt", "a") as f:
    f.write(f"ANNOTATORS AGREEMENT\\n\nFleiss Kappa: {kappa}\n\n\n\n")
# Flux images statistics
IMG_NUMBER = len(df)

failure_rate = (len(df[df['Gold_Standard'] == 0]) / IMG_NUMBER) * 100
chimera_rate = (len(df[df['Gold_Standard'] == 3]) / IMG_NUMBER) * 100
vp_rate = (len(df[df['Gold_Standard'] == 1]) / IMG_NUMBER) * 100
np_rate = (len(df[df['Gold_Standard'] == 2]) / IMG_NUMBER) * 100

with open(f"{OUTPUT_DIR}/ambiguity_results.txt", "a") as f:
    f.write('GENERAL AMBIGUITY IMAGES STATISTICS\n\n')
    f.write(f'Failure rate: {failure_rate}%\nChimeras rate: {chimera_rate}%\nVP (meaning 1) preference: {vp_rate}%\nNP (meaning 2) preference: {np_rate}%\n\n\n\n')

# FLUX vs SpaCy
# keep only the lines where flux produced images present a clear meaning preference and where spacy produced a valid result
df_valid = df[(df['Gold_Standard'].isin([1, 2])) & (df['spacy_prior'].isin([1, 2]))].copy()

with open(f"{OUTPUT_DIR}/ambiguity_results.txt", "a") as f:
    f.write(f"FLUX vs SpaCy ANALYSIS:\n\nValid images: {len(df_valid)}\n")

# Contingency table
contingency_tab = pd.crosstab(
    df_valid['spacy_prior'].replace({1: 'VP (1)', 2: 'NP (2)'}),
    df_valid['Gold_Standard'].replace({1: 'VP (1)', 2: 'NP (2)'}),
    rownames=['SpaCy'],
    colnames=['FLUX']
)

with open(f"{OUTPUT_DIR}/ambiguity_results.txt", "a") as f:
    f.write(f"Contingency Table:\n{contingency_tab}\n\n")

# Concordancy rate
conc = (df_valid['spacy_prior'] == df_valid['Gold_Standard']).sum()
conc_rate = (conc / len(df_valid)) * 100
with open(f"{OUTPUT_DIR}/ambiguity_results.txt", "a") as f:
    f.write(f"Concordancy Rate: {conc_rate:.1f}%\n")

# Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(contingency_tab)
with open(f"{OUTPUT_DIR}/ambiguity_results.txt", "a") as f:
        f.write(f"Chi-Square Test: p-value = {p_value:.4f}")