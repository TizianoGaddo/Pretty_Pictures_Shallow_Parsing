import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import bambi as bmb
import arviz as az


RESULTS_DIR = "./results"   # path to the folder with the results of the VLM question answering
OUTPUT_DIR  = "./results"   # path to the folder in which to save the csvs with the statistical results and the plots 

res_LDD = pd.read_csv(f'{RESULTS_DIR}/res_ldd.csv')
answer_cols = ['a1','a2','a3','a4','a5','a6']

# clean the data and convert yes to 1, no to 0
res_LDD[answer_cols] = res_LDD[answer_cols].apply(
    lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x)
)
res_LDD[answer_cols] = res_LDD[answer_cols].apply(
    lambda col: col.map(
        lambda x: (
            1 if isinstance(x, str) and re.fullmatch(r"\W*yes\W*", x)
            else 0 if isinstance(x, str) and re.fullmatch(r"\W*no\W*", x)
            else x
        )
    )
)

LDD_short = res_LDD.iloc[::3].copy()
LDD_medium = res_LDD.iloc[1::3].copy()
LDD_long = res_LDD.iloc[2::3].copy()

# Preparing data for binomial GLMM
LDD_short['length'] = 'Short'
LDD_medium['length'] = 'Medium'
LDD_long['length'] = 'Long'

tot_df = pd.concat([LDD_short, LDD_medium, LDD_long], ignore_index=True)
df_short = tot_df[['prompt_index','img_name','length','a1','a2','a3','a4']] # only interested in this answers (subject and verb of main and subordinate clauses)

# melt into long format
df_long = df_short.melt(
    id_vars=["prompt_index","img_name", "length"],
    value_vars=["a1","a2","a3","a4"],
    var_name="ans_number",
    value_name="accuracy"
)

mapping = {
    "a1": "Main",
    "a2": "Main",
    "a3": "Subordinate",
    "a4": "Subordinate"
}

length_order = ["Short", "Medium", "Long"]
df_long["length"] = pd.Categorical(
    df_long["length"], 
    categories=length_order, 
    ordered=True
)
df_long["sentence_type"] = df_long["ans_number"].map(mapping)
df_long['prompt_index'] = df_long['prompt_index'].astype('str')
df_long = df_long.drop('ans_number', axis=1)

# Bayesian GLMM
model = bmb.Model(
    "accuracy ~ length * sentence_type + (1|prompt_index) + (1|img_name)",
    df_long,
    family="bernoulli"
)

results = model.fit(chains=4, cores=4, draws=2000, tune=2000, target_accept=.9)

res = az.summary(results)
res.to_csv(f'{OUTPUT_DIR}/LDD_bGLMM.csv')

# pairwise comparisons
comparisons_sentence = bmb.interpret.comparisons(
    model=model,
    idata=results,
    contrast="sentence_type",
    conditional="length",
    prob=[0.89, 0.95]
)

comparisons_length = bmb.interpret.comparisons(
    model=model,
    idata=results,
    contrast="length",
    conditional="sentence_type",
    prob=[0.89, 0.95]
)

# conditionally adjusted predictions of accuracy
predictions = bmb.interpret.predictions(
    model,
    results,
    conditional=['length','sentence_type'],
    prob=[0.89, 0.95]
)

comparisons_sentence.summary.to_csv(f'{OUTPUT_DIR}/LDD_comp_stype.csv')
comparisons_length.summary.to_csv(f'{OUTPUT_DIR}/LDD_comp_len.csv')
predictions.summary.to_csv(f'{OUTPUT_DIR}/LDD_predict.csv')



# PLOTS

contrasts = [
    {"length": ["Short", "Medium"]},
    {"length": ["Short", "Long"]},
    {"length": ["Medium", "Long"]},
]

titles = ["A. Medium - Short", "B. Long - Short", "C. Long - Medium"]
fig_kwargs = {
    "theme": {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10     
    }
}

# Plot the pairwise comparisons in 4 subplots

fig, axes = plt.subplots(2, 2, sharey='all', figsize=(10, 8))
axes = axes.flatten()
for ax, c, t in zip(axes[:3], contrasts, titles):
    bmb.interpret.plot_comparisons(
        model,
        results,
        contrast=c,
        conditional="sentence_type",
        prob=[0.89, 0.95],
        fig_kwargs=fig_kwargs,
    ).on(ax).plot()
    ax.set_title(t)
    ax.set_xlabel('Clause type')
    ax.set_ylabel('Estimated difference')
    ax.axhline(0,
        color="black",
        linewidth=0.8,
        linestyle="--",
        alpha=0.7
    )

ax=axes[3]
bmb.interpret.plot_comparisons(model,
    results,
    contrast={"sentence_type": ['Subordinate', 'Main']},
    conditional="length",
    prob=[0.89, 0.95],
    fig_kwargs=fig_kwargs
).on(ax).plot()
ax.set_title('D. Main - Subordinate')
ax.set_xlabel('Dependency length')
ax.set_ylabel('Estimated difference')
ax.axhline(0,
    color="black",
    linewidth=0.8,
    linestyle="--",
    alpha=0.7
)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/LDD_comparisons.png')


# Plot the predictions of the GLMM

fig, ax = plt.subplots()

bmb.interpret.plot_predictions(model,
    results,
    conditional=["length", "sentence_type"],
    prob=[0.89, 0.95],
    fig_kwargs=fig_kwargs
).label(x='Dependency Length',
        y='Predicted accuracy',
        title='GLMM model predicted accuracy'
).on(ax).plot()

legend = fig.legends.pop(0)
ax.legend(
    legend.legend_handles,
    ["Main", "Subordinate"],
    title="Clause type",
    loc="lower right"
)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/LDD_predictions.png')