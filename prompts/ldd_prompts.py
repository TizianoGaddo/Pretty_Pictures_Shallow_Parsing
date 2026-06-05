from itertools import product

OUTPUT_DIR = "./prompts" # path to the folder in which to save the txt file with ldd prompts

subj    = set(['cat', 'dog', 'rabbit', 'mouse'])
verb    = set(['black', 'sitting'])
s_subj  = set(['man', 'woman', 'girl', 'boy', 'child'])
s_verb  = set(['petting', 'looking at'])

l = list(product(subj, verb, s_subj, s_verb))       # create all (80) possible combinations of the above elements

adj1    = set(['red', 'yellow', 'blue'])
attr1   = set(['shirt', 'hat', 'backpack'])
adj2    = set(['green', 'purple', 'brown'])
attr2   = set(['gloves', 'trousers', 'shoes'])

l_attr = list(product(adj1, attr1, adj2, attr2))    # create all (81) possible combinations of the above elements
l_attr.pop()                                        # since we only need 80, remove the last one

ldd_base    = []
ldd_med     = []
ldd_long    = []

for l_m, l_s in zip(l, l_attr):
    ldd_base.append(f'The {l_m[0]} that the {l_m[2]} is {l_m[3]} is {l_m[1]}.')
    ldd_med.append(f'The {l_m[0]} that the {l_m[2]} with a {l_s[0]} {l_s[1]} is {l_m[3]} is {l_m[1]}.')
    ldd_long.append(f'The {l_m[0]} that the {l_m[2]} with a {l_s[0]} {l_s[1]} and {l_s[2]} {l_s[3]} is {l_m[3]} is {l_m[1]}.')

lists = [ldd_base, ldd_med, ldd_long]
ldd = [val for tup in zip(*lists) for val in tup]
with open(f'{OUTPUT_DIR}/ldd.txt', 'w') as f:
    for l in ldd:
        f.write(f'{l}\n')
