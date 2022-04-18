import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules,fpgrowth, fpmax

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

import streamlit as st

# setting up pandas options
pd.set_option('display.max_colwidth', None)


#paths
path = 'ecomm-open-cdp/items_ohe_2019_oct.parquet'

#sidebar
st.sidebar.markdown('### Welcome to the Market Basket analysis Demo')
st.sidebar.markdown('Market Basket analysis is used to identify association rules between products.')
st.sidebar.markdown('Here, we use the FPgrowth algorithm to identify good rules and prune them further based on useful metrics.')
st.sidebar.markdown('Created by Hassan Javed: [Linkedin](https://www.linkedin.com/in/hassan-javed-610/)')
#Title
st.title('Ecommerce Data Market Basket Analysis')
st.markdown('This dataset has been taken from a multi-category store and collected by the [Open CDP](https://rees46.com/en/datasets) project.')

st.markdown('A snapshot of the transaction data is shown below:')

#read csv util
@st.cache
def read_csv(path):
    return pd.read_parquet(path)

items_df = read_csv(path).reset_index(drop=True)
items_df = items_df.drop(['Unnamed: 0'],axis=1)

st.write(items_df.head())

#transaction data stats
col1, col2 = st.columns(2)

with col1:
    st.metric(label='Total transactions',value=items_df.shape[0], delta=None)
with col2:
    st.metric(label='Total categories',value=items_df.shape[1], delta=None)



#exploring support
supports = items_df.apply(np.mean)
st.subheader("View the most selling and least selling items based on support")


#top_n slider
supports_range = np.arange(1,len(supports))
n_top = st.select_slider('Select N top selling items', options=supports_range, value=10 )

#plot 10 most selling items
fig, ax = plt.subplots(figsize=(10,3))
ax.bar(supports.nlargest(n_top).index, supports.nlargest(n_top))
ax.set_title('Most selling items')
ax.set_xlabel('Products')
ax.set_ylabel('Support')
plt.xticks(rotation=70)

st.pyplot(fig)

#bot_n slider
supports_range = np.arange(1,len(supports))
bot_n = st.select_slider('Select N least selling items', options=supports_range, value=10 )

fig, ax = plt.subplots(figsize=(10,3.5))
ax.bar(supports.nsmallest(bot_n).index, supports.nsmallest(bot_n))
ax.set_title('Least selling items')
ax.set_xlabel('Products')
ax.set_ylabel('Support')
plt.xticks(rotation=70)

st.pyplot(fig)


#apriori algorithm
#st.subheader('Apriori association rules')
st.subheader('Generate itemsets and association rules')

# algo_opt = st.selectbox(
#      'Select Algorithm',
#      ('FPgrowth', 'FPmax'))

col1, col2 = st.columns(2)
with col1:
    #slider min_support
    N=6
    sup_opts = np.sort([(0.1/(10**i)) for i in range(N)])
    minsup = st.select_slider('min_support',options=sup_opts, value=10**-4)

with col2:
    #slider max_len
    len_opts = np.arange(1,11)
    maxlen = st.select_slider('max_len',options=len_opts, value=3)

#fpmax call
@st.cache
def fpmax_algo(df, min_support, max_len):
    return fpmax(items_df, min_support = min_support, max_len = max_len, use_colnames=True)
#FPgrowth calls
@st.cache
def fpgrowth_algo(df, min_support, max_len):
    return fpgrowth(items_df, min_support = min_support, max_len = max_len, use_colnames=True)

# algo_dict = {'FPgrowth':fpgrowth_algo,'FPmax':fpmax_algo}
# frequent_itemsets = algo_dict[algo_opt](items_df, minsup, maxlen)
frequent_itemsets = fpgrowth_algo(items_df, minsup, maxlen)



# association rules

col1, col2 = st.columns(2)
N=5
sup_opts = np.sort([(1/(10**i)) for i in range(N)])
lift_opts = np.sort([(i) for i in range(1,(N+1)*10,10)])

with col1:
    
    option = st.selectbox(
        'association rule metric',
        ('support', 'lift', 'confidence'))
    

with col2:
    if option == 'support':
        thresh = st.select_slider('min_thresh',options=sup_opts, value=np.min(sup_opts))
    elif option == 'lift':
        thresh = st.select_slider('min_thresh',options=lift_opts, value=np.min(lift_opts))
    elif option == 'confidence':
        thresh = st.select_slider('min_thresh',options=sup_opts, value=np.min(sup_opts))

rules = association_rules(frequent_itemsets, metric=option.lower(), min_threshold=thresh)

rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))


st.markdown('Association rules')
st.write(rules)

if rules.shape[0] == 0:
    col1, col2 = st.columns(2)
    with col1:
       st.write('#### No association rules to display.')
    with col2:
       st.write('#### Please, change the parameters.')
else:
    # Manually prune dataframes
    st.subheader('Manually prune rules')


    col1, col2, col3 = st.columns(3)
    with col1:
        sup_opts = np.sort(rules['support'].round(5).unique())
        sup = st.select_slider('support',options=sup_opts, value=sup_opts[0])
    with col2:
        lift_opts = np.sort(rules['lift'].round(2).unique())
        lft = st.select_slider('lift',options=lift_opts, value=lift_opts[0])
    with col3:
        conf_opts = np.sort(rules['confidence'].round(2).unique())
        cnf = st.select_slider('confidence',options=conf_opts, value=conf_opts[0])

   
    ant_rules = np.array(rules['antecedents'].to_list()).flatten().tolist()
    con_rules = np.array(rules['consequents'].to_list()).flatten().tolist()

    # col1, col2 = st.columns(2)
    # with col1:
    #     opts_ant = st.multiselect(
    #     'choose antecedents',
    #     ant_rules,None)
    # with col2:
    #     opts_con = st.multiselect(
    #     'choose consequents',
    #     con_rules,None)
    
    opts_ant = st.multiselect(
        'Choose antecedents',
        ant_rules,None)
    opts_con = st.multiselect(
        'Choose consequents',
        con_rules,None)

    # Filtering on filters
    rules_manual =  rules[(rules.support >= sup) & (rules.lift >= lft) & (rules.confidence >= cnf)]
    if opts_ant or opts_con:
        rules_manual = rules_manual[rules_manual.antecedents.isin(opts_ant) | rules_manual.consequents.isin(opts_con)]
    st.markdown('Pruned Association rules')
    st.write(rules_manual)

    #Association rules stats
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Association Rules before pruning", value=rules.shape[0], delta=None)
    with col2:
        st.metric(label="Association Rules after pruning", value=rules_manual.shape[0], delta=\
            str(format(((rules_manual.shape[0]/rules.shape[0])-1)*100,'.2f'))+'%')



    #Visualizations
    st.subheader('Visualize rules')
    #Visualizations: Heatmap
    st.markdown('Heatmap of relations')
    rules_sorted = rules_manual.sort_values(by='lift', ascending=False)
    rules_t20 = rules_sorted.nlargest(20,'lift')
    lift_table = rules_t20.pivot(index='antecedents', columns='consequents', values='lift')

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(lift_table, annot=True, ax=ax)
    st.write(fig)

    #Visualization: Parallel coordinates
    st.markdown('Parallel co-ordinates plot')
    rules_t20['rule'] = rules_t20.index
    coords = rules_t20[['antecedents','consequents','rule']]
    fig, ax = plt.subplots(figsize=(10,12))
    parallel_coordinates(coords, 'rule', colormap = 'ocean', ax=ax)
    st.write(fig)

    #Market basket use-cases
    st.subheader('Custom Market Basket use-cases')

    #Promotional buckets
    rules = rules_manual
    st.markdown('1. Promotional Buckets from relations')
    CONF_THRESH = 0.05

    #slider CONF_THRESH
    N=6
    conf_opts = np.sort([(CONF_THRESH*i) for i in range(1,N)])
    conf_sel = st.select_slider('Confidence difference threshold',options=conf_opts, value=conf_opts.min())

    #rounding off floats before grouping
    rules['lift'] = round(rules['lift'],5)
    # promotional bundles logic 
    promos_filt = rules.groupby('lift') \
        .filter(lambda x: np.abs( x['confidence'].iloc[0] - x['confidence'].iloc[1] ) < conf_sel) \

    promos = promos_filt.groupby('lift') \
        .apply(lambda x: ','.join(list(x['antecedents'])))\
        .to_frame('promo_buckets')\
        .reset_index()

    st.write(promos)
   
    # cross sell logic
    st.markdown('2. Cross-sell candidate products')
    cross_sells_cand = rules[~rules.antecedents.isin(promos_filt.antecedents)]
    cross_sells = cross_sells_cand.groupby('lift')\
                .apply(lambda x: x[x['confidence'] ==  x['confidence'].max()])\
                .reset_index(drop=True)

    st.dataframe(cross_sells[['lift','antecedents','consequents']])

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label='Promotional buckets #', value=promos.shape[0], delta=None)
    with col2:
        st.metric(label='Cross-sell candidates #', value=cross_sells.shape[0], delta=None)