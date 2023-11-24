import pandas as pd
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import joblib
import matplotlib.pyplot as plt 

data=pd.read_csv(r'D:/assignments/Data_Science/Association_rule/book.csv')
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root1", pw = "Reddy2000", db = "ml")) 
data.to_sql('book',con=engine,if_exists='replace',chunksize=1000,index=False)
sql='select * from book;'
df=pd.read_sql_query(sql,engine)


# Most popular items
count = df.loc[:, :].sum()

# Generates a series
pop_item = count.sort_values(ascending = False).head(10)

pop_item = pop_item.to_frame()

pop_item = pop_item.reset_index()
pop_item

pop_item = pop_item.rename(columns = {"index": "items", 0: "count"})
pop_item

plt.rcParams['figure.figsize'] = (10, 6) # rc stands for runtime configuration 
plt.style.use('dark_background')
pop_item.plot.barh()
plt.title('Most popular items')
plt.gca().invert_yaxis() 

frequent_itemsets = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)
frequent_itemsets

frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(10)
rules.sort_values('lift', ascending = False).head(10)

def to_list(i):
    return (sorted(list(i)))

# Sort the items in Antecedents and Consequents based on Alphabetical order
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

# Sort the merged list of items - transactions
ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

# Capture the index of unique item sets
index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    
index_rules

rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy

# Sorted list and top 10 rules 
rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(10)

rules10.plot(x = "support", y = "confidence", c = rules10.lift, 
             kind = "scatter", s = 12, cmap = plt.cm.coolwarm)

# Removing frozenset from dataframe
rules10['antecedents'] = rules10['antecedents'].astype('string')
rules10['consequents'] = rules10['consequents'].astype('string')

rules10['antecedents'] = rules10['antecedents'].str.removeprefix("frozenset({")
rules10['antecedents'] = rules10['antecedents'].str.removesuffix("})")

rules10['consequents'] = rules10['consequents'].str.removeprefix("frozenset({")
rules10['consequents'] = rules10['consequents'].str.removesuffix("})")

rules10.to_sql('book_1', con = engine, if_exists = 'replace', chunksize = 1000, index = False)










