from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
from sqlalchemy import create_engine

# Connecting to sql by creating sqlachemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root1", # user
                               pw = "Reddy2000", # password
                               db = "ml")) # database
# Define flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        
        df=pd.read_csv(f)
        count = df.loc[:, :].sum()

        # Generates a series
        pop_item = count.sort_values(ascending = False).head(10)

        pop_item = pop_item.to_frame()

        pop_item = pop_item.reset_index()
        

        pop_item = pop_item.rename(columns = {"index": "items", 0: "count"})
      
 
        frequent_itemsets = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)
       

        frequent_itemsets.sort_values('support', ascending = False, inplace = True)
        

        rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
        
        rules.sort_values('lift', ascending = False)

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
            
        

        rules_no_redundancy = rules.iloc[index_rules, :]
       

        # Sorted list and top 10 rules 
        rules10 = rules_no_redundancy.sort_values('lift', ascending = False)
        rules10 = rules10.replace([np.inf, -np.inf], np.nan)

      

        # Removing frozenset from dataframe
        rules10['antecedents'] = rules10['antecedents'].astype('string')
        rules10['consequents'] = rules10['consequents'].astype('string')

        rules10['antecedents'] = rules10['antecedents'].str.removeprefix("frozenset({")
        rules10['antecedents'] = rules10['antecedents'].str.removesuffix("})")

        rules10['consequents'] = rules10['consequents'].str.removeprefix("frozenset({")
        rules10['consequents'] = rules10['consequents'].str.removesuffix("})")
        
        rules10.to_sql('book1', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        html_table = rules10.to_html(classes = 'table table-striped')
        
        return render_template("new.html", Y =   f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #5e617d;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}") 
                
if __name__=='__main__':
    app.run(debug = True)

        