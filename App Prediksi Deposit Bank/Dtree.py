# import pandas as pd
# from sklearn import tree
# import pickle
# import seaborn as sns
# data = pd.read_csv('bank.csv')

# from sklearn.preprocessing import LabelEncoder
# enc = LabelEncoder()

# job = {'admin': 0, 'technician': 1, 'services': 2, 'management': 3,
#          'retired': 4, 'blue-collar':5, 'unemployed':6, 'entrepreneur':7,
#          'housemaid':8, 'unknown':9, 'self-employed':10,'student':11 }
# data['job'] = data['job'].map(job)

# marital = {'married':0,'single':1,'divorced':2}
# data['marital'] = data['marital'].map(marital)

# education = {'secondary':0, 'tertiary':1, 'primary':2, 'unknown':3}
# data['education'] = data['education'].map(education)

# data['default'] = enc.fit_transform(data['default'].values)
# data['housing'] = enc.fit_transform(data['housing'].values)
# data['loan'] = enc.fit_transform(data['loan'].values)

# contact = {'unknown':0,'cellular':1,'telephone':2}
# data['contact'] = data['contact'].map(contact)

# month = {'jan':0, 'feb':1, 'mar':2, 'apr':3,
#          'may':4, 'jun':5, 'jul':6, 'aug':7,
#          'sep':8, 'oct':9, 'nov':10, 'dec':11}
# data['month'] = data['month'].map(month)

# poutcome = {'unknown':0, 'other':1,'failure':2, 'success':3}
# data['poutcome'] = data['poutcome'].map(poutcome)

# data['deposit'] = enc.fit_transform(data['deposit'].values)


# dataku = data.drop(columns='deposit')
# target = data['deposit']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(dataku, target, test_size=0.3,random_state=500)

# from sklearn.tree import DecisionTreeClassifier
# drugTree = DecisionTreeClassifier(criterion="gini", max_depth=10)
# dt = drugTree.fit(X_train,y_train)

# pickle.dump(dt, open('bank.pkl', 'wb'))

import pandas as pd
import numpy as np
from sklearn import tree
import pickle
import seaborn as sns
from sklearn.tree import plot_tree
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score

data = pd.read_csv('bank.csv')

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

job = {'admin': 0, 'technician': 1, 'services': 2, 'management': 3,
         'retired': 4, 'blue-collar':5, 'unemployed':6, 'entrepreneur':7,
         'housemaid':8, 'unknown':9, 'self-employed':10,'student':11 }
data['job'] = data['job'].map(job)

marital = {'married':0,'single':1,'divorced':2}
data['marital'] = data['marital'].map(marital)

education = {'secondary':0, 'tertiary':1, 'primary':2, 'unknown':3}
data['education'] = data['education'].map(education)

data['default'] = enc.fit_transform(data['default'].values)
data['housing'] = enc.fit_transform(data['housing'].values)
data['loan'] = enc.fit_transform(data['loan'].values)

contact = {'unknown':0,'cellular':1,'telephone':2}
data['contact'] = data['contact'].map(contact)

month = {'jan':0, 'feb':1, 'mar':2, 'apr':3,
         'may':4, 'jun':5, 'jul':6, 'aug':7,
         'sep':8, 'oct':9, 'nov':10, 'dec':11}
data['month'] = data['month'].map(month)

poutcome = {'unknown':0, 'other':1,'failure':2, 'success':3}
data['poutcome'] = data['poutcome'].map(poutcome)

data['deposit'] = enc.fit_transform(data['deposit'].values)

features = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']

dataku = data[features]
target = data['deposit']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataku, target, test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="gini", max_depth=10)
drugTree = drugTree.fit(X_train,y_train)

predTree = drugTree.predict(X_test)

pickle.dump(drugTree, open('bank.pkl', 'wb'))

# Untuk save Dtree ke format png
# dataa = tree.export_graphviz(drugTree, out_file=None, feature_names=features)
# graph = pydotplus.graph_from_dot_data(dataa)
# graph.write_png('PohonKeputusan.png')

# Untuk save sns_pairplot ke format png
# sns.set()
# sns_plot = sns.pairplot(data, hue='deposit', height= 3)
# sns_plot.savefig("pairplot_sns.png")