This analysis is sourced from the study “The Effects of a Joke on Tipping When it is Delivered at the Same Time as the Bill,” by Nicholas Gueguen (2002). 

Does telling a joke affect whether a waiter in a coffee bar gets a tip? 
This was studied at a famous resort's coffee bar on the west coast of France. 
The waiter split customers into three groups randomly: 1. A joke card with the bill. 2. A restaurant ad with the bill 3. Nothing with the bill. 
Then, the waiter noted if each customer left a tip or not.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix
```

```
# Load data 
df = pd.read_csv('TipJoke.csv')
df
```

The experiment groups serve as the explanatory variables, and the outcome variable is whether customers left a tip.
Split the dataset into 70% of training data, and 30% remaining for testing.

```
X = df[['Ad','Joke','None']]
y = df['Tip']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
```



`
