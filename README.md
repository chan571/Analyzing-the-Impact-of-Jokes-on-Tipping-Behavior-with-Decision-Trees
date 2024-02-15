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
<img width="242" alt="Screenshot 2024-02-14 at 9 42 07 PM" src="https://github.com/chan571/Impact-of-Jokes-on-Tipping-Behavior/assets/157858508/f30f08b5-0558-4d72-88ed-9ce714d3886d">

The experiment groups serve as the explanatory variables, and the outcome variable is whether customers left a tip.
Split the dataset into 70% of training data, and 30% remaining for testing.

```
X = df[['Ad','Joke','None']]
y = df['Tip']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
```
Train on the training data, and used to make predictions on the testing data. 
Check the accuracy of the model on the testing data, followed by the output of the confusion matrix.

```
X = df.drop(columns='Tip')
y = df['Tip']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```
<img width="94" alt="Screenshot 2024-02-14 at 9 46 31 PM" src="https://github.com/chan571/Impact-of-Jokes-on-Tipping-Behavior/assets/157858508/403f135d-608f-4512-bb04-5ccae14d3ed1">

Visualization of a decision tree classifier using the Graphviz library
```
import graphviz
dot_data = tree.export_graphviz(dtree,out_file=None,feature_names=('Ad','Joke','None'),
                                class_names=('0','1'),
                                filled=True)
graph = graphviz.Source(dot_data, format="png")
graph.render('TipJoke_dt',view=True)

with open("TipJoke_dt") as f:
  dot_graph = f.read()
graphviz.Source(dot_graph)
```
<img width="420" alt="Screenshot 2024-02-14 at 9 48 13 PM" src="https://github.com/chan571/Impact-of-Jokes-on-Tipping-Behavior/assets/157858508/2449bb37-cf56-474a-b194-a67247c81c92">

Also the text representation. 
```
text_representation = tree.export_text(dtree)
print(text_representation)
```
<img width="231" alt="Screenshot 2024-02-14 at 9 48 53 PM" src="https://github.com/chan571/Impact-of-Jokes-on-Tipping-Behavior/assets/157858508/485c0b1c-ca09-42c0-89cd-c003329cd840">

Model Interpretation:
Indicator ‘Joke’ is put as the root node, which has maximum information gain. It represents the best predictor for whether customers tip. 
The majority class within the root node is 0, indicating no tipping, and constitutes 68% of sample. 
In the group that received a card featuring a joke with the bill, the likelihood of leaving a tip increases to almost 50%. 
Among those who did not receive a card with a joke but did receive one containing an advertisement for a local restaurant, the majority class remains no tipping, with only 17% of the sample leaving a tip. 
For the group that received neither a joke nor an advertisement, 26% of the sample left a tip. 
In conclusion, not receiving any card offers a better chance of leaving a tip compared to receiving a card containing advertisements. The group that received a card with a joke has the highest probability of leaving tips.
The gini index serves as a metric representing impurities, where a higher value indicates a greater chance of misclassification. 
The decision node of 'Joke' > 0.5 exhibits the highest gini value, signifying higher impurity, while the group that received an advertisement card has the lowest value of 0.278, indicating higher accuracy.



`
