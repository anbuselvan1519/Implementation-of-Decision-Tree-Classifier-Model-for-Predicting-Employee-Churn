# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
### NAME:ANBUSELVAN.S
### DEPARTMENT:AIML
### REFERENCE NUMBER:212223240008

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values from dataframe and apply label encoder.

3.Apply decision tree classifier on the dataframe.

4.obtain the value of accuracy and data prediction.
## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ANBUSELVAN.S
RegisterNumber:  212223240008

import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

### Initial dataset:

![Screenshot 2024-04-02 083927](https://github.com/anbuselvan1519/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841744/d4e40521-84b4-44e6-8762-a7961a05084a)


### Data info:

![Screenshot 2024-04-02 084007](https://github.com/anbuselvan1519/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841744/181bf207-7f70-48d3-a5e4-066b1a1e73cf)

### Null values:

![Screenshot 2024-04-02 084039](https://github.com/anbuselvan1519/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841744/dbe90d64-9665-4b98-81e6-cbdc36de6a78)


### Assignment of x and y values:

![Screenshot 2024-04-02 084114](https://github.com/anbuselvan1519/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841744/26483060-8d73-4319-bc94-31f689f66674)
![Screenshot 2024-04-02 084145](https://github.com/anbuselvan1519/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841744/23fbd47f-674b-4ebb-a569-cd2f1332139b)


### Converting string literals to numerical values using label encoder:

![Screenshot 2024-04-02 084209](https://github.com/anbuselvan1519/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841744/8a529eb6-6fe0-47a5-b49b-4cbbd8191a5e)


### Accuracy:

![Screenshot 2024-04-02 084313](https://github.com/anbuselvan1519/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841744/558410c2-ec8e-43f6-8732-8ffc4fa5cb68)


### Prediction:

![Screenshot 2024-04-02 084421](https://github.com/anbuselvan1519/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841744/2b22dc0c-f683-4ef0-a9e5-c18f44526b50)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
