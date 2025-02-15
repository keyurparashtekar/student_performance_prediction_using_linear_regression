import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import sklearn.model_selection
from sklearn.utils import shuffle
import pickle

data = pd.read_csv("student+performance/student/student-mat.csv",sep= ";")
prediction = "G3"

data = data[["G1","G2","absences","failures","studytime","freetime","goout","Dalc","Walc","health","G3"]]

data = shuffle(data)

x = np.array(data.drop([prediction], axis=1)) # dropping column G3 from features 
y =np.array(data[prediction])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1) 

best_accuracy = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)

    accuracy = linear.score(x_test,y_test)

    print(f"Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        with open("student_grade_predictor_new.pickle","wb") as f:
            pickle.dump(linear,f)
            