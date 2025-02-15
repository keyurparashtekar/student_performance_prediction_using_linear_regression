import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import sklearn.model_selection
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv("student+performance/student/student-mat.csv",sep= ";")
prediction = "G3"

data = data[["G1","G2","absences","failures","studytime","G3"]]

data = shuffle(data)

x = np.array(data.drop([prediction], axis=1)) # dropping column G3 from features 
y =np.array(data[prediction])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

style.use("ggplot")

pickle_in = open("student_grade_predictor.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])


# Drawing and plotting model
plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()