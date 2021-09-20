import pickle
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

data = pd.read_csv("C:/Users/maina/Documents/R term 4/ass2/D1.csv")
datac=data
#drop columns
headers=["Program.Code", "From.Grade", "To.Grade", "Group.State", "Is.Non.Annual.", "Days", "Travel.Type", "Special.Pay", "Tuition", "FRP.Active", "FRP.Cancelled", "FRP.Take.up.percent.", "Cancelled.Pax", "Total.Discount.Pax",  "Poverty.Code", "Region", "CRM.Segment", "School.Type", "Parent.Meeting.Flag", "MDR.Low.Grade", "MDR.High.Grade", "Total.School.Enrollment", "Income.Level", "EZ.Pay.Take.Up.Rate", "School.Sponsor", "SPR.Product.Type", "SPR.New.Existing", "FPP", "Total.Pax", "SPR.Group.Revenue", "NumberOfMeetingswithParents", "DifferenceTraveltoFirstMeeting", "DifferenceTraveltoLastMeeting", "SchoolGradeTypeLow", "SchoolGradeTypeHigh", "SchoolGradeType", "DepartureMonth", "GroupGradeTypeLow", "GroupGradeTypeHigh", "GroupGradeType", "MajorProgramCode", "SingleGradeTripFlag", "FPP.to.School.enrollment", "FPP.to.PAX", "Num.of.Non_FPP.PAX", "SchoolSizeIndicator", "DifferenceTravelToDeposit", "DifferenceERPLTravel", "DifferenceLRPLTravel", "DifferenceSystemTravel"]
Y=data["Retained.in.2012."]
X=(data[data.columns[data.columns.isin(headers)]])

F = (data[data.columns[data.columns.isin(headers)]])

#categorical=tuple(["Program.Code", " Group.State", " Travel.Type", " Special.Pay", " Poverty.Code", " Region", " School.Type", " MDR.Low.Grade", " Income.Level", " SPR.Product.Type", " SPR.New.Existing", " SchoolGradeTypeLow", " SchoolGradeTypeHigh", " SchoolGradeType", " DepartureMonth", " GroupGradeTypeLow", " GroupGradeTypeHigh", " GroupGradeType", " MajorProgramCode", " SchoolSizeIndicator", " DifferenceERPLTravel", " DifferenceLRPLTravel", " DifferenceSystemTravel"])
indices=[1, 4, 7, 8, 15, 16, 18, 20, 23, 26, 27, 34, 35, 36, 37, 38, 39, 40, 41, 46, 47, 48, 49]

#onehotencoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
for i in indices:
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

X2 = pd.DataFrame(X)

#dropping categorical
old_categorical = [54, 55 ,57, 58, 65,66, 68, 70, 71, 72, 73, 76, 77, 84, 85, 86, 87, 88, 89, 90, 91, 96, 98, 99, 100]
X2.drop(X2.columns[old_categorical], axis = 1, inplace = True)

X2 = X2.to_numpy()

#split training and test sets
seed=42
test_size=0.33
X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size=test_size, random_state=seed)

#######################################
#XGBOOST machine learning
###################################	

#fitting model to training data
#model XGBoost
modelxg = XGBClassifier(enable_categorical=True)
modelxg.fit(X_train, y_train)

print(modelxg)

y_pred = modelxg.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


filenm="xgboostmod"
save_xgboost = pickle.dump(modelxg, open(filenm, 'wb'))

#Now we will also check with a NN running adam

metrics.plot_roc_curve(modelxg, X_test, y_test)  
plt.show()     
                              

variation = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
labels=["retain", "not retain"]
for var, normalize in variation:
    disp = plot_confusion_matrix(modelxg, X_test, y_test,display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)


##############################
#Question 2
#############################

data2 = pd.read_csv("C:/Users/maina/Documents/R term 4/ass2/New folder/UV7583-XLS-ENG.csv")
data2c=data2

headers2=["NPS 2011", "NPS 2010", "NPS 2009", "NPS 2008"]

XI = (data2[data2.columns[data2.columns.isin(headers2)]])
AB = pd.DataFrame(XI)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(XI.iloc[:, 0:4])
XI.iloc[:, 0:4] = imputer.transform(XI.iloc[:, 0:4])


# Dropping last 3 rows of this dataframe so it matches Question 1 dataframe in no of rows
n = 3
XI.drop(XI.tail(n).index,
        inplace = True)
#print(XI)
X2temp = pd.DataFrame(X2)
NewX = X2temp.join(XI)
NewX = NewX.to_numpy()
#split training and test sets
seed=42
test_size=0.33
X_train2, X_test2, y_train2, y_test2 = train_test_split(NewX, Y, test_size=test_size, random_state=seed)

#model xgboost
modelxg2 = XGBClassifier(enable_categorical=True)
modelxg2.fit(X_train2, y_train2)

print(modelxg2)

y_pred = modelxg2.predict(X_test2)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test2, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


filenm="xgboostmod2"
save_xgboost = pickle.dump(modelxg2, open(filenm, 'wb'))

#Now we will also check with a NN running adam

metrics.plot_roc_curve(modelxg2, X_test2, y_test2)  
plt.show()     
                              

variation = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
labels=["retain", "not retain"]
for var, normalize in variation:
    disp = plot_confusion_matrix(modelxg2, X_test2, y_test2,display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)