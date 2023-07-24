import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random




StudentID = '108062313'
input_dataroot = 'input.csv'
output_dataroot = StudentID + '_basic_prediction.csv'

input_datalist =  [] 
output_datalist =  [] 
test_datalist=[]
training_datalist=[]
validation_list=[]



with open(input_dataroot, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))

    
def SplitData(): 
    validation = input_datalist[0:30,:]
    training = input_datalist[0:158,:]
    test = input_datalist[159:186,:]
    return test, training, validation

def PreprocessData(training_datalist, test_datalist):
    temp=[]
    temp2=[]
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    date=[]
    for i in range (len(training_datalist)):
        temp.append(1)
        X_train.append(float(training_datalist[i][1]))
        Y_train.append(float(training_datalist[i][2]))
    for j in range (len(test_datalist)):
        date.append(test_datalist[j][0])
        temp2.append(1)
        X_test.append(float(test_datalist[j][1]))
        Y_test.append(float(test_datalist[j][2]))
    X_train=[X_train,temp]
    X_train=np.array(X_train)
    X_train=X_train.T
    X_test=[X_test,temp2]
    X_test=np.array(X_test)
    X_test=X_test.T
    return X_train, Y_train, X_test, Y_test, date

def Regression(X_train, Y_train, learning_rate=1e-7, iteration=20):
# TODO 4: Implement regression
    w=np.array([1,1])
    for i in range (iteration):
        y_=[]
        for j in range (len(X_train)):
            prediction=np.dot(w.T,X_train[j])
            y_.append(prediction)
        g=CountLoss(Y_train,y_)
        w=w-learning_rate*g/len(X_train)    
    print(w)
#     print(y_)
    return w

def CountLoss(y, y_):
    j=[0,0]
    s=0
    for i in range (len(y)):
        difference=y_[i]-y[i]
        s=s+abs(difference)
        j=j+(difference*X_train[i]) 
    j=j*2
    return j


def MakePrediction(x,y,w,date):
    prediction=[]
    for i in range(len(x)):
        y_=np.dot(w,x[i])
        prediction.append(int(y_))
    prediction=[date,prediction]
    return prediction 
def evaluate(x,y,w):
    prediction=0
    s=0
    for i in range(len(x)):
        y_=np.dot(w,x[i])
        s=s+abs(y[i]-y_)
        prediction=prediction+abs((y[i]-y_)/y[i])
        # print(y[i])
        # print(y[i]-y_)
        # print((y[i]-y_)/y[i])
        # print("=======")
    prediction=prediction/len(x)
    print(s/len(x))
    return prediction*100

test_datalist, training_datalist, validation_list = SplitData()
X_train, Y_train, X_test, Y_test, date= PreprocessData(training_datalist, test_datalist)
w=Regression(X_train, Y_train)
print("W:")
print(w)
output_datalist=np.array(MakePrediction(X_test, Y_test, w, date)).T
print(output_datalist)
print(evaluate(X_test,Y_test,w))
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'TSMC Price'])
    for row in output_datalist:
        writer.writerow(row)

