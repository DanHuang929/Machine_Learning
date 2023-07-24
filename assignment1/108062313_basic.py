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
    training = input_datalist[0:189,:]
    validation = input_datalist[159:189,:]
    test = input_datalist[189:209,:]
    return test, training, validation

def PreprocessData(datalist):
# TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    temp=[]
    X_train=[]
    X2_train=[]
    X3_train=[]
    X4_train=[]
    Y_train=[]
    date=[]
    for i in range (len(datalist)):
        temp.append(1)
        date.append(datalist[i][0])
        X_train.append(float(datalist[i][1]))
        X2_train.append(float(datalist[i][1])**2)
        X3_train.append(float(datalist[i][1])**3)
        X4_train.append(float(datalist[i][1])**4)
        Y_train.append(float(datalist[i][2]))
    X_train=[X4_train,X3_train,X2_train,X_train,temp]
    X_train=np.array(X_train)
    X_train=X_train.T
    return X_train, Y_train, date

def Regression(X_train, Y_train, learning_rate=[1e-29,1e-22,1e-15,1e-9,1e-5], iteration=20000):
# TODO 4: Implement regression
    w=np.array([0,0,0,0,0])
    for i in range (iteration):
        y_=[]
        for j in range (len(X_train)):
            prediction=np.dot(w.T,X_train[j])
            y_.append(prediction)
        g=CountLoss(Y_train,y_)
        w=w-learning_rate*g   
        # print(w)
        # if(evaluate(X_train, Y_train, w)<2.2):
            # return w
        # print(evaluate(X_train, Y_train, w))
#     print(y_)
    return w

def CountLoss(y, y_):
    j=[0,0,0,0,0]
    s=0
    for i in range (len(y)):
        difference=y_[i]-y[i]
        s=s+abs(difference)
        j=j+(difference*X_train[i]) 
    j=j*2
    # print(s/len(y))
    return j


def MakePrediction(x,w,date):
    prediction=[]
    for i in range(len(x)):
        y_=np.dot(w,x[i])
        prediction.append(int(y_))
    prediction=[date,prediction]
    return prediction
# def evaluate(x,y,w):
#     prediction=0
#     s=0
#     for i in range(len(x)):
#         y_=np.dot(w,x[i])
#         s=s+abs(y[i]-y_)
#         prediction=prediction+abs((y[i]-y_)/y[i])
#         # print(y[i])
#         # print(y[i]-y_)
#         # print((y[i]-y_)/y[i])
#         # print("=======")
#     prediction=prediction/len(x)
#     # print(s/len(x))
#     return prediction*100

test_datalist, training_datalist, validation_datalist = SplitData()
X_train, Y_train ,train_date= PreprocessData(training_datalist)
X_validation, Y_validation ,validation_date= PreprocessData(validation_datalist)
X_test, Y_test ,test_date= PreprocessData(test_datalist)
w=Regression(X_train, Y_train)
# print("W:")
print(w)
# print(evaluate(X_validation, Y_validation, w))
output_datalist=np.array(MakePrediction(X_test, w, test_date)).T
print(output_datalist)
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for row in output_datalist:
        writer.writerow(row)

