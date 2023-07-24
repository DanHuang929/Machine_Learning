import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random
StudentID = '108062313' # TODO 1 : Fill your student ID here
input_dataroot = 'bonus_input.csv' # Please name your input csv file as 'input.csv'
output_dataroot = StudentID + '_bonus_prediction.csv' # Output file will be named as '[StudentID]_basic_prediction.csv'

input_datalist =  [] # Initial datalist, saved as numpy array
output_datalist =  [] # Your prediction, should be 20 * 2 matrix and saved as numpy array
                      # The format of each row should be [Date, TSMC_Price_Prediction] 
                      # e.g. ['2021/10/15', 512]
with open(input_dataroot, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))
def SplitData(): 
    validation = input_datalist[0:30,:]
    training = input_datalist[0:189,:]
    test = input_datalist[189:209,:]
    date=[]
    for i in range(len(test)):
        date.append(test[i][0])
    return test, training, validation, date
def PreprocessData(training_datalist):
# TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    temp=[]
    temp2=[]
    
    X_train=[]
    X2_train=[]
    Y_train=[]
    
    X_test=[]
    X2_test=[]
    Y_test=[]
    
    for i in range (len(training_datalist)):
        temp.append(1)
        X_train.append(float(training_datalist[i][1]))
        X2_train.append(float(training_datalist[i][2])*10)
        Y_train.append(float(training_datalist[i][3]))
#     for j in range (len(test_datalist)):
#         temp2.append(1)
#         date.append(test_datalist[j][0])
#         X_test.append(float(test_datalist[j][1]))
#         X2_test.append(float(test_datalist[j][2])*10)
#         Y_test.append(float(test_datalist[j][3]))
    X_train=[X_train,X2_train,temp]
    X_train=np.array(X_train)
    X_train=X_train.T
#     X_test=[X_test,X_test,temp2]
#     X_test=np.array(X_test)
#     X_test=X_test.T
    return X_train, Y_train
def auto_PreprocessData(training_datalist):
# TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    temp=[]
    X_train=[]
    X1_train=[]
    X2_train=[]
    X3_train=[]
    X4_train=[]
    Y_train=[]
    for i in range (len(training_datalist)):
        temp.append(1)
        if(i!=0):
            X1_train.append(float(training_datalist[i-1]))
        else:
            X1_train.append(float(training_datalist[i]))
        if(i!=0 and i!=1):
            X2_train.append(float(training_datalist[i-2]))
        else:
            X2_train.append(float(training_datalist[i]))
        if(i!=0 and i!=1 and i!=2):
            X3_train.append(float(training_datalist[i-3]))
        else:
            X3_train.append(float(training_datalist[i]))
        if(i!=0 and i!=1 and i!=2 and i!=3):
            X4_train.append(float(training_datalist[i-4]))
        else:
            X4_train.append(float(training_datalist[i]))
        Y_train.append(float(training_datalist[i]))
    X_train=[X4_train,X3_train,X2_train,X1_train,temp]
    X_train=np.array(X_train)
    X_train=X_train.T
    return X_train, Y_train

def CountLoss(y, y_,X_train):
# TODO 5: Count loss of training and validation data
    j=[0,0,0,0,0]
    s=0
    for i in range (len(y)):
        difference=y_[i]-y[i]
        s=s+abs(difference)
        j=j+(difference*X_train[i]) 
    j=j*2
#     print(s/len(y))
#     print("========")
    return j
def TSMC_CountLoss(y, y_,X_train):
# TODO 5: Count loss of training and validation data
    j=[0,0,0]
    s=0
    for i in range (len(y)):
        difference=y_[i]-y[i]
        s=s+abs(difference)
        j=j+(difference*X_train[i]) 
    j=j*2
#     print(s/len(y))
#     print("========")
    return j
def TSMC_Regression(X_train, Y_train, learning_rate=1e-7, iteration=20000):
# TODO 4: Implement regression
    w=np.array([0,0,0])
    for i in range (iteration):
        y_=[]
        for j in range (len(X_train)):
            prediction=np.dot(w.T,X_train[j])
            y_.append(prediction)
        g=TSMC_CountLoss(Y_train,y_,X_train)
        w=w-learning_rate*g/len(X_train)    
    print(w)
    return w
def MTK_AutoRegression(X_train, Y_train, learning_rate=(1e-7), iteration=20000):
    # TODO 4: Implement regression
#     X_train=[X4_train,X3_train,X2_train,X1_train,temp]
    w=np.array([-1,1,-1,1,30])
    for i in range (iteration):
        y_=[]
        for j in range (len(X_train)):
            prediction=np.dot(w.T,X_train[j])
            y_.append(prediction)
        g=CountLoss(Y_train,y_,X_train)
        if((abs(learning_rate*g[0]/len(X_train))) < 1e-5 ):
            return w
        w=w-learning_rate*g/len(X_train)    
    print(w)
    return w
def UMC_AutoRegression(X_train, Y_train, learning_rate=(1e-5), iteration=20000):
    # TODO 4: Implement regression
    w=np.array([-1,1,-1,1,10])
    for i in range (iteration):
        y_=[]
        for j in range (len(X_train)):
            prediction=np.dot(w.T,X_train[j])
#             if(j==50):
#                 print(prediction)
#                 print(X_train[j])
            y_.append(prediction)
        g=CountLoss(Y_train,y_,X_train)
#         if((learning_rate*g[0]/len(X_train)) < 1e-9 ):
#             return w
        w=w-(learning_rate*g)/len(X_train)
    print(w)
    return w
def evaluate(y_,y):
    prediction=0
    s=0
    for i in range(len(y)):
        s=s+abs(y[i]-y_[i])
        prediction=prediction+abs((y[i]-y_[i])/y[i])
#         print(y[i])
#         print(y[i]-y_)
#         print((y[i]-y_)/y[i])
#         print("=======")
    prediction=prediction/len(y)
    print(s/len(y))
    return prediction*100
def Auto_MakePrediction(x1,x2,x3,x4,w):
# TODO 6: Make prediction of testing data 
# [X3_train,X2_train,X1_train,temp]
    future=[]
    day=[]
    for i in range(20):
        if(i==0):
            y_=np.dot(w,[x4,x3,x2,x1,1])
            future.append(int(y_))
        elif(i==1):
            y_=np.dot(w,[x3,x2,x1,future[0],1])
            future.append(int(y_))
        elif(i==2):
            y_=np.dot(w,[x2,x1,future[0],future[1],1])
            future.append(int(y_))
        elif(i==3):
            y_=np.dot(w,[x1,future[i-3],future[i-2],future[i-1],1])
            future.append(int(y_))
        else:
            y_=np.dot(w,[future[i-4],future[i-3],future[i-2],future[i-1],1])
            future.append(int(y_))
#     prediction=[date,prediction]
    return future
def MakePrediction(x,w,date):
# TODO 6: Make prediction of testing data 
    prediction=[]
#     print(len(x))
#     print(len(date))
    for i in range(len(x)):
        y_=np.dot(w,x[i])
        prediction.append(int(y_))
    prediction=np.array([date,prediction])
    return prediction
test_datalist, training_datalist, validation_list, test_date= SplitData()
TSMC_X_train, TSMC_Y_train= PreprocessData(training_datalist)
# TSMC_X_test, TSMC_Y_test= PreprocessData(test_datalist)
MTK_X_train,MTK_Y_train= auto_PreprocessData(training_datalist[:,1])
# MTK_X_test,MTK_Y_test= auto_PreprocessData(test_datalist[:,1])
UMC_X_train,UMC_Y_train= auto_PreprocessData(training_datalist[:,2])
# UMC_X_test,UMC_Y_test= auto_PreprocessData(test_datalist[:,2])
TSMC_w=TSMC_Regression(TSMC_X_train, TSMC_Y_train)
# TSMC_prediction=MakePrediction(TSMC_X_test,TSMC_w)

MTK_w=MTK_AutoRegression(MTK_X_train, MTK_Y_train)
MTK_future=Auto_MakePrediction(MTK_Y_train[-1],MTK_Y_train[-2],MTK_Y_train[-3],MTK_Y_train[-4],MTK_w)
UMC_w=UMC_AutoRegression(UMC_X_train, UMC_Y_train)
UMC_future=Auto_MakePrediction(UMC_Y_train[-1],UMC_Y_train[-2],UMC_Y_train[-3],UMC_Y_train[-4],UMC_w)

future_X_test=[]
temp=[]
for i in range(len(MTK_future)):
    UMC_future[i]=UMC_future[i]*10
    temp.append(1)

future_X_test=np.array([MTK_future,UMC_future,temp]).T
# print(UMC_future)
# print("======")
output_datalist=np.array(MakePrediction(future_X_test, TSMC_w, test_date)).T
# Write prediction to output csv
print(output_datalist)
# print(TSMC_Y_test)
# print(evaluate(output_datalist,TSMC_Y_test))
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for row in output_datalist:
        writer.writerow(row)