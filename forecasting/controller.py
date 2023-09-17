from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



def forecast(req):
    if req.method == "POST":
        excel_file = req.FILES["excel_file"]
        data_frame = pd.read_excel(excel_file)
        series_data_frame = data_frame.set_index(data_frame.columns[0])[data_frame.columns[1]].resample('D').sum()
        X_train, Y_train, X_test, Y_test = datasets(series_data_frame)

        reg_model = LinearRegression()
        reg_model = reg_model.fit(X_train,Y_train)
        Y_test_pred = reg_model.predict(X_test)
        accuracy = ((abs(np.sum(Y_test)) - abs(np.sum((Y_test - Y_test_pred))))/ abs(np.sum(Y_test))) * 100
        
        future_pred = reg_model.predict([list(series_data_frame.iloc[len(series_data_frame)-30:])])
        X_future = list(series_data_frame.iloc[len(series_data_frame)-30:])
        X_future.append(future_pred[0])
        for i in range(29):
          future_pred = reg_model.predict([list(X_future[len(X_future)-30:])])
          X_future.append(future_pred[0])
        
        

        return HttpResponse(future_pred)
    

def datasets(df, x_len=30, test_loops=30):
  X_train = []
  Y_train = []
  #creating the training set
  for index in range(x_len,len(df) - test_loops):
    X_train.append(list(df.iloc[index - x_len:index].values))
    Y_train.append(df.iloc[index])
  
  X_test = []
  Y_test = []
  #creating the testing set
  for index in range(len(df) - test_loops,len(df)):
    X_test.append(list(df.iloc[index - x_len:index].values))
    Y_test.append(df.iloc[index])

  return X_train, Y_train, X_test, Y_test