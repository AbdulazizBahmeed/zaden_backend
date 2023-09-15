from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np



def forecast(req):
    if req.method == "POST":
        excel_file = req.FILES["excel_file"]
        data_frame = pd.read_excel(excel_file)
        df_out = data_frame.set_index(data_frame.columns[0],drop=False)[data_frame.columns[1]].resample('D')
        print(df_out)
        df = pd.pivot_table(data = data_frame,
                            columns=data_frame.columns[0],
                            values=data_frame.columns[1],
                            index=data_frame.columns[2],
                            aggfunc='sum',
                            fill_value=0)
        # print(df)
        # #here we test the model
        # x_len = int(len(df.values[0]) / 3)  #with the previous half period we predict the future
        # X_train, Y_train, X_test, Y_test = datasets(df, x_len=x_len, y_len=future_period,test_loops=1)
        
        # model.fit(X_train,Y_train)
        # Y_train_pred = model.predict(X_train)
        # Y_test_pred = model.predict(X_test)
        # error = kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name = model, train= False)["Bias"]["Test"]
        # print(Y_test)
        # print(Y_test_pred)
        # #here we prepare the input for future focracsting
        # X_test = df.iloc[: , -x_len:].values
        # future_forcast = model.predict(X_test)

        # resutl = {
        #     "future": future_forcast,
        #     "error":error
        # }
        return HttpResponse(df_out)
    

def datasets(df, x_len=12, y_len=1, test_loops=12):
    D = df.values
    rows, periods = D.shape
    loops = periods + 1 - x_len - y_len
    train = []

  # Training set creation
    for col in range(loops):
        train.append(D[:,col:col+x_len+y_len])
    train = np.vstack(train)
    X_train, Y_train = np.split(train,[-y_len],axis=1)

  # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(X_train,[-rows*test_loops],axis=0)
        Y_train, Y_test = np.split(Y_train,[-rows*test_loops],axis=0)
    else: # No test set: X_test is used to generate the future forecast
        X_test = D[:,-x_len:]
        Y_test = np.full((X_test.shape[0],y_len),np.nan) 

  # Formatting required for scikit-learn
    if y_len == 1:
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()
    return X_train, Y_train, X_test, Y_test