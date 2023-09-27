from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from django.core.files.storage import default_storage
from .models import File


def forecast(req):
    if req.method == "POST":
        # preparing the excel file
        excel_file = req.FILES["excel_file"]
        data_frame = pd.read_excel(excel_file)
        data_frame[data_frame.columns[0]] = pd.to_datetime(
            data_frame[data_frame.columns[0]])
        series_data_frame = data_frame.set_index(data_frame.columns[0])[
            data_frame.columns[1]].resample('D').sum()
        X_train, Y_train, X_test, Y_test = datasets(series_data_frame)

        # creting the method them feed the excel file data
        reg_model = LinearRegression()
        reg_model = reg_model.fit(X_train, Y_train)

        # predcitng with trained model
        Y_test_pred = reg_model.predict(X_test)

        # calculating the accuracy
        accuracy = ((abs(np.sum(Y_test)) - abs(np.sum((Y_test -
                    Y_test_pred)))) / abs(np.sum(Y_test))) * 100

        # genreating the future prediction
        future_pred = reg_model.predict(
            [list(series_data_frame.iloc[len(series_data_frame)-30:])])
        X_future = list(series_data_frame.iloc[len(series_data_frame)-30:])
        X_future.append(future_pred[0])
        for i in range(29):
            future_pred = reg_model.predict(
                [list(X_future[len(X_future)-30:])])
            X_future.append(future_pred[0])

        # formating the history data and future data in pandas series format
        date = series_data_frame.index[len(series_data_frame)-31]
        date = pd.date_range(date, periods=61, freq='D', inclusive="neither")
        result_data = Y_test_pred.tolist() + X_future[30:]
        future_series = pd.Series(result_data, index=date)

        return JsonResponse({
            "status": True,
            "message": "forecasted the excel file successfully",
            "data": {
                "history": format_data(series_data_frame),
                "future": format_data(future_series),
                "accuracy": accuracy
            }
        }, status=200)
    else:
        return JsonResponse({
            "status": False,
            "message": "wrong method"
        }, status=405)


# splitting the data set to training and testing
def datasets(df, x_len=30, test_loops=30):
    X_train = []
    Y_train = []
    # creating the training set
    for index in range(x_len, len(df) - test_loops):
        X_train.append(list(df.iloc[index - x_len:index].values))
        Y_train.append(df.iloc[index])

    X_test = []
    Y_test = []
    # creating the testing set
    for index in range(len(df) - test_loops, len(df)):
        X_test.append(list(df.iloc[index - x_len:index].values))
        Y_test.append(df.iloc[index])

    return X_train, Y_train, X_test, Y_test

# format the pandas series object to json format


def format_data(series):
    labels = series.index.astype(str).to_list()
    values = series.values.astype(str)
    result_array = []
    for label, value in zip(labels, values):
        result_array.append({"x": label, "y": value})
    return result_array

def upload(req):
    if req.method == "POST":
        uploaded_file = req.FILES.get('excel_file')
        if uploaded_file is not None:
            File.objects.create(file=uploaded_file, owner=req.user)
            return JsonResponse({
                "status": True,
                "message": "تم رفع الملف بنجاح",
            })
        else:
            return JsonResponse({
            "status": False,
            "message": "لم تقم بارفاق ملف"
        }, status=400)
    else:
        return JsonResponse({
            "status": False,
            "message": "wrong method"
        }, status=405)
