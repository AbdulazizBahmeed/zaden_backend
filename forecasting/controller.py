import math
from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np
from .models import File
from django.core.exceptions import ObjectDoesNotExist
import requests
import uuid

from sklearn.linear_model import LinearRegression

# here we define the AI algortithms
import xgboost as xgb
XGB_regressor = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3,
                                 learning_rate=0.1, max_depth=100, alpha=10, n_estimators=140)

#decision tree model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
max_depth = list(range(5,11)) + [None]
min_samples_split = range(5,20)
min_samples_leaf = range(2,20)
param_dist = {'max_depth': max_depth,
'min_samples_split': min_samples_split,
 'min_samples_leaf': min_samples_leaf}
tree = DecisionTreeRegressor()
decision_tree = RandomizedSearchCV(tree, param_dist, n_jobs=-1, cv=5,verbose=1, n_iter=5, scoring='neg_mean_absolute_error')

#random forest model
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(bootstrap=True, max_samples=0.95,max_features=5, min_samples_leaf=18, max_depth=7)

#machine learning algorithm
from sklearn.neural_network import MLPRegressor
activation = 'relu'
solver = 'adam'
early_stopping = True
n_iter_no_change = 50
validation_fraction = 0.1
tol = 0.0001
param_fixed = {'activation':activation, 'solver':solver,'early_stopping':early_stopping, 'n_iter_no_change':n_iter_no_change, 'validation_fraction':validation_fraction,'tol':tol}
hidden_layer_sizes = [[neuron]*hidden_layer for neuron in range(10,60,10) for hidden_layer in range(2,7)]
alpha = [5,1,0.5,0.1,0.05,0.01,0.001]
learning_rate_init = [0.05,0.01,0.005,0.001,0.0005]
beta_1 = [0.85,0.875,0.9,0.95,0.975,0.99,0.995]
beta_2 = [0.99,0.995,0.999,0.9995,0.9999]
param_dist = {'hidden_layer_sizes':hidden_layer_sizes, 'alpha':alpha,'learning_rate_init':learning_rate_init, 'beta_1':beta_1, 'beta_2':beta_2}
NN = MLPRegressor(**param_fixed)
neural_network = RandomizedSearchCV(NN, param_dist, cv=5, verbose=1, n_jobs=-1, n_iter=5, scoring='neg_mean_absolute_error')

# here is the array of all the AI algortithms
algorithms = [LinearRegression(), XGB_regressor,decision_tree, random_forest, neural_network]


def upload(req):
    if req.method == "POST":
        uploaded_file = req.FILES.get('excel_file')
        if uploaded_file is not None:
            binary_file = {"file": uploaded_file.read()}
            identifier = uuid.uuid4()
            headers = {
                'Host': 'kpnzs85sk8.execute-api.ap-northeast-2.amazonaws.com',
            }
            url = f"https://kpnzs85sk8.execute-api.ap-northeast-2.amazonaws.com/upload-api/zaden-bucket/{identifier}.xlsx"
            response = requests.put(url, files=binary_file, headers=headers)
            if response.status_code == 200:
                File.objects.create(
                    file_name=uploaded_file.name, uuid=identifier, owner=req.user)
                return JsonResponse({
                    "status": True,
                    "message": "تم رفع الملف بنجاح",
                })
            else:
                return JsonResponse({
                    "status": False,
                    "message": "حصلت مشكلة أثناء عملية الرفع الرجاء المحاولة لاحقا"
                }, status=500)
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

def is_valid_excel(file):
    try:
        data_frame = pd.read_excel(file, engine='openpyxl')
    except:
        return JsonResponse({
            "status": False,
            "message": "الرجاء رفع ملف اكسل صحيح"
        }, status=400)

    try:
        data_frame[data_frame.columns[0]] = pd.to_datetime(
            data_frame[data_frame.columns[0]])
    except:
        return JsonResponse({
            "status": False,
            "message": "يجب ان يكون نوع اول عامود تاريخ ويمثل تاريخ المبيعات لكل صف"
        }, status=400)

    try:
        data_frame[data_frame.columns[1]] = pd.to_numeric(
            data_frame[data_frame.columns[1]])
    except:
        return JsonResponse({
            "status": False,
            "message": "يجب ان يكون نوع ثاني عامود عدد رقمي يمثل عدد المبيعات لكل صف"
        }, status=400)

    try:
        data_frame.set_index(data_frame.columns[0])[
            data_frame.columns[1]].resample('D').sum()
    except Exception:
        return JsonResponse({
            "status": False,
            "message": "حدث خطأ اثناء محاولة قراءة الملف الرجاء التاكد من صحة البيانات الموجودة في الملف حسب التنسيق المذكور"
        }, status=400)
    return None

def list_files(req):
    files_list = [file.as_dict() for file in req.user.files.all().order_by('-created_at')]
    return JsonResponse({
        "status": True,
        "message": "تم جلب البيانات بنجاح",
        "data": files_list
    })

def forecast(req, file_id):
    if req.method == "POST":
        period = req.GET.get('period')
        future_period = int(period) if period is not None else 30
        try:
            file = req.user.files.get(id=file_id)
        except ObjectDoesNotExist:
            return JsonResponse({
                "status": False,
                "message": "لايوجد ملف  بهذا المعرف"
            }, status=404)
        data_frame = pd.read_excel(file.file(), engine='openpyxl')
        # data_frame = pd.read_excel("C:\\Users\\azooz\\Downloads\\sales.xlsx")
        data_frame[data_frame.columns[0]] = pd.to_datetime(
            data_frame[data_frame.columns[0]])
        series_data_frame = data_frame.set_index(data_frame.columns[0])[
            data_frame.columns[1]].resample('D').sum()

        x_len = math.floor(len(series_data_frame) / 4)
        best_model, accuracy, Y_test_pred, X_future = best_model_analyzer(series_data_frame, x_len,future_period)

        # genreating the future prediction
        Y_future = best_model.predict(X_future)

        # formating the history data and future data in pandas series format
        date = series_data_frame.index[len(series_data_frame)- (future_period + 1)]
        date = pd.date_range(date, periods=(future_period * 2) + 1, freq='D', inclusive="neither")
        result_data = np.concatenate([Y_test_pred[0], Y_future[0]])
        future_series = pd.Series(result_data, index=date)
        print(int(np.sum(Y_future[0])))
        return JsonResponse({
            "status": True,
            "message": "forecasted the excel file successfully",
            "data": {
                "history": format_data(series_data_frame, future_period),
                "future": format_data(future_series, future_period),
                "accuracy": accuracy,
                "result": int(np.sum(Y_future[0]))
            }
        }, status=200)
    else:
        return JsonResponse({
            "status": False,
            "message": "wrong method"
        }, status=405)


def best_model_analyzer(series_df, x_len, future_period):
    X_train, Y_train, X_test, Y_test, X_future = dataset(series_df, x_len=x_len, y_len=future_period)
    best_accuracy = 0
    best_model = None

    for algorithm in algorithms:
        #train the model then predict
        trained_model = algorithm.fit(X_train, Y_train)
        Y_test_pred = trained_model.predict(X_test)
        # calculating the accuracy
        current_accuracy = (abs(np.sum(Y_test) - abs(np.sum((Y_test - Y_test_pred)))) / np.sum(Y_test)) * 100
        current_accuracy = (abs(np.sum(Y_test) - abs(np.sum((Y_test - Y_test_pred)))) / np.sum(Y_test)) * 100
        print(current_accuracy)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model = trained_model

    return best_model, best_accuracy, Y_test_pred, X_future


# splitting the data set to training and testing
def dataset(series_data_frame, x_len=30, y_len=30):
    series_length = len(series_data_frame)
    
    X_train = []
    Y_train = []
    #creating the training set
    for index in range(len(series_data_frame) - x_len - (y_len * 2) + 1):
        X_train.append(list(series_data_frame.iloc[index:index + x_len].values))
        Y_train.append(list(series_data_frame.iloc[index + x_len : index + x_len + y_len].values))

    X_test = []
    Y_test = []
    # # creating the testing set
    X_test.append(list(series_data_frame.iloc[series_length - x_len - y_len:series_length - y_len].values))
    Y_test.append(list(series_data_frame.iloc[series_length - y_len: ].values))
    
    X_future = []
    X_future.append(list(series_data_frame.iloc[series_length - x_len: ].values))

    return X_train, Y_train, X_test, Y_test, X_future


# format the pandas series object to json format
def format_data(series, future_period):
    series = series.resample("W").sum()
    labels = series.index.astype(str).to_list()
    values = series.values.astype(str)
    result_array = []
    for label, value in zip(labels, values):
        result_array.append({"x": label, "y": value})
    return result_array