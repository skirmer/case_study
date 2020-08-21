# Clean Data
# This script prepares data for the R markdown EDA and for the Jupyter notebook modeling

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
import xgboost as xgb
import math
import pandas as pd

# functions


def one_hot_encode(df, feature):
    # Onehots
    values = np.array(df[feature], dtype=str)
    values2 = values.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(values2)
    cols = onehot_encoder.get_feature_names()

    def newfun(strval):
        return(strval.replace(" ", "_"))
    newfunct = np.vectorize(newfun)
    cols = newfunct(cols)

    df2 = pd.DataFrame(onehot_encoded, columns=cols)
    df = pd.concat([df, df2], axis=1)
    return(df)


def cv_model(dataframe, feature_list):
    # split train/test
    trainingset = dataframe[feature_list]
    trainingset.dropna(inplace=True)
    feature_names = trainingset.iloc[:, 1:].columns.to_list()

    X = trainingset.iloc[:, 1:].values
    y = trainingset.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    # cross validation, select hyperparams
    param = {'objective': 'reg:squarederror', 'booster': 'gbtree'}
    gridparam = {'max_depth': [6, 8], 'eta': [0.01, 0.05]}

    fitob = xgb.XGBRegressor(**param)
    clf = GridSearchCV(fitob, gridparam)
    gridresults = clf.fit(X, y)

    kfold = KFold(n_splits=5, random_state=42)
    cv = cross_validate(clf, X, y, cv=kfold)

    cv_results = pd.DataFrame(clf.cv_results_)
    return(dtrain, dtest, cv_results, X_train, X_test, y_train, y_test)


def train_model(dtrain, dtest, eta, max_depth):
    # train model
    param = {'max_depth': max_depth, 'eta': eta,
             'objective': 'reg:squarederror', 'booster': 'gbtree'}

    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 300
    loaded_model = xgb.train(param, dtrain, num_round, watchlist)
    return(loaded_model)


def predict_eval(dtrain, dtest, y_train, y_test, modelobj):
    # predict, evaluate
    preds_y = modelobj.predict(dtest)
    preds_train = modelobj.predict(dtrain)

    test_residuals = np.absolute(y_test - preds_y)
    train_residuals = np.absolute(y_train - preds_train)

    # RMSE - train set
    print("RMSE, training:")
    print(math.sqrt(mean_squared_error(y_train, preds_train)))

    # RMSE - test set
    print("RMSE, testing:")
    print(math.sqrt(mean_squared_error(y_test, preds_y)))

    # Residuals - train set
    print("Median residuals, training:")
    print(np.median(train_residuals))

    # Residuals - test set
    print("Median residuals, testing:")
    print(np.median(test_residuals))

    return(preds_y, preds_train, train_residuals, test_residuals)


# Load data - Open University Online Learning Files
assessment = pd.read_csv("../data/raw/distance_learning/assessments.csv")
courses = pd.read_csv("../data/raw/distance_learning/courses.csv")
st_assessment = pd.read_csv(
    "../data/raw/distance_learning/studentAssessment.csv")
st_info = pd.read_csv("../data/raw/distance_learning/studentInfo.csv")
st_regis = pd.read_csv("../data/raw/distance_learning/studentRegistration.csv")
st_vle = pd.read_csv("../data/raw/distance_learning/studentVle.csv")
vle = pd.read_csv("../data/raw/distance_learning/vle.csv")

##### Analytic planning #####

# Looks like courses dataset is not that pertinent- I am going to guess that the length of course is not a huge factor in student performance. The values are not widely varied, range is just 234-269

### feature engineering ###

# VLE bits
vle_plus = st_vle.merge(
    vle, left_on=['code_presentation', 'id_site', 'code_module'], right_on=['code_presentation', 'id_site', 'code_module'])

new_vle_plus = (
    vle_plus.groupby(['id_student', 'code_presentation',
                      'code_module', 'activity_type'])
    .agg(
        {'sum_click': [min, max, 'mean', 'std', 'median', sum]}
    )
).dropna()

new_vle_plus.columns = new_vle_plus.columns.get_level_values(
    0) + new_vle_plus.columns.get_level_values(1)
new_vle_plus = new_vle_plus.reset_index()

df = new_vle_plus.pivot_table(index=['id_student', 'code_presentation', 'code_module'],
                              columns='activity_type', values=['sum_clicksum'], aggfunc='sum')

df.columns = df.columns.get_level_values(
    0) + '_' + df.columns.get_level_values(1)
df = df.reset_index().replace(np.nan, 0)


# Add student personal traits to the assessment scores
st_personal = st_info[["id_student", "gender", "region",
                       "highest_education", "imd_band", "age_band", "disability", "num_of_prev_attempts", "studied_credits"]]
st_personal.drop_duplicates(inplace=True)
st_assess_plus = st_assessment.merge(
    st_personal, left_on='id_student', right_on='id_student')

# Add some additional details about the assessments
st_assess_plus = st_assess_plus.merge(
    assessment, left_on='id_assessment', right_on='id_assessment')

# Finally add student activity
st_assess_act = st_assess_plus.merge(
    df, left_on=['id_student', 'code_presentation', 'code_module'], right_on=['id_student', 'code_presentation', 'code_module'])


# Question: do student personal traits have a meaningful impact on the assessment outcomes? Are they more meaningful than the date, for example?

df = st_assess_act
# One hot the categoricals

df = one_hot_encode(df, "gender")
df = one_hot_encode(df, "region")
df = one_hot_encode(df, "disability")
df = one_hot_encode(df, "code_module")
df = one_hot_encode(df, "code_presentation")
df = one_hot_encode(df, "assessment_type")

# ordinal encode others

df['highest_education_ordinal'] = df.highest_education.replace(
    ['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification'], [1, 2, 3, 4, 5])

df['age_band_ordinal'] = df.age_band.replace(
    ['55<=', '35-55', '0-35'], [3, 2, 1])

df['imd_band_ordinal'] = df.imd_band.replace(
    ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%', '30-40%', '20-30%', '10-20', '0-10%'], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

df.to_csv("../data/clean/features.csv")
