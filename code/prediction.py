# Case Study Take-home
# Basic prediction/recommendation
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import math
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


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

# Note: after completing, I realized this split methodology has unwanted leakage due to individuals potentially being in both train and test for different courses.


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

    kfold = KFold(n_splits=2, random_state=42)
    cv = cross_validate(clf, X, y, cv=kfold)

    cv_results = pd.DataFrame(clf.cv_results_)
    return(dtrain, dtest, cv_results, X_train, X_test, y_train, y_test)


def train_model(dtrain, dtest):
    # train model
    param = {'max_depth': 6, 'eta': 0.05,
             'objective': 'reg:squarederror', 'booster': 'gbtree'}

    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 200
    loaded_model = xgb.train(param, dtrain, num_round, watchlist)
    return(loaded_model)


def predict_eval(dtrain, dtest, y_train, y_test):
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

    return(preds_y, preds_train)


# Load data
assessment = pd.read_csv("./distance_learning/assessments.csv")
courses = pd.read_csv("./distance_learning/courses.csv")
st_assessment = pd.read_csv("./distance_learning/studentAssessment.csv")
st_info = pd.read_csv("./distance_learning/studentInfo.csv")
st_regis = pd.read_csv("./distance_learning/studentRegistration.csv")
st_vle = pd.read_csv("./distance_learning/studentVle.csv")
vle = pd.read_csv("./distance_learning/vle.csv")

##### Analytic planning #####

# Looks like courses dataset is not that pertinent- I am going to guess that the length of course is not a huge factor in student performance. The values are not widely varied, range is just 234-269

### feature engineering ###

# Add student personal traits to the assessment scores
st_personal = st_info[["id_student", "gender", "region",
                       "highest_education", "imd_band", "age_band", "disability"]]
st_personal.drop_duplicates(inplace=True)
st_assess_plus = st_assessment.merge(
    st_personal, left_on='id_student', right_on='id_student')

# Add some additional details about the assessments
st_assess_plus = st_assess_plus.merge(
    assessment, left_on='id_assessment', right_on='id_assessment')

# Question: do student personal traits have a meaningful impact on the assessment outcomes? Are they more meaningful than the date, for example?

df = st_assess_plus
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

df['imd_band_ordinal_class'] = df.imd_band.replace(
    ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%', '30-40%', '20-30%', '10-20', '0-10%'], [5, 4, 4, 3, 3, 3, 2, 2, 2, 1])

# Outcome- does score of zero mean they didn't take it? What's the situation? Distribution has a weird long tail on the low end, with a bump at zero. Not normally distributed at all...

# Computer marked ends up with a weirdly high proportion of 100s as well. CMA definitekly weird enough that it should be examined independently.

##### General EDA #####

# Ireland group is small
# IMD is poverty measure

pd.crosstab(df['imd_band'], df['age_band'], normalize='index')
pd.crosstab(df['x0_F'], df['age_band'], normalize='index')
pd.crosstab(df['region'], df['gender'], normalize='index')
pd.crosstab(df['imd_band'], df['gender'], normalize='index')


# correlation tests

feature_list = ['score', 'date',
                'x0_F', 'x0_Y', 'x0_2013B', 'x0_2013J', 'x0_2014B', 'x0_2014J', 'x0_CMA', 'imd_band_ordinal_class',
                'x0_Exam', 'highest_education_ordinal', 'imd_band_ordinal', 'age_band_ordinal']

feature_list = ['score', 'date',
                'x0_F', 'x0_Y', 'x0_AAA', 'x0_BBB', 'x0_CCC', 'x0_DDD', 'x0_EEE', 'x0_FFF', 'x0_GGG',
                'x0_Exam', 'highest_education_ordinal', 'imd_band_ordinal', 'age_band_ordinal']


feature_list = ['score', 'x0_CMA', 'x0_2014J', 'x0_F',
                'x0_DDD', 'imd_band_ordinal', 'x0_FFF', 'x0_GGG']

corrMatrix = df[feature_list].corr()
sn.heatmap(corrMatrix, annot=True)

# Computer scored tests seem to get higher scores
# FFF most likely to have computer scoring, least likely to have girls
# DDD has unusually low scores
# Girls more likely to take GGG

# TMA and CMA are multicollinear
# date and date submitted are essentially multicollinear


##### Modeling #####
# outcome: score
# features: assessment, date submitted, gender, region, highest ed, imd band, age band, disability, date

# Type of course seems top of the influence on score, followed by the time since the course was done, the student's prior education.

# Three types of assessments exist: Tutor Marked Assessment (TMA), Computer Marked Assessment (CMA) and Final Exam (Exam).

# Having a computer marked assessment is strangely very instrumental in student performance, but this might be collinear with some other traits of the course. Some topics are not suitable for computer assessment, and they might have other commonalities.

#"The structure of B and J presentations may differ and therefore it is good practice to analyse the B and J presentations separately. Nevertheless, for some presentations the corresponding previous B/J presentation do not exist and therefore the J presentation must be used to inform the B presentation or vice versa. In the dataset this is the case of CCC, EEE and GGG modules."

# Try a model split by test type

### Model 2 ###
feature_list = ['score',
                'x0_F', 'x0_East_Anglian_Region',
                'x0_East_Midlands_Region', 'x0_Ireland', 'x0_London_Region', 'x0_North_Region', 'x0_North_Western_Region', 'x0_Scotland',
                'x0_South_East_Region', 'x0_South_Region', 'x0_South_West_Region', 'x0_Wales', 'x0_West_Midlands_Region', 'x0_Yorkshire_Region', 'x0_N',
                # 'x0_AAA', 'x0_BBB', 'x0x_CCC', 'x0_EEE',
                'x0_DDD', 'x0_FFF', 'x0_GGG',
                'x0_2013B', 'x0_2013J', 'x0_2014B', 'x0_2014J',
                'x0_Exam', 'highest_education_ordinal', 'imd_band_ordinal', 'age_band_ordinal', 'date']

# Not Computer Marked
df2 = df[df['x0_CMA'] == 0.0]
# df2 = df[df['x0_Exam'] == 1.0] it overfits like crazy if I do this
df2 = df2[df2['score'] > 1.0]
# Interesting: irish region becomes top predictor among the features above

# Train and predict
dtrain, dtest, cv_results, X_train, X_test, y_train, y_test = cv_model(
    dataframe=df2, feature_list=feature_list)
cv_results

modelobj = train_model(dtrain, dtest)
preds_y, preds_train = predict_eval(dtrain, dtest, y_train, y_test)

result_train = pd.DataFrame(X_train, columns=feature_list[1:])
result_test = pd.DataFrame(X_test, columns=feature_list[1:])

result_train['predictions'] = preds_train
result_test['predictions'] = preds_y
result_train['truth'] = y_train
result_test['truth'] = y_test

result_test.to_csv("preds_test_m2.csv")
result_train.to_csv("preds_train_m2.csv")

# Feature importance
gainfeat = modelobj.get_score(importance_type='gain')
sorted(gainfeat, key=gainfeat.get, reverse=True)

modelobj.get_score(importance_type='weight')
