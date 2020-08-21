# Valkyrie Take-home
# Basic prediction/recommendation
from patsy import dmatrices
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import math
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import cleaning
df = cleaning.df
df.columns

# Question: how much do student traits impact the score?
# Omitting GGG and BBB tests with CMA grading because they have such an odd distribution

df = df.query('x0_GGG + x0_CMA < 2 & x0_BBB + x0_CMA < 2')

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

feature_list = ['score', 'sum_clicksum_dataplus', 'sum_clicksum_dualpane',
                'sum_clicksum_externalquiz', 'sum_clicksum_folder',
                'sum_clicksum_forumng', 'sum_clicksum_glossary',
                'sum_clicksum_homepage', 'sum_clicksum_htmlactivity',
                'sum_clicksum_oucollaborate', 'sum_clicksum_oucontent',
                'sum_clicksum_ouelluminate', 'sum_clicksum_ouwiki', 'sum_clicksum_page',
                'sum_clicksum_questionnaire', 'sum_clicksum_quiz',
                'sum_clicksum_repeatactivity', 'sum_clicksum_resource',
                'sum_clicksum_sharedsubpage', 'sum_clicksum_subpage',
                'sum_clicksum_url']

feature_list = ['score', 'date',
                'x0_F', 'x0_Y', 'x0_AAA', 'x0_BBB', 'x0_CCC', 'x0_DDD', 'x0_EEE', 'x0_FFF', 'x0_GGG',
                'x0_Exam', 'highest_education_ordinal', 'imd_band_ordinal', 'age_band_ordinal']


feature_list = ['score', 'x0_CMA', 'x0_2014J', 'x0_F',
                'x0_DDD', 'imd_band_ordinal', 'x0_FFF', 'x0_GGG']

corrMatrix = df[feature_list].corr()
sn.heatmap(corrMatrix, annot=False)

# T-tests for groups

# Boys and girls - definitely different
stats.ttest_ind(df.query('x0_F == 0.0')['score'].to_numpy(), df.query(
    'x0_F == 1.0')['score'].to_numpy(), nan_policy="omit")

# disabled and not - different
stats.ttest_ind(df.query('x0_Y == 0.0')['score'].to_numpy(), df.query(
    'x0_Y == 1.0')['score'].to_numpy(), nan_policy="omit")

# computer marked test
stats.ttest_ind(df.query('x0_CMA == 0.0')['score'].to_numpy(), df.query(
    'x0_CMA == 1.0')['score'].to_numpy(), nan_policy="omit")

# Final exam
stats.ttest_ind(df.query('x0_Exam == 0.0')['score'].to_numpy(), df.query(
    'x0_Exam == 1.0')['score'].to_numpy(), nan_policy="omit")

# Prior education high
stats.ttest_ind(df.query('highest_education_ordinal > 3')['score'].to_numpy(), df.query(
    'highest_education_ordinal <= 3')['score'].to_numpy(), nan_policy="omit")

##### Modeling #####
# outcome: score
# features: student characteristics, assessment, date submitted, gender, region, highest ed, imd band, age band, disability, date
# Questioning whether any of the student personal traits rise to the top of the feature importance, outpacing things like the test itself

### Model 1 ###
outcome = 'score'
feature_list = ['score',
                'x0_F', 'x0_East_Anglian_Region',
                'x0_East_Midlands_Region', 'x0_Ireland', 'x0_London_Region', 'x0_North_Region', 'x0_North_Western_Region', 'x0_Scotland',
                'x0_South_East_Region', 'x0_South_Region', 'x0_South_West_Region', 'x0_Wales', 'x0_West_Midlands_Region', 'x0_Yorkshire_Region', 'x0_N', 'x0_AAA', 'x0_BBB', 'x0_CCC', 'x0_EEE',
                'x0_DDD', 'x0_FFF', 'x0_GGG',
                'x0_2013B', 'x0_2013J', 'x0_2014B', 'x0_2014J', 'x0_CMA', 'x0_Exam', 'highest_education_ordinal', 'imd_band_ordinal', 'age_band_ordinal', 'date',
                'sum_clicksum_dataplus', 'sum_clicksum_dualpane',
                'sum_clicksum_externalquiz', 'sum_clicksum_folder',
                'sum_clicksum_forumng', 'sum_clicksum_glossary',
                'sum_clicksum_homepage', 'sum_clicksum_htmlactivity',
                'sum_clicksum_oucollaborate', 'sum_clicksum_oucontent',
                'sum_clicksum_ouelluminate', 'sum_clicksum_ouwiki', 'sum_clicksum_page',
                'sum_clicksum_questionnaire', 'sum_clicksum_quiz',
                'sum_clicksum_repeatactivity', 'sum_clicksum_resource',
                'sum_clicksum_sharedsubpage', 'sum_clicksum_subpage',
                'sum_clicksum_url']


# Train and predict
dtrain, dtest, cv_results, X_train, X_test, y_train, y_test = cleaning.cv_model(
    dataframe=df, feature_list=feature_list)
cv_results

modelobj = cleaning.train_model(dtrain, dtest)

preds_y, preds_train, train_residuals, test_residuals = cleaning.predict_eval(
    dtrain, dtest, y_train, y_test, modelobj)

result_train = pd.DataFrame(X_train, columns=feature_list[1:])
result_test = pd.DataFrame(X_test, columns=feature_list[1:])

result_train['predictions'] = preds_train
result_test['predictions'] = preds_y
result_train['truth'] = y_train
result_test['truth'] = y_test
result_train['residual'] = train_residuals
result_test['residual'] = test_residuals
# Examine the character of the outcome to see whether the predictions are any use

result_train.residual.mean()
result_train.residual.std()
result_train.residual.median()

result_test.residual.mean()
result_test.residual.std()
result_test.residual.median()

df.score.mean()
df.score.std()
df.score.median()
mean_squared_error(y_train, preds_train)
mean_squared_error(y_test, preds_y)


result_test.to_csv("../data/clean/preds_test_m1.csv")
result_train.to_csv("../data/clean/preds_train_m1.csv")
# Import into R and visualize the predictions results

# Feature importance
gainfeat = modelobj.get_score(importance_type='gain')
sorted(gainfeat, key=gainfeat.get, reverse=True)

modelobj.get_score(importance_type='weight')


# LM tests as experiment


# add student traits .154 R2
y, X = dmatrices('score ~ gender + disability + date + date_submitted + region + code_module + code_presentation + assessment_type+ highest_education_ordinal+imd_band_ordinal+age_band_ordinal+sum_clicksum_dataplus + sum_clicksum_dualpane + sum_clicksum_externalquiz + sum_clicksum_folder + sum_clicksum_forumng + sum_clicksum_glossary + sum_clicksum_homepage + sum_clicksum_htmlactivity + sum_clicksum_oucollaborate + sum_clicksum_oucontent + sum_clicksum_ouelluminate + sum_clicksum_ouwiki + sum_clicksum_page + sum_clicksum_questionnaire + sum_clicksum_quiz + sum_clicksum_repeatactivity + sum_clicksum_resource + sum_clicksum_sharedsubpage + sum_clicksum_subpage + sum_clicksum_url + code_module*code_presentation + code_module*assessment_type + code_presentation*assessment_type', data=df, return_type='dataframe')

est = sm.OLS(y, X)
est2 = est.fit()
print(est2.summary())

preds_y2 = est2.predict(X)
X['preds'] = preds_y2
X['truth'] = y
X['residuals'] = np.absolute(X['truth'] - X['preds'])

X.residuals.mean()
X.residuals.std()
X.residuals.median()

X.truth.mean()
X.truth.std()
X.truth.median()
mean_squared_error(X['truth'], X['preds'])


# Clicks and academics - .12 R2
y, X = dmatrices('score ~ code_module + code_presentation + assessment_type+sum_clicksum_dataplus + sum_clicksum_dualpane + sum_clicksum_externalquiz + sum_clicksum_folder + sum_clicksum_forumng + sum_clicksum_glossary + sum_clicksum_homepage + sum_clicksum_htmlactivity + sum_clicksum_oucollaborate + sum_clicksum_oucontent + sum_clicksum_ouelluminate + sum_clicksum_ouwiki + sum_clicksum_page + sum_clicksum_questionnaire + sum_clicksum_quiz + sum_clicksum_repeatactivity + sum_clicksum_resource + sum_clicksum_sharedsubpage + sum_clicksum_subpage + sum_clicksum_url + code_module*code_presentation + code_module*assessment_type + code_presentation*assessment_type', data=df, return_type='dataframe')

est = sm.OLS(y, X)
est2 = est.fit()
print(est2.summary())

preds_y2 = est2.predict(X)
X['preds'] = preds_y2
X['truth'] = y
X['residuals'] = np.absolute(X['truth'] - X['preds'])

X.residuals.mean()
X.residuals.std()
X.residuals.median()

mean_squared_error(X['truth'], X['preds'])

# Just the course .065 R2
y, X = dmatrices('score ~ code_module + code_presentation + assessment_type+code_module*code_presentation + code_module*assessment_type + code_presentation*assessment_type', data=df, return_type='dataframe')

est = sm.OLS(y, X)
est2 = est.fit()
print(est2.summary())


# Things to do next:
# - determine what the deal is with GGG and BBB when computer scored. There's something there that might indicate a serious failure of academic assessment, and it's having a gender biased impact.
# - Delve into the experiences of the disabled participants in the program - they are underperforming and this doesn't have to be the case
# - Look at the relationships between different types of content clicks - which ones really matter, are some just irrelevant?
# - look closer at the residuals for heteroscedasticity- if we have uneven error at different score levels that could be a problem
# - work on the temporality- test date x vle date obviously related, not currently addressed
