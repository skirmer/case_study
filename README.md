# Project Overview - Case Study

## Stephanie Kirmer
## August 2020


In this repo you'll find three files that comprise the analysis:

* cleaning.py: initial data ingestion, recoding, some cleaning. Functions used in recoding and modeling.
* eda.Rmd: visualizations of initial features and outcome, overview of data
* model_nb.ipynb: Jupyter notebook with modeling tasks. Imports cleaning.py to use data and functions.

This analysis uses the Open University Distance Learning dataset, and analyzed the factors that seem predictive of student performance on tests and assignments available.

* https://analyse.kmi.open.ac.uk/open_dataset
* http://www.open.ac.uk/
* https://www.nature.com/articles/sdata2017171  
* Zip File: https://drive.google.com/file/d/1nt1AZKIbcUGB4mNwXbmPMZT79I4DWTGs/view?usp=sharing 


In general, student behavior (use of course materials) seems strongest, with choice of course also distinguishing student outcomes. I have generated an XGBoost regression for this purpose, and a linear regression as a comparison model.

Interestingly, there are significant differences in course participation by gender, suggesting that perhaps marketing of certain subject areas could be used to encourage different enrollment trends. Further notes about correlations and differences identified in the dataset can be found in `eda.html` and in the model notebook.


## Things to investigate next:
* Certain courses (BBB and GGG) have very odd distributions of scores where computer marking is used. I'd like to investigate that and know more about why this is only the case in certain courses. There's something there that might indicate a serious failure of academic assessment, and it may have a disproportionate impact on female students because they are more likely to take one of the courses in question.  
* Delve into the experiences of the disabled participants in the program - they are underperforming, and I'd like to see if there's any more that can be known about their learning.
* Look at the relationships between different types of content clicks - there are a lot of content types noted in the student VLE file. Which ones really matter, and what actually do they contain?
* Deeper dive on the model results. I would like to look closer at the residuals from the XGBoost model for heteroscedasticity. I want to ensure the model is not returning a higher error rate for one end of the score spectrum, and didn't have time to check this here. 
* There is a large element of temporality that I think needs attention. There is data about the date of each assignment/test, as well as the dates of content being clicked, and in the interests of time I didn't separate these out. However, it's clear that content accessed after an assignment can't impact the assignment score. If a student increases their participation over the course of the term, that would be misleading in the current model.


## Other Data
Besides the data made available, there are a few areas that could add context.

* Investigate the regions, and incorporate summary statistics about the regions into model. https://www.ukdataservice.ac.uk/help/new-user/academic.aspx
* Find out if there are any deeper insights about the IMD system to be gained. How is IMD really measured? https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019
