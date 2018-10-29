# Home Credit Default Risk
## Project Description
This repository is a simplified version of my solution to Kaggle competition ["Home credit default risk"](https://www.kaggle.com/c/home-credit-default-risk#description). The competitors are asked to predict the Home Credit's clients repayment abilities, given customer's current application, as well as previous loan records, credit accounts information at other institutions and monthly payment data in the past. The predictions are evaluated on area under the ROC curve between the predicted probability and the observed target.

#### Dataset
* **application_{train|test}.csv**
Main table, broken into two files for Train (with TARGET) and Test (without TARGET).
* **bureau.csv**
All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
* **bureau_balance.csv**
Monthly balances of previous credits in Credit Bureau.
* **POS_CASH_balance.csv**
Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
* **credit_card_balance.csv**
Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
* **previous_application.csv**
All previous applications for Home Credit loans of clients who have loans in our sample.
* **installments_payments.csv**
Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.

#### Final score for the full solution:
Private LB: 0.80265
Public LB: 0.80835
Rank 17th/7198 (#1st of solo Kagger)

## Feature Engineering
#### Features from business intuition
I don't have any professional background in business, but it is still possible to generate new features from the given raw features based on our understanding of finacial ability. Some useful features are:
* loan payment length = credit amount / annuity
* difference between actual and expected monthly payment
* when is the last time a customer has payment past due
* ratio between credit usage and credit limit
* difference between actual and planned account close data

......

#### Statistics computed by grouping by accounts and months
The supplementary tables (previous application, bureau records, installment etc) cannot be directly merged to the main table, because clients have various number of previous loans and different length of credit history. Thus a lot of statistical features can be computed by first grouping by current application ID and averaging/summing over both different account and records of different months. Some statistics I computed includes:
* mean, sum, max, median, variance ...
* the above statistic functions calculated on subset of accounts, such as all active accounts, approved/refused applications, the most recent application...
* time scaled sum/mean (with more recent records weighted more), or regular statistics computed within a certain time window (such as within 3 years).

With these statistic features, single models have ~0.796 private LB, ~0.802 public LB and ~0.798 local CV, which would won a silver medal.

#### Features from training on each previous, bureau.
One pitfull of grouping by current application ID and then compute statistics is that sometimes it may be problematic to average/sum over different previous records, as one can expect some previous application may be more important than others, but it is hard to decide how do weigh each records. Although we have computed statistic on subset of previous applications as we have mentioned, this may still not be optimal and some information can be lost when we transform the raw data in the supplementary table to statistical features.
One alternative is to train a model directly using each previous record as a training sample,  and corresponding current target as training targets (so previous applications of a same customer will share the target of this customer's current target). We will have a traing set looks like:

|  current ID | previous ID | days decision  | amount_credit  |  .....   | target  | prediction |
| ---| ---| ----| --  | ----| --| ---- |
|   1|  1 |  .. |   ..| ..  | 1 | 0.44 |
|   1|  2 |  .. |  .. |  .. | 1 | 0.52 |
|   2|  3 |  .. |..   |  .. | 0 | 0.25 |
|   2|  4 |  .. |..   |  .. | 0 | 0.15 |
|   2|  5 |  .. |..   |  .. | 0 | 0.23 |
|   3|  6 |  .. |   ..| ..  | 1 | 0.61 |
|   3|  7 |  .. |  .. |  .. | 1 | 0.31 |
....

The features are those describing previous applications, so we can use the previous_application table directly, and for credit card, pos cash and installment tables, we will group by previous_id instead of grouping by current_id. This way, the model will find the correlation between a specific previous application and current probability of defaulting, or "what is the probability of a certain previous application belongs to someone who has defaulted loan currently". After we get the prediction for each previous application, we can do:

agg_prev_score = df.groupby('current_id')['prediction'].agg({'mean','max','sum'...})

The aggregated predictions are pretty good features, and we can merge it to our regular training set by the current ID. So instead of doing aggregations like mean/sum on previous applications as people normally do, we look at each previous application separately and aggregate later. This would give some complementary view on the dataset.

And we can do the same thing on each bureau record as well. These generated features gives ~0.003 boost of local CV, and are the main break through for winning a gold.

#### Features from training on each monthly records
Following the idea above, we can also group by different account first and train a model on each monthly records. However, one has to be careful because monthly records of a same loan are likely to share same values for some features -- such as the same amount of monthly payment. This may introduce a leak to our model as the model may start to find out records with a certain monthly payment all have a certain target. To avoid the leak, we will place records of same customer in the same fold while doing cross validation, so that early-stopping can be triggered when the model starts to exploit information that cannot be generalized to test set. If the special kfold is not applied, the generated features will give unrealistic boost in CV.

#### Features from time series
The installment payment, pos-cash, credit card and bureau balance tables contain time series information. In addition to the statistics we have already computed, I trained GRU networks on each of these four tables and extract the model prediction as features for final model training. The GRU network achieved 0.55-0.61 auc score during training.

#### Document and house features
In the main table there are ~20 features with value 0 or 1 descibing whether a certain document was provided in an application, as well as house features (scaled between 0 and 1) describing housing situation in applicant's residential area. I have simple logistic regression model trained only on these features and use the model prediction as features in final training.

#### Generic programming feature by [Scripus](https://www.kaggle.com/scirpus)
Some of my models use features generated by generic programming provided by Scripus. The generic features can by found in these kernels/threads:
* [Hybrid Jeepy and LGB](https://www.kaggle.com/scirpus/hybrid-jeepy-and-lgb)
* [Hybrid Jeepy and LGB II ](https://www.kaggle.com/scirpus/hybrid-jeepy-and-lgb-ii)
* [Pure GP with Mean Squared Error](https://www.kaggle.com/scirpus/pure-gp-with-mean-squared-error)
* [Pure GP with LogLoss](https://www.kaggle.com/scirpus/pure-gp-with-logloss)
* [Discussion: Pure GP](https://www.kaggle.com/c/home-credit-default-risk/discussion/62983)

Note the GP features seems to be prone to overfitting. In my experience, using GP features from the last three sources above give very high local CV (~0.807) but only
~0.802 public LB. 

## Modeling
#### Validation
I use stratified KFold cross-validation in all single models. With early stopping round set to be 100, and learning rate = 0.003 in LightGBM and xgboost classifiers.

Because of the unbalanced target (less than 10% accounts are defaulted in the training set), in some models I downsampled the major class: In each fold I divide the major class into, say 3, and average the results of three runs, where each run trains the all minor class samples with 1/3 major class samples.


#### Single models
Model diversity are essential for later ensembling. In this competition, with structured data being dominant, I don't have much success with neural networks, all my single models are gradient boosting decision trees using [LightGBM](https://lightgbm.readthedocs.io/en/latest/) and [xgboost](https://xgboost.readthedocs.io/en/latest/) packages.

On the other hand, to add diversity at the feature level, I created three groups of features, each group differs in minor aspects such as:
* use mean v.s. median v.s. mean of recent X years ...
* ways of weight on time: exponential decay or reciprocal of time.
* use mean encoding or lightgbm's build in categorical features

...

The three groups of features can be trained with LightGBM in gbdt, goss, dart modes as well as xgboost. Top 200 features (by lightGBM feature importance) are selected and trained along with different groups of ~500 GP features.

Eventually I had **27** of my own single models and **5** different runs of [neptune.ml](https://neptune.ml)'s [open solution](https://www.kaggle.com/c/home-credit-default-risk/discussion/57175) model.

#### Hyperparameter tunning
In about half way through the competition, I tried to optimized my model hyperparameter using this [Bayesian optimization package](https://github.com/fmfn/BayesianOptimization). The optimization I've done is very preliminary: I searched for only 50 rounds and did not retune the parameters after I added more features later. However, the bayesian optimization seems to show that shallow tree structure (depth=4/5) and feature fraction ~= 0.3 are two key parameters in this case.

#### Ensembling
I use weighted average to blend all my single models. The weights are tuned by **hand**: first determine the relative weights between open solution models and my own models, then fine tune the weights between the three feature groups and between lgb/xgb models. Ensembling gets **~0.002** boost in my final solution.

Because of the potential overfitting of GP features, it would be problematic to soly rely on local CV when determining the weight in ensembling. Eventually I had one submission without using models with GP features, and another submission that gives models with GP features a weight of 8%. The second submission has higher private LB score, indicating that although there may be overfitting present in models using GP features, the diversity they brought are still beneficial.

## What's not (fully) working?
Attempting to do some "pseudo data augmentations" using previous applications. Because the previous applications share some features with current application, we can build a model trained on previous applications and use its prediction of current applications as features. As the dataset provides 8 years of previous application results, I selected the most recent 3 years of previous applications, and each previous application can find 5 more years records prior to itself as it's features. The targets are set to be things like "whether a previous application has days past due" or "average underpayment of each previous application". The generated features have a high feature importance in the final training but did not give much improvement on the auc score, probably because they did not add new information to our model.

## Note on repository.
As said, this is a simplified version of my final solution, I only include three representive single models. Also, in the full solution I have slightly different versions of previous/bureau training to add diversity on these features, while here I am using the same features from previous/bureau training in all three single models.

In order to run the code, first download dataset to the input folder: if you have [Kaggle API](https://github.com/Kaggle/kaggle-api), you can run "kaggle competitions download -c home-credit-default-risk". The notebooks should be run in the following order:
1. Preprocessing features
	1. prev_training.ipynb (training on each previous application)
	1. buro_training.ipynb (training on each bureau record)
	1. month_training.ipynb (training on each monthly record)
	1. house-doc-feats.ipynb (house and document features)
	1. inst-ts.ipynb, bubl-ts.ipynb, pos-ts.ipynb, cc-ts.ipynb (time series features)
1. Single models
	1. lgb1.ipynb, lgb2.ipynb, lgb3.ipynb
1. Ensembling
	1. ensembling.ipynb
	
All intermediate and final outputs are saved in the output folder.


#### scores

|   |  Private LB | Public LB  | Local CV  |
| ------------ | ------------ | ------------ | ------------ |
| LightGBM 1  |  0.7998 | 0.8050  | 0.8020  |
| LightGBM 2  | 0.8003 | 0.8042  | 0.8018  |
| LightGBM 3  | 0.7992 | 0.8028  | 0.8012  |
|Blended solution | 0.8004| 0.8046 | 0.8028 |

#### prerequisite:
python3, pandas, sklearn, matplotlib, seaborn, keras, tensorflow, lightgbm.

## Acknowledgement
* I would like to thank [neptune.ml](https://neptune.ml) for providing the [open solution](https://www.kaggle.com/c/home-credit-default-risk/discussion/57175) that makes ~25% of my final blended model.
* I would like to thank [Scripus](https://www.kaggle.com/scirpus) for his generic programming features. These features add valuable diversity to my solution and I learned the idea of generic programming for the first time.
* Thanks [Oliver](https://www.kaggle.com/ogrellier) for his script and early stage. As a newbee to Kaggle competition, I used his script as baseline model, it helped me get started quickly.
* Thanks all fellow kagglers for helpful discussions and wonderful ideas.

