import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool, cv
from catboost.datasets import titanic


def build_titanic_dataset():
    df_train, df_test = titanic()
    df_train.fillna(-999, inplace=True)
    df_test.fillna(-999, inplace=True)
    X = df_train.drop('Survived', axis=1)
    y = df_train.Survived
    # if the feature value type != float, then be treat as a categorial feature
    cate_feat_idx = np.where(X.dtypes != np.float)[0]
    x_train, x_vali, y_train, y_vali = train_test_split(X, y, train_size=0.75, random_state=42)
    x_test = df_test
    return X, y, x_train, x_vali, y_train, y_vali, x_test, cate_feat_idx


def catboost_cv_eval(X, y, cate_feat_idx, params):
    print('start catboost model cv eval...')
    cv_data = cv(Pool(X, y, cat_features=cate_feat_idx), params, nfold=5)
    print('Best vali acc: {:.2f}±{:.2f} on step {}'.format(np.max(cv_data['test-Accuracy-mean']), cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])], np.argmax(cv_data['test-Accuracy-mean'])))
    print('Precise vali acc: {}'.format(np.max(cv_data['test-Accuracy-mean'])))


def catboost_train_evel(x_train, y_train, x_vali, y_vali, x_test, params, cate_feat_idx):
    print('=============================================')
    print('catboost model training...')
    train_pool = Pool(x_train, y_train, cat_features=cate_feat_idx)
    vali_pool = Pool(x_vali, y_vali, cat_features=cate_feat_idx)
    model = CatBoostClassifier(**params, task_type='CPU')       # sometimes GPU slower then CPU
    model.fit(train_pool, eval_set=vali_pool)

    print('=============================================')
    print('catboost vali acc: {:06.4f}'.format(accuracy_score(y_vali, model.predict(x_vali))))

    print('=============================================')
    print('catboost model training parameters:')
    for k, v in params.items():
        print('{:15}: {}'.format(k, v))

    print('=============================================')
    print('catboost model predicting...')
    test_pred_result = model.predict(x_test)
    test_pred_prob = model.predict_proba(x_test)
    print(test_pred_result[:10])
    print(test_pred_prob[:10])

    print('=============================================')
    print('catboost feature importances evaluate...')
    feat_importances = model.get_feature_importance(train_pool)
    feat_names = x_train.columns
    feat_importances_df = pd.DataFrame()
    feat_importances_df['feat'] = feat_names
    feat_importances_df['score'] = feat_importances
    feat_importances_df.sort_values(['score'], ascending=False, inplace=True)
    feat_importances_df = feat_importances_df.reset_index(drop=True)
    print(feat_importances_df)


def test_catboost():
    X, y, x_train, x_vali, y_train, y_vali, x_test, cate_feat_idx = build_titanic_dataset()
    params = {
        'iterations': 2000,
        'learning_rate': 0.01,
        'eval_metric': 'Accuracy',
        'logging_level': 'Verbose',
        'loss_function': 'Logloss',
        'use_best_model': True
    }
    # catboost_cv_eval(X, y, cate_feat_idx, params)
    catboost_train_evel(x_train, y_train, x_vali, y_vali, x_test, params, cate_feat_idx)

if __name__ == '__main__':
    test_catboost()


