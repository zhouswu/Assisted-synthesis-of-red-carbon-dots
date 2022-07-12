import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from numpy.random import seed
seed(111)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate,StratifiedKFold,cross_val_predict
from sklearn.decomposition import PCA
import os
import warnings
from scipy import stats
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
xgboost.set_config(verbosity=0)
sns.set(style='darkgrid')
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus'] = False



if __name__=="__main__":
    df = pd.read_excel("./data/train_data.xls",header=None)
    x = df.values[0:152,0:-1]
    y = df.values[:,-1]
    xgb = XGBClassifier(n_estimators=70,max_depth=4,learning_rate=0.2,use_label_encoder=False)
    xgb.fit(x, y)
    x_xgb = xgb.apply(x)
    encoder_onehot = OneHotEncoder()
    encoder_onehot.fit(x_xgb)
    x_xgb_encoded = np.array(encoder_onehot.transform(x_xgb).toarray())
    pca = PCA(n_components=52)
    data_pca = pca.fit_transform(x_xgb_encoded)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    score = ["roc_auc","f1_micro"]

    # DT
    dt_params = {"max_depth":8}
    dt = DecisionTreeClassifier(**dt_params)
    acc_list_dt = cross_validate(estimator=dt,X=data_pca,y=y,cv=kfold,
                              scoring=score,return_train_score=True)
    auc_dt_average = np.mean(acc_list_dt["test_roc_auc"])
    f1_dt_average = np.mean(acc_list_dt["test_f1_micro"])
    print("dt_auc_average：{}".format(auc_dt_average))
    print("dt_f1_average：{}".format(f1_dt_average))

    # KNN
    knn_params = {"n_neighbors":5}
    knn = KNeighborsClassifier(**knn_params)
    acc_list_knn = cross_validate(estimator=knn,X=data_pca,y=y,cv=kfold,
                              scoring=score,return_train_score=True)
    auc_knn_average = np.mean(acc_list_knn["test_roc_auc"])
    f1_knn_average = np.mean(acc_list_knn["test_f1_micro"])
    print("knn_auc_average：{}".format(auc_knn_average))
    print("knn_f1_average：{}".format(f1_knn_average))

    # RF
    rf_params = {"n_estimators":80,"max_depth":5}
    rf = RandomForestClassifier(**rf_params)
    acc_list_rf = cross_validate(estimator=rf,X=data_pca,y=y,cv=kfold,
                              scoring=score,return_train_score=True)
    auc_rf_average = np.mean(acc_list_rf["test_roc_auc"])
    f1_rf_average = np.mean(acc_list_rf["test_f1_micro"])
    print("rf_auc_average：{}".format(auc_rf_average))
    print("rf_f1_average：{}".format(f1_rf_average))

    # svc
    svc_params = {"C":100, "kernel":'linear'}
    svc = SVC(**svc_params)
    acc_list_svc = cross_validate(estimator=svc,X=data_pca,y=y,cv=kfold,
                              scoring=score,return_train_score=True)
    auc_svc_average = np.mean(acc_list_svc["test_roc_auc"])
    f1_svc_average = np.mean(acc_list_svc["test_f1_micro"])
    print("svc_auc_average：{}".format(auc_svc_average))
    print("svc_f1_average：{}".format(f1_svc_average))

    # xgboost
    XGB_params = {'n_estimators':100,'max_depth':5,'learning_rate':0.15,'use_label_encoder':False}
    XGBOOST = XGBClassifier(**XGB_params)
    acc_list_XGB = cross_validate(estimator=XGBOOST, X=data_pca, y=y, cv=kfold,
                                  scoring=score, return_train_score=True)
    auc_xgboost_average = np.mean(acc_list_XGB["test_roc_auc"])
    f1_xgboost_average = np.mean(acc_list_XGB["test_f1_micro"])
    print("xgboost_auc_average：{}".format(auc_xgboost_average))
    print("xgboost_f1_average：{}".format(f1_xgboost_average))

    # lr
    logistic_regression_params = {"penalty": "l1", "solver":"saga","max_iter": 50, "C": 0.5}
    logistic_regression = LogisticRegression(**logistic_regression_params)
    acc_list_lr = cross_validate(estimator=logistic_regression,X=data_pca,y=y,cv=kfold,
                              scoring=score,return_train_score=True)
    auc_lr_average = np.mean(acc_list_lr["test_roc_auc"])
    f1_lr_average = np.mean(acc_list_lr["test_f1_micro"])
    print("lr_auc_average：{}".format(auc_lr_average))
    print("lr_f1_average：{}".format(f1_lr_average))

    # Confusion matrix
    LR = LogisticRegression(**logistic_regression_params)
    y_pred_cv = cross_val_predict(LR, data_pca, y, cv=kfold)
    confusion_mat_train = confusion_matrix(y, y_pred_cv,normalize="true")
    confusion_mat_train_df = pd.DataFrame(np.round(confusion_mat_train, 2),
    index=['Not Red', 'Red'], columns=['Not Red', 'Red'])
    sns.set(font_scale=1.8)
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    sns.heatmap(confusion_mat_train_df, cmap='Blues', annot=True, fmt='g')
    ax.set_ylabel('True Label', fontsize=18)
    ax.set_xlabel('Predicted Label', fontsize=18)
    ax.tick_params(axis='y', labelsize=18, labelrotation=45)
    ax.tick_params(axis='x', labelsize=18)
    ax.set_title('Confusion Matrix', fontsize=18)
    plt.show()

    # ROC
    y_pred_cv_pro = cross_val_predict(LR, data_pca, y, cv=kfold,method="predict_proba")
    fpr, tpr, _ = roc_curve(y, y_pred_cv_pro[:, 1], pos_label=1)
    auc = roc_auc_score(y, y_pred_cv)
    lw = 2
    plt.plot(fpr, tpr, 'k--', color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color=(0.6, 0.6, 0.6), lw=lw, linestyle='--', label='random guessing')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC Curve', fontsize=18)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    # Model comparison
    models = ['dt','knn','rf','svc','xgboost','lr']
    auc_hit = [auc_dt_average,auc_knn_average,auc_rf_average,auc_svc_average,auc_xgboost_average,auc_lr_average]
    f1_hit = [f1_dt_average,f1_knn_average,f1_rf_average,f1_svc_average,f1_xgboost_average,f1_lr_average]
    plt.bar(x=np.arange(6),height=f1_hit,width=0.3,label='f1-score')
    plt.bar(x=np.arange(6)+0.3, height=auc_hit, width=0.3,label='auc')
    plt.xticks(np.arange(6)+0.1,models)
    plt.legend()
    plt.show()

    # Data Analysis
    # Correlation analysis
    raw_x = pd.DataFrame(x)
    raw_coor = raw_x.iloc[:,0:5].corr(method='pearson')
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        raw_coor,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        annot=True,
        annot_kws={'fontsize': 18})
    plt.show()
    processed_x = pd.concat((pd.DataFrame(data_pca), pd.Series(y)), axis=1)
    processed_x.columns.values[-1] = 52
    processed_coor = processed_x.iloc[:,0:10].corr(method='pearson')
    processed_coor = processed_coor.round(0)
    sns.heatmap(
        processed_coor,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        annot=True,
        annot_kws={'fontsize': 15})
    plt.show()

    # Normality analysis
    fig = plt.figure(figsize=(10, 9))
    for i in range(8):
        f = fig.add_subplot(2, 4, i + 1)
        if (i < 4):
            stats.probplot(raw_x.iloc[:, i], plot=plt)
        else:
            stats.probplot(processed_x.iloc[:, i - 4], plot=plt)
    plt.show()

    # Scatter diagram of distribution
    for i in df.columns.tolist()[0:22]:
        sns.stripplot(x=22, y=i, data=df, jitter=True, palette="Set2", dodge=False)
        plt.show()
    columns_list = processed_x.columns.tolist()
    for i in columns_list[0:-1]:
        sns.stripplot(x=columns_list[-1], y=i, data=processed_x, jitter=True, palette="Set2", dodge=False)
        plt.show()

    # Model validation
    lr = LogisticRegression(**logistic_regression_params)
    lr.fit(X=data_pca, y=y)
    data = pd.read_excel('./data/val_data.xls', header=None)
    X = data.values
    design_ten = data.values[:, :-1]
    realcolor = data.values[:, -1]
    design_xgb = xgb.apply(design_ten)
    data_xgb = xgb.apply(x)
    data_combine = np.concatenate((data_xgb, design_xgb), axis=0)
    data_combine_encode = np.array(encoder_onehot.transform(data_combine).toarray())
    design_xgb_encode_pca = pca.fit_transform(data_combine_encode)
    resulttwo_lr = lr.predict(design_xgb_encode_pca[len(x):,:])
    print("Accuracy on validation set：{0}".format(accuracy_score(resulttwo_lr, realcolor)))