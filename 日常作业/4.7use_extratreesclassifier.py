# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 21:45:23 2017

@author: Mr.王
"""
import numpy as np
import pandas as pd
import time
import os
import uuid

### 0.定义函数及公用变量、文件等
file_print_to = open("file_print_to1.txt", 'a')
print('\n\n\n--****-- 本次实验开始时间：' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), file=file_print_to)
print('-------- 实验计算机名：' + str(os.environ['COMPUTERNAME'] \
        + '   MAC地址：' + str(uuid.UUID(int = uuid.getnode()).hex[-12:])), file=file_print_to)
data_file = 'feature_offline_in15days_2018_01_06.csv'
print('-------- 实验数据文件是：' + data_file, file=file_print_to)

### 1.读取数据，划分训练集（train）和验证集（verify）
print('1.读取数据，划分训练集（train）和验证集（verify）')
features = pd.read_csv(data_file)

features_train = features[features.Date_received <= 20160501]
features_verify = features[(features.Date_received >= 20160516) & \
                          (features.Date_received <= 20160616)]

### 2.指定训练用的特征，生成训练特征集（X_train）、训练标签集（y_train）、验证特征集（X_verify）、验证标签集（y_verify）。
print('2.指定训练用的特征，生成训练特征集（X_train）、训练标签集（y_train）、验证特征集（X_verify）、验证标签集（y_verify）')
fe_parameters = ['User_id', 'Merchant_id', \
                          #-------- 用户、优惠券相关特征 --------
                          'distance', 'discount_man', 'discount_jian', 'discount_rate', \
                          'day_of_week', 'is_weekend', 'day_of_month', \
                          #-------- 商户相关特征 --------
                          'total_sales', 'sales_use_coupon', 'total_coupons', \
                          'use_coupon_rate', 'transfer_rate', 'merchant_max_distance', \
                          'merchant_min_distance', 'merchant_mean_distance' \
                          ]
print('采用' + str(len(fe_parameters)) + '个特征：' +  ','.join(fe_parameters))
print('采用' + str(len(fe_parameters)) + '个特征：' +  ','.join(fe_parameters), file=file_print_to)
X_train = features_train[fe_parameters]
y_train = np.ravel(features_train[['coupon_apply']])
X_verify = features_verify[fe_parameters]
y_verify = np.ravel(features_verify[['coupon_apply']])


### 3.应用ExtraTreesClassifier算法
print('3.应用ExtraTreesClassifier算法')
print('\n\n------ExtraTreesClassifier预测', file=file_print_to)
time_start = time.time()

from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(random_state=2)
etc.fit(X_train, y_train)
print('训练用时：' + str(time.time() - time_start), file=file_print_to)

print('采用ExtraTreesClassifier预测的准确率：' + str(etc.score(X_verify, y_verify)), file=file_print_to)

print('\n各特征重要程度：', file=file_print_to)
print('\n各特征重要程度：', file=file_print_to)
print(list(zip(fe_parameters, map(lambda x: round(x, 4), etc.feature_importances_))), file=file_print_to)

file_print_to.close()


