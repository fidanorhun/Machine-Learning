# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:06:29 2021

@author: Orhun Fidan
"""
#Kütüphane atamaları
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold,cross_val_score,train_test_split,LeaveOneOut
import time


#Verinin okunması
veriler = pd.read_csv('ReplicatedAcousticFeatures-ParkinsonDatabase (2).csv')




minmax_scale = preprocessing.MinMaxScaler().fit(veriler.iloc[:,3:48])
df_minmax = minmax_scale.transform(veriler.iloc[:,3:48])

 


#Verinin bağımsız değişken ve bağımlı olarak bölünmesi
x=df_minmax#Bağımsız Değişkenler DataFrame Pandas
y=veriler.iloc[:,2]#Bağımlı Değişkenler DataFrame Pandas

Y=y.values #Numpy array formunda
"X=x.values"

start = time.time()
print(veriler.corr()) #Yakınlık matrisi oluşturmak için kullanılır

#Veri setinin bölünmesi
x_train, x_test,y_train,y_test = train_test_split(x,Y,test_size=0.33, random_state=0) #Genel olarak en yüksek doğruluk oranı olan 33 67 olarak seçilmiştir

#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

model=sm.OLS(regressor.predict(x_test),x_test)
print(model.fit().summary())
print("Linear R2 Değeri")

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
sc1=StandardScaler() 
x_olcekli = sc1.fit_transform(x) 
sc2=StandardScaler() 
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


#svr regression
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

print('SVR OLS')
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())


#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x_test,y_test)

print('Decision Tree OLS')
model4=sm.OLS(r_dt.predict(x_train),x_train)
print(model4.fit().summary())

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(x_test,y_test.ravel())


print('Random Forest OLS')
model5=sm.OLS(rf_reg.predict(x_train),x_train)
print(model5.fit().summary())

#Sınıflandırma Algoritmaları

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=1000)
logr.fit(x_train,y_train)

y_pred = logr.predict(x_test)
print(y_pred)
print(y_test)


cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Linear")
print(cm)
print(accuracy)

scores=cross_val_score(logr,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(logr,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(logr,x_test,y_pred,cv=loo)
print(scores.mean())

#En yakın komşu algoritması(1 neighbors)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("K-NN-1")
print(cm)
print(accuracy)

scores=cross_val_score(knn,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(knn,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(knn,x_test,y_pred,cv=loo)
print(scores.mean())

#En yakın komşu algoritması(3 neighbors)
from sklearn.neighbors import KNeighborsClassifier

knn3 = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("K-NN-3")
print(cm)
print(accuracy)

scores=cross_val_score(knn3,x_test,y_pred,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(knn3,x,y,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(knn3,x_test,y_pred,cv=loo)
print(scores.mean())

#En yakın komşu algoritması(5 neighbors)
from sklearn.neighbors import KNeighborsClassifier

knn5 = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("K-NN-5")
print(cm)
print(accuracy)


scores=cross_val_score(knn5,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(knn5,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(knn5,x_test,y_pred,cv=loo)
print(scores.mean())


#Support Vector Machine sigmoid
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('SVC-Sigmoid')
print(cm)
print(accuracy)


scores=cross_val_score(svc,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(svc,x,y,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(svc,x_test,y_pred,cv=loo)
print(scores.mean())


#Support Vector Machine rbf
from sklearn.svm import SVC
svcrbf = SVC(kernel='rbf')
svcrbf.fit(x_train,y_train)

y_pred = svcrbf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('SVC-rbf')
print(cm)
print(accuracy)

scores=cross_val_score(svc,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(svcrbf,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(svcrbf,x_test,y_pred,cv=loo)
print(scores.mean())

#Support Vector Machine linear
from sklearn.svm import SVC
svcl = SVC(kernel='linear')
svcl.fit(x_train,y_train)

y_pred = svcl.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('SVC-linear')
print(cm)
print(accuracy)

scores=cross_val_score(svcl,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(svcl,x,y,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(svcl,x_test,y_pred,cv=loo)
print(scores.mean())


#Support Vector Machine linear
from sklearn.svm import SVC
svcpoly = SVC(kernel='poly')
svcpoly.fit(x_train,y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('SVC-poly')
print(cm)
print(accuracy)

scores=cross_val_score(svcpoly,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(svcpoly,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(svcpoly,x_test,y_pred,cv=loo)
print(scores.mean())

#Naive Bayes GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)
print(accuracy)

scores=cross_val_score(gnb,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(gnb,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(gnb,x_test,y_pred,cv=loo)
print(scores.mean())

#Naive Bayes BernoulliNB
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('BNB')
print(cm)
print(accuracy)

scores=cross_val_score(bnb,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(bnb,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(bnb,x_test,y_pred,cv=loo)
print(scores.mean())

#Karar Ağacı entropy
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('DTC entropy')
print(cm)
print(accuracy)

scores=cross_val_score(dtc,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(dtc,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(dtc,x_test,y_pred,cv=loo)
print(scores.mean())

#Karar Ağacı gini
from sklearn.tree import DecisionTreeClassifier
dtcgini = DecisionTreeClassifier(criterion = 'gini')

dtcgini.fit(x_train,y_train)
y_pred = dtcgini.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('DTC gini')
print(cm)
print(accuracy)

scores=cross_val_score(dtcgini,x,y,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(dtcgini,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(dtcgini,x_test,y_pred,cv=loo)
print(scores.mean())

#Random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')#n_estimators=10 ağaç sayısı
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('RFC')
print(cm)
print(accuracy)

scores=cross_val_score(rfc,x_test,y_pred,cv=10)
print(scores)

print(scores.mean())

kfold=KFold(n_splits=3,shuffle=True,random_state=0)
print(cross_val_score(rfc,x_test,y_pred,cv=kfold))

loo=LeaveOneOut()
scores=cross_val_score(rfc,x_test,y_pred,cv=loo)
print(scores.mean())
end = time.time()

print(f"Runtime of the program is {end - start}")

