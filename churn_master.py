# Importing the libraries
import math 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import matplotlib.pylab as pylab

from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.pyplot import figure



#Repeat CV 3 times, to reduce variance
NUM_TRIALS = 3
rows = np.around(np.logspace(math.log10(50),math.log10(10000),8))
#Sample size
rows = rows.astype(int)


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable - Geography
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Encoding the Independent Variable - Gender
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
 
# Normalization x_norm = (x- min(x))/(max(x)-min(x))
from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Set up possible values of parameters to optimize over
C_range = 2**np.arange(11)
gamma_range = np.logspace(-6, 1, 8)
kernels=['linear','rbf']
param_grid = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range},
          {'kernel': ['linear'], 'C': C_range}]
 
# Training the SVM model on the Training set
from sklearn.svm import SVC
svm = SVC()
 
# Arrays to store scores
non_nested_scores = np.zeros((len(rows),NUM_TRIALS))
nested_scores = np.zeros((len(rows),NUM_TRIALS))
non_nested_scores_loo = np.zeros(len(rows))
non_nested_scores_5_f = np.zeros((len(rows),NUM_TRIALS))

non_nested_times = np.zeros((len(rows),NUM_TRIALS))
nested_times = np.zeros((len(rows),NUM_TRIALS))
non_nested_times_loo = np.zeros(len(rows))
non_nested_times_5_f = np.zeros((len(rows),NUM_TRIALS))

non_nested_params = []
non_nested_params_5_f = []
non_nested_params_loo = []
nested_params = []

nested_cm = []
non_nested_cm_5_f = []
non_nested_cm_loo = []

nested_kappa = np.zeros((len(rows),NUM_TRIALS))
non_nested_kappa_loo = np.zeros(len(rows))
non_nested_kappa_5_f = np.zeros((len(rows),NUM_TRIALS))

nested_mcc = np.zeros((len(rows),NUM_TRIALS))
non_nested_mcc_loo = np.zeros(len(rows))
non_nested_mcc_5_f = np.zeros((len(rows),NUM_TRIALS))

nested_tpr = np.zeros((len(rows),NUM_TRIALS))
non_nested_tpr_loo = np.zeros(len(rows))
non_nested_tpr_5_f = np.zeros((len(rows),NUM_TRIALS))

# Loop for different amount of rows in the dataframe
j=0

for N in rows:
    X_sample = X[:N, ]
    y_sample = y[:N, ]
    print("Number of rows: ",N)
    
    param_column=[]
    nested_cm_column=[]
    non_nested_cm_5_f_column=[]
    
    
    # Loop for each trial
    for i in range(NUM_TRIALS):
        
        inner_three_f_cv = KFold(n_splits=3, shuffle=True, random_state=i)
        outer_five_f_cv = KFold(n_splits=5, shuffle=True, random_state=i)
        k_f_cv = KFold(n_splits=5, shuffle=True, random_state=i)

        
        # Non_nested parameter search and scoring
        start_time = time.time()
        
        clf = GridSearchCV(estimator=svm, param_grid=param_grid, cv=inner_three_f_cv, n_jobs=-1)
        clf.fit(X_sample, y_sample)
        y_pred_inner=cross_val_predict(clf.best_estimator_, X_sample, y_sample,cv=outer_five_f_cv, n_jobs=-1)
        
        #these non_nested values won't be used, but there is no harm in saving them
        non_nested_scores[j][i] = accuracy_score(y_sample,y_pred_inner)
        non_nested_times[j][i] = time.time() - start_time
                          
        
        # Nested CV with parameter optimization
        start_time = time.time()
        
        y_pred_outer = cross_val_predict(clf, X_sample, y_sample,cv=outer_five_f_cv, n_jobs=-1)
        
        nested_scores[j][i] = accuracy_score(y_sample,y_pred_outer)
        nested_times[j][i] = time.time() - start_time
        
        cm_i=confusion_matrix(y_true=y_sample,y_pred=y_pred_outer)
        nested_cm_column.append(cm_i.copy())
            
        TP = cm_i[1][1]
        FP = cm_i[0][1]
        FN = cm_i[1][0]
        TN = cm_i[0][0]
        TPR = TP/(TP+FN)
                
        nested_tpr[j][i]=TPR
        nested_mcc[j][i] = matthews_corrcoef(y_true=y_sample,y_pred=y_pred_outer)
        acc_random=(np.sum(cm_i[0])/np.sum(cm_i))**2+(np.sum(cm_i[1])/np.sum(cm_i))**2
        nested_kappa[j][i]=(nested_scores[j][i]-acc_random)/(1-acc_random)

        print(i,'nested \n',cm_i,nested_scores[j][i])
       
        
        # 5 Fold CV
        start_time = time.time()
        
        clf = GridSearchCV(estimator=svm, param_grid=param_grid, cv=k_f_cv, n_jobs=-1)
        clf.fit(X_sample, y_sample)
        y_pred_5_f=cross_val_predict(clf.best_estimator_, X_sample, y_sample,cv=outer_five_f_cv, n_jobs=-1)
        
        non_nested_scores_5_f[j][i] = accuracy_score(y_sample,y_pred_5_f)
        non_nested_times_5_f[j][i] = time.time() - start_time
        param_column.append(clf.best_params_.copy())
        
        cm_i=confusion_matrix(y_true=y_sample,y_pred=y_pred_5_f)
        non_nested_cm_5_f_column.append(cm_i.copy())
            
        TP = cm_i[1][1]
        FP = cm_i[0][1]
        FN = cm_i[1][0]
        TN = cm_i[0][0]
        TPR = TP/(TP+FN)
                
        non_nested_tpr_5_f[j][i]=TPR        
        non_nested_mcc_5_f[j][i] = matthews_corrcoef(y_true=y_sample,y_pred=y_pred_5_f)
        acc_random=(np.sum(cm_i[0])/np.sum(cm_i))**2+(np.sum(cm_i[1])/np.sum(cm_i))**2
        non_nested_kappa_5_f[j][i]=(non_nested_scores_5_f[j][i]-acc_random)/(1-acc_random)

        
        print(i,'5f \n',confusion_matrix(y_true=y_sample,y_pred=y_pred_5_f),clf.best_params_,non_nested_scores_5_f[j][i],'\n')
        
    
    non_nested_params_5_f.append(param_column)
    non_nested_cm_5_f.append(non_nested_cm_5_f_column)
    nested_cm.append(nested_cm_column) 
    
    
    # LOO parameter search and scoring
    if(N<3000):
    
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        
        start_time = time.time()
         
        clf_loo = GridSearchCV(estimator=svm, param_grid=param_grid, cv=loo, n_jobs=-1)
        clf_loo.fit(X_sample, y_sample)
        y_pred_loo=cross_val_predict(clf_loo.best_estimator_, X_sample, y_sample,cv=loo, n_jobs=-1)
    
        non_nested_scores_loo[j]=accuracy_score(y_sample,y_pred_loo)
        non_nested_times_loo[j]=time.time() - start_time
        non_nested_params_loo.append(clf_loo.best_params_)
        
        cm_i=confusion_matrix(y_true=y_sample,y_pred=y_pred_loo)
        non_nested_cm_loo.append(cm_i)
            
        TP = cm_i[1][1]
        FP = cm_i[0][1]
        FN = cm_i[1][0]
        TN = cm_i[0][0]
        TPR = TP/(TP+FN)
        
        non_nested_tpr_loo[j]=TPR
        non_nested_mcc_loo[j] = matthews_corrcoef(y_true=y_sample,y_pred=y_pred_loo)
        acc_random=(np.sum(cm_i[0])/np.sum(cm_i))**2+(np.sum(cm_i[1])/np.sum(cm_i))**2
        non_nested_kappa_loo[j]=(non_nested_scores_loo[j]-acc_random)/(1-acc_random)

        print(cm_i)
        print("LOO", non_nested_params_loo[j],non_nested_scores_loo[j],'\n')
        
    j += 1
    
  
        
##############################################################
#Saving results
##############################################################


np.savetxt("churn_nested_scores_3.txt", nested_scores, fmt="%s")
np.savetxt("churn_3f_non_nested_scores_3.txt", non_nested_scores, fmt="%s")
np.savetxt("churn_non_nested_scores_loo.txt", non_nested_scores_loo, fmt="%s")
np.savetxt("churn_5f_non_nested_scores_3.txt", non_nested_scores_5_f, fmt="%s")

np.savetxt("churn_nested_times_3.txt", nested_times, fmt="%s")
np.savetxt("churn_3f_non_nested_times_3.txt", non_nested_times, fmt="%s")
np.savetxt("churn_non_nested_times_loo.txt", non_nested_times_loo, fmt="%s")
np.savetxt("churn_5f_non_nested_times_3.txt", non_nested_times_5_f, fmt="%s")

np.savetxt("churn_nested_mcc_3.txt", nested_mcc, fmt="%s")
np.savetxt("churn_non_nested_mcc_loo.txt", non_nested_mcc_loo, fmt="%s")
np.savetxt("churn_5f_non_nested_mcc_3.txt", non_nested_mcc_5_f, fmt="%s")

np.savetxt("churn_nested_kappa_3.txt", nested_kappa, fmt="%s")
np.savetxt("churn_non_nested_kappa_loo.txt", non_nested_kappa_loo, fmt="%s")
np.savetxt("churn_5f_non_nested_kappa_3.txt", non_nested_kappa_5_f, fmt="%s")

np.savetxt("churn_nested_tpr_3.txt", nested_tpr, fmt="%s")
np.savetxt("churn_non_nested_tpr_loo.txt", non_nested_tpr_loo, fmt="%s")
np.savetxt("churn_5f_non_nested_tpr_3.txt", non_nested_tpr_5_f, fmt="%s")

np.savetxt("churn_non_nested_params_loo.txt", non_nested_params_loo, fmt="%s")
np.savetxt("churn_5f_non_nested_params_3.txt", non_nested_params_5_f, fmt="%s")


k=0
l=0
rownumber=np.arange(len(rows))
for k in rownumber:
    with open('churn_5f_non_nested_cm_3.txt', 'a') as outfile:
        outfile.write('# {0} rows\n'.format(rows[k]))
        for l in range(NUM_TRIALS):      
            np.savetxt(outfile, non_nested_cm_5_f[k][l], fmt='%s')
            outfile.write('\n')

k=0
l=0
rownumber=np.arange(len(rows)-2)
for k in rownumber:
    with open('churn_non_nested_cm_loo.txt', 'a') as outfile:
        outfile.write('# {0} rows\n'.format(rows[k]))    
        np.savetxt(outfile, non_nested_cm_loo[k], fmt='%s')
        outfile.write('\n')
            
k=0
l=0
rownumber=np.arange(len(rows))
for k in rownumber:
    with open('churn_nested_cm_3.txt', 'a') as outfile:
        outfile.write('# {0} rows\n'.format(rows[k]))
        for l in range(NUM_TRIALS):      
            np.savetxt(outfile, nested_cm[k][l], fmt='%s')
            outfile.write('\n')

    
##############################################################
#Plots
##############################################################

 
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)



# Plot Accuracy results
# Adjust font family and font size 
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
figure(num=None,figsize=(9, 6), dpi=160, facecolor='w', edgecolor='k')

accuracy = np.zeros((3,len(rows)))

accuracy[0]=non_nested_scores_loo
accuracy[0][6:]=np.nan
accuracy[1]=non_nested_scores_5_f.mean(axis=1)
accuracy[2]=nested_scores.mean(axis=1)

plt.plot(rows,accuracy[0], label="Flat LOOCV", marker='o')
plt.plot(rows,accuracy[1], label="5-Fold Flat CV", marker='o')
plt.plot(rows,accuracy[2], label="5x3 Nested CV", marker='o')
plt.xscale("log")
plt.xticks(rows, labels=rows)

plt.xlabel("Minták száma [db]")
plt.ylabel("Pontosság")
plt.legend()



# Plot runtimes
# Adjust font family and font size 
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
figure(num=None,figsize=(9, 6), dpi=160, facecolor='w', edgecolor='k')

runtime = np.zeros((3,len(rows)))

runtime[0]=non_nested_times_loo
runtime[0][6:]=np.nan
runtime[1]=non_nested_times_5_f.mean(axis=1)
runtime[2]=nested_times.mean(axis=1)

plt.axhline(60,color='lightgrey',linestyle='dashed')
plt.text(52,32,'1 perc',size='xx-large',color='grey')
plt.axhline(3600,color='lightgrey',linestyle='dashed')
plt.text(52,1800,'1 óra',size='xx-large',color='grey')
plt.axhline(86400,color='lightgrey',linestyle='dashed')
plt.text(52,48000,'1 nap',size='xx-large',color='grey')

plt.plot(rows,runtime[0], label="Flat LOOCV", marker='o')
plt.plot(rows,runtime[1], label="5-Fold Flat CV", marker='o')
plt.plot(rows,runtime[2], label="5x3 Nested CV", marker='o')
plt.xscale("log"), plt.yscale("log")
plt.xticks(rows, labels=rows)

plt.xlabel("Minták száma [db]")
plt.ylabel("Futási idő [s]")
plt.legend()



# Plot Kappa results
# Adjust font family and font size 
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
figure(num=None,figsize=(9, 6), dpi=160, facecolor='w', edgecolor='k')

kappa = np.zeros((3,len(rows)))

kappa[0]=non_nested_kappa_loo
kappa[0][6:]=np.nan
kappa[1]=non_nested_kappa_5_f.mean(axis=1)
kappa[2]=nested_kappa.mean(axis=1)

plt.plot(rows,kappa[0], label="Flat LOOCV", marker='o')
plt.plot(rows,kappa[1], label="5-Fold Flat CV", marker='o')
plt.plot(rows,kappa[2], label="5x3 Nested CV", marker='o')

plt.xscale("log")
plt.xticks(rows, labels=rows)

plt.xlabel("Minták száma [db]")
plt.ylabel("Kappa")
plt.legend()



# Plot MCC results
# Adjust font family and font size 
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
figure(num=None,figsize=(9, 6), dpi=160, facecolor='w', edgecolor='k')
axes=plt.gca()
#axes.set_ylim([0.10,0.71])

mcc = np.zeros((3,len(rows)))

mcc[0]=non_nested_mcc_loo
mcc[0][6:]=np.nan
mcc[1]=non_nested_mcc_5_f.mean(axis=1)
mcc[2]=nested_mcc.mean(axis=1)

plt.plot(rows,mcc[0], label="Flat LOOCV", marker='o')
plt.plot(rows,mcc[1], label="5-Fold Flat CV", marker='o')
plt.plot(rows,mcc[2], label="5x3 Nested CV", marker='o')

plt.xscale("log")
plt.xticks(rows, labels=rows)

plt.xlabel("Minták száma [db]")
plt.ylabel("Matthews koefficiens")
plt.legend(loc=4)



# Plot TPR results
# Adjust font family and font size 
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
figure(num=None,figsize=(9, 6), dpi=160, facecolor='w', edgecolor='k')
axes=plt.gca()
#axes.set_ylim([0.10,0.71])

tpr = np.zeros((3,len(rows)))

tpr[0]=non_nested_tpr_loo
tpr[0][6:]=np.nan
tpr[1]=non_nested_tpr_5_f.mean(axis=1)
tpr[2]=nested_tpr.mean(axis=1)

plt.plot(rows,tpr[0], label="Flat LOOCV", marker='o')
plt.plot(rows,tpr[1], label="5-Fold Flat CV", marker='o')
plt.plot(rows,tpr[2], label="5x3 Nested CV", marker='o')

plt.xscale("log")
plt.xticks(rows, labels=rows)

plt.xlabel("Minták száma [db]")
plt.ylabel("True Positive Rate")
plt.legend(loc=4)



##############################################################
#Reading data back, if necessary
##############################################################
            
"""
non_nested_times_loo=np.append(np.loadtxt("churn_non_nested_times_loo.txt"),[np.nan,np.nan])
nested_times=np.loadtxt("churn_nested_times_3.txt")
non_nested_times_5_f=np.loadtxt("churn_5f_non_nested_times_3.txt")

non_nested_scores_loo=np.append(np.loadtxt("churn_non_nested_scores_loo.txt"),[np.nan,np.nan])
nested_scores=np.loadtxt("churn_nested_scores_3.txt")
non_nested_scores_5_f=np.loadtxt("churn_5f_non_nested_scores_3.txt")

non_nested_cm_loo=np.loadtxt("churn_non_nested_cm_loo.txt")
non_nested_cm_5_f=np.loadtxt("churn_5f_non_nested_cm_3.txt")
nested_cm=np.loadtxt("churn_nested_cm_3.txt")

non_nested_kappa_loo=np.loadtxt("churn_non_nested_kappa_loo.txt")
non_nested_kappa_5_f=np.loadtxt("churn_5f_non_nested_kappa_3.txt")
nested_kappa=np.loadtxt("churn_nested_kappa_3.txt")

non_nested_mcc_loo=np.loadtxt("churn_non_nested_mcc_loo.txt")
non_nested_mcc_5_f=np.loadtxt("churn_5f_non_nested_mcc_3.txt")
nested_mcc=np.loadtxt("churn_nested_mcc_3.txt")

non_nested_tpr_loo=np.loadtxt("churn_non_nested_tpr_loo.txt")
non_nested_tpr_5_f=np.loadtxt("churn_5f_non_nested_tpr_3.txt")
nested_tpr=np.loadtxt("churn_nested_tpr_3.txt")


nested_cm=[]
testsite_array = np.loadtxt('churn_nested_cm_3.txt')
testsite_array=testsite_array.astype(int)

z=0
for j in range(len(rows)):
    cm_list_perm=[]
    for i in range(NUM_TRIALS):
        cm_list_perm.append(testsite_array[z:z+2,:2])
        z+=2
    nested_cm.append(cm_list_perm)
    
    
non_nested_cm_5_f=[]
testsite_array = np.loadtxt('churn_5f_non_nested_cm_3.txt')
testsite_array=testsite_array.astype(int)

z=0
for j in range(len(rows)):
    cm_list_perm=[]
    for i in range(NUM_TRIALS):
        cm_list_perm.append(testsite_array[z:z+2,:2])
        z+=2
    non_nested_cm_5_f.append(cm_list_perm)
    
    
non_nested_cm_loo=[]
testsite_array = np.loadtxt('churn_non_nested_cm_loo.txt')
testsite_array=testsite_array.astype(int)

z=0
for j in range(len(rows)):
    non_nested_cm_loo.append(testsite_array[z:z+2,:2])
    z+=2
"""