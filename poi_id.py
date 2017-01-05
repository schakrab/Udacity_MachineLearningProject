
import sys
import pickle
sys.path.append("../tools/")
import sklearn
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import grid_search
from sklearn.cross_validation import StratifiedShuffleSplit
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features= ['salary', 'deferral_payments', 'total_payments']
email_features= ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
new_features=['fraction_from_poi', 'fraction_to_poi', 'shared_receipt_with_poi']
features_list = ['poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### Task 3: Create new feature(s)

fraction_from_poi=[]
fraction_to_poi=[]
for i in data_dict:
        if data_dict[i]['from_this_person_to_poi']=="NaN" or data_dict[i]['to_messages']=="NaN":
            fraction_to_poi.append(0.0)
        elif data_dict[i]['from_this_person_to_poi']>=0:
            fraction_to_poi.append(float(data_dict[i]['from_this_person_to_poi'])/float(data_dict[i]['to_messages'])) 
for s in data_dict:
        if data_dict[s]['from_poi_to_this_person']=="NaN" or data_dict[s]['from_messages']=="NaN":
            fraction_from_poi.append(0.0)
        elif data_dict[s]['from_poi_to_this_person']>=0:
            fraction_from_poi.append(float(data_dict[s]['from_poi_to_this_person'])/float(data_dict[s]['from_messages']))

                                                                                                     
index=0
for x in data_dict:
    data_dict[x]['fraction_from_poi']=fraction_from_poi[index]
    data_dict[x]['fraction_to_poi']=fraction_to_poi[index]
    index=index+1
                                                                                                  
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#features_list+=email_features
#features_list+=financial_features
features_list+=new_features

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

                                                                                                     
params={"n_estimators":[2, 3, 5], "criterion":('gini', 'entropy')}
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf_nb = GaussianNB()
                                     

clf_dt=DecisionTreeClassifier()



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
                                     
clf_nb.fit(features_train, labels_train)
pred_nb=clf_nb.predict(features_test)
acc_nb=accuracy_score(pred_nb, labels_test)  #.77777


clf_dt.fit(features_train, labels_train)
pred_dt=clf_dt.predict(features_test)
acc_dt=accuracy_score(pred_dt, labels_test) #.888888




#tune
dt_param={ "min_samples_split":list(range(1, 30)),  "criterion": ('gini', 'entropy')}
cv=StratifiedShuffleSplit(labels, 1000, random_state=42)
dt_tune = grid_search.GridSearchCV(clf_dt, dt_param, scoring="f1", cv=cv)
dt_tune.fit(features, labels)
dt_tune_pred=dt_tune.predict(features_test)
acc_dt_tune=accuracy_score(dt_tune_pred, labels_test)  #.8888
prec_dt_tune=precision_score(dt_tune_pred, labels_test) #.5
recall_dt_tune=recall_score(dt_tune_pred, labels_test) #1




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(dt_tune.best_estimator_, my_dataset, features_list)
