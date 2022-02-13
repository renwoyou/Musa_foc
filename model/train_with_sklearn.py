from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import os, pandas, hppi
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, recall_score, log_loss

classifiers = {
  'svm' : svm.SVC(kernel='linear', C=0.01),
  'random_forest' : RandomForestClassifier(n_estimators=15,),
  'ada_boost' : AdaBoostClassifier(learning_rate=0.1),
  'decision_tree' : DecisionTreeClassifier(random_state=0),
  'kneighbors' : KNeighborsClassifier(n_neighbors=2),
}

def train_and_test(data_sets_dir, classifier):
  # Load datasets.
  hppids = hppi.read_data_sets(data_sets_dir, one_hot=False)
  train_datas, train_labels, test_datas, test_labels = hppids.shuffle().split()

  # train_datas  = train_datas [:100]
  # train_labels = train_labels[:100]
  # test_datas   = test_datas  [:100]
  # test_labels  = test_labels [:100]

  # train
  begin_time = datetime.now()
  classifier.fit(train_datas, train_labels)
  end_time = datetime.now()
  train_time = (end_time-begin_time).total_seconds()

  # test
  begin_time = datetime.now()
  mean_accuracy = classifier.score(test_datas, test_labels)
  end_time = datetime.now()
  test_time = (end_time-begin_time).total_seconds()

  # predict
  begin_time = datetime.now()
  prediction = classifier.predict(test_datas)
  # confusion_matrix(test_labels, prediction)
  end_time = datetime.now()
  predict_time = (end_time-begin_time).total_seconds()

  fpr, tpr, thresholds = roc_curve(test_labels, prediction)

  return (mean_accuracy,
          auc(fpr, tpr),
          average_precision_score(test_labels, prediction),
          recall_score(test_labels, prediction),
          log_loss(test_labels, prediction),
          train_time,
          test_time,
          predict_time,
          )

def do_with(sub_dir, flag):
  cwd = os.getcwd()
  df = pandas.DataFrame(columns=('accuracy', 'auc', 'average_precision', 'recall', 'log_loss', 'train_time', 'test_time', 'predict_time', ))
  df.loc[len(df)] = train_and_test(cwd + "/data/" + sub_dir, classifiers[flag])
  df.to_csv(cwd +"/data/"+ '/results/' + flag + '-' + sub_dir + '.csv')

def main():

  from sys import argv
  _, flag1,flag2,flag3,flag4 = argv
  # flag='svm'

  do_with("musa10", flag1)
  do_with("musa10", flag2)
  do_with("musa10", flag3)
  do_with("musa10", flag4)
  #
  # do_with("human-ac-bin", flag1)#human  ac
  # do_with("human-ac-bin", flag2)
  # do_with("human-ac-bin", flag3)
  # do_with("human-ac-bin", flag4)
  #
  # do_with("human-ct-bin", flag1)  # human  ct
  # do_with("human-ct-bin", flag2)
  # do_with("human-ct-bin", flag3)
  # do_with("human-ct-bin", flag4)

  # do_with("02-ct-bin", flag)
  # do_with("03-ac-bin", flag)
  # do_with("04-ld-bin", flag)
  # do_with("05-mos-bin", flag)
  # do_with("09-hppids", flag)

if __name__ == "__main__":
    main()
