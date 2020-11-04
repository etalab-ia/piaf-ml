import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./results/analysis_results.csv')

df = df[df.level == 'theme']

to_work = ['fiche_ok', 'consistency_dense_no_lemma', 'consistency_sparse_no_lemma']
#       'position_dense_lemma', 'position_sparse_lemma']

X = df[to_work]

X.groupby(['fiche_ok'])[['consistency_dense_no_lemma', 'consistency_sparse_no_lemma']].mean()
X.groupby(['fiche_ok'])[['consistency_dense_no_lemma', 'consistency_sparse_no_lemma']].std()

X_pass = X[X.fiche_ok == True]

grouped = X_pass.groupby(['consistency_dense_no_lemma', 'consistency_sparse_no_lemma']).count()
total = grouped.sum().values[0]
unstack = (grouped.unstack('consistency_dense_no_lemma') * 100)/ total
plt.subplot(121)
plt.imshow(unstack)

X_pass = X[X.fiche_ok == False]
grouped = (X_pass.groupby(['consistency_dense_no_lemma', 'consistency_sparse_no_lemma']).count()*100) /total
total = grouped.sum().values[0]
unstack = (grouped.unstack('consistency_dense_no_lemma') * 100)/ total
plt.subplot(122)
plt.imshow(unstack)

plt.show()

grouped = (X.groupby(['fiche_ok', 'consistency_sparse_no_lemma']).count())
unstack = (grouped.unstack('fiche_ok'))
plt.imshow(unstack)
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
to_work = ['fiche_ok', 'consistency_sparse_no_lemma']
X = df[to_work]
X = X.dropna()
y_true = X['fiche_ok']
y_pred = X['consistency_sparse_no_lemma'] > 0.7
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

X_pass = X[X.fiche_ok == True]

grouped = X_pass.groupby(['consistency_sparse_no_lemma'])['consistency_sparse_no_lemma'].count()
total = grouped.sum()
unstack = (grouped * 100)/total
plt.plot(unstack, label = ('fiche_ok'))

X_pass = X[X.fiche_ok == False]

grouped = X_pass.groupby(['consistency_sparse_no_lemma'])['consistency_sparse_no_lemma'].count()
total = grouped.sum()
unstack = (grouped * 100)/total
plt.plot(unstack, label = ('fiche_nok'))
plt.legend()
plt.show()


X_pass = X[X.fiche_ok == True]

grouped = X_pass.groupby(['consistency_dense_no_lemma'])['consistency_dense_no_lemma'].count()
total = grouped.sum()
unstack = (grouped * 100)/total
plt.plot(unstack, label = ('fiche_ok'))

X_pass = X[X.fiche_ok == False]

grouped = X_pass.groupby(['consistency_dense_no_lemma'])['consistency_dense_no_lemma'].count()
total = grouped.sum()
unstack = (grouped * 100)/total
plt.plot(unstack, label = ('fiche_nok'))

plt.show()

X=X[X.consistency_sparse_no_lemma<=0.4]
X_pass = X[X.fiche_ok == True]

grouped = X_pass.groupby(['consistency_dense_no_lemma'])['consistency_dense_no_lemma'].count()
total = grouped.sum()
unstack = (grouped * 100)/total
plt.plot(unstack, label = ('fiche_ok'))

X_pass = X[X.fiche_ok == False]

grouped = X_pass.groupby(['consistency_dense_no_lemma'])['consistency_dense_no_lemma'].count()
total = grouped.sum()
unstack = (grouped * 100)/total
plt.plot(unstack, label = ('fiche_nok'))

plt.show()