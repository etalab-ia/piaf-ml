import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./results/analysis_results.csv')

df = df[df.level == 'theme']

to_work = ['fiche_ok', 'score_dense_no_lemma', 'score_sparse_no_lemma']
#       'position_dense_lemma', 'position_sparse_lemma']

X = df[to_work]
X.dropna(inplace=True)

y = X.fiche_ok
X.drop(['fiche_ok'], axis=1, inplace=True)

x = scaler.fit_transform(X)
X =pd.DataFrame(x, columns=X.columns)

"""
X['prod'] = X['score_dense_no_lemma'] * X['score_sparse_no_lemma']
X['square'] = X['prod'] * X['prod']
X = X[['prod','square']]"""

class FakeSampler(BaseSampler):

    _sampling_type = 'bypass'

    def _fit_resample(self, X, y):
        return X, y

def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res.iloc[:, 0], X_res.iloc[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
sampler = FakeSampler()
clf = make_pipeline(sampler, LinearSVC())
plot_resampling(X, y, sampler, ax1)
ax1.set_title('Original data')

ax_arr = (ax2, ax3, ax4)
for ax, sampler in zip(ax_arr, (RandomOverSampler(random_state=0),
                                SMOTE(random_state=0),
                                ADASYN(random_state=0))):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_resampling(X, y, sampler, ax)
    ax.set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()
plt.show()


print('hello')