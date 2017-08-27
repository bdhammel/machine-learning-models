import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

plt.close('all')

try:
    train
except:
    train = pd.read_csv("data/train.csv")
    # save the labels to a Pandas series target
    targets = train['label']
    # Drop the label feature
    data = train.drop("label",axis=1)

    data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.33, random_state=42)


clf = GaussianNB()
clf.fit(data_train, targets_train)
print(clf.score(data_test, targets_test))

"""
try:
    averaged_df
except:
    averaged_df = pd.DataFrame()
    for label in sorted(targets_train.unique()):
        averaged_df.loc[:, label] = data_train[targets_train == label].mean()

axes = averaged_df.plot(
        y=averaged_df.columns, kind="line", sharex=True, sharey=True, 
        subplots=True, use_index=True, yticks=[], legend=False)


for i, axis in enumerate(axes):
    axis.set_ylabel(averaged_df.columns[i])

plt.draw()

plt.figure()
plt.imshow(averaged_df[0].values.reshape(28,28))

X = train.values
X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

eig_vecs = [vec[1] for vec in eig_pairs]

"""
# Invoke SKlearn's PCA method
n_components = 90
pca = PCA(n_components=n_components).fit(data_train.values)

evecs = pca.components_.reshape(n_components, 28, 28)
data_train = pca.transform(data_train)

# Extracting the PCA components ( eignevalues )
#eigenvalues = pca.components_.reshape(n_components, 28, 28)
evacs = pca.components_

n_row = 5
n_col = 6

# Plot the first 8 eignenvalues
plt.figure(figsize=(8,7))

for i in range(31):
    #     for offset in [10, 30,0]:
    #     plt.subplot(n_row, n_col, i + 1)
    offset = 0
    plt.subplot(n_row, n_col, i)
    plt.imshow(evecs[i].reshape(28,28), cmap='jet')

    title_text = 'Eigenvalue ' + str(i + 1)
    plt.title(title_text, size=6.5)
    plt.xticks(())
    plt.yticks(())

plt.show()

data_test = pca.transform(data_test)

clf = GaussianNB()

clf.fit(data_train, targets_train)
print(clf.score(data_test, targets_test))


eigenvalues = pca.explained_variance_
eigenvalues = np.array(sorted(eigenvalues, reverse=True))
plt.plot(eigenvalues.cumsum())


