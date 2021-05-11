from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def createScatterPlot(list_of_samples):
    sample_class1 = []
    sample_class2 = []
    for i, example in enumerate(list_of_samples["target"]):
        if example == 1:
            sample_class1.append(list_of_samples.values[i])
        else:
            sample_class2.append(list_of_samples.values[i])
    s1 = np.array(sample_class1)
    s2 = np.array(sample_class2)
    plt.scatter(s1[:, 0], s1[:, 1], c='grey', label='First Wine')
    plt.scatter(s2[:, 0], s2[:, 1], c='green', label='Second Wine')
    plt.xlabel("alcohol")
    plt.ylabel("magnesium")
    plt.legend()
    plt.show()


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=50, linewidth=1, facecolors='none', edgecolor='black');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def main():
    # Read the wine dataset
    dataset = load_wine()
    df = pd.DataFrame(data=dataset['data'], columns=dataset['feature_names'])
    df = df.assign(target=pd.Series(dataset['target']).values)
    # Filter the irrelevant columns
    df = df[['alcohol', 'magnesium', 'target']]
    # Filter the irrelevant label
    df = df[df.target != 2]
    train_df = df.drop(df.index[45:75])
    val_df = df[45:75]
    # *****Q1.1
    # createScatterPlot(train_df)
    # createScatterPlot(val_df)
    # ******** Q1.2
    X = train_df.iloc[:, :-1].values
    Y = train_df["target"]
    X_val = val_df.iloc[:, :-1].values
    Y_val = val_df["target"]
    set1 = [X, Y]
    set2 = [X_val, Y_val]
    margins=[]
    for c in [0.01, 0.05, 0.1]:
        for set in [set1, set2]:
            model = SVC(kernel='linear', C=c)
            model.fit(set[0], set[1])
            plt.scatter(set[0][:, 0], set[0][:, 1], c=set[1], s=50, cmap='autumn')
            plot_svc_decision_function(model)
            plt.title('scatter with C=' + str(c))
            plt.show()
            margins.append(1 / (np.sqrt(np.sum(model.coef_ ** 2))))
            accuracy_score()


    # *********

if __name__ == "__main__":
    main()
