from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC


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
    model = SVC(kernel='linear', C=1.0)
    model.fit(train_df.iloc[:,:-1].values,train_df["target"])


if __name__ == "__main__":
    main()
