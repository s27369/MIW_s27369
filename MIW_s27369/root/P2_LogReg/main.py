from MyLogRegSoftmax import *
from plotka import *
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    model = MyLogRegSoftmax(eta=0.05, n_iter=1000, random_state=1)
    model.fit(X_train, y_train)

    plot_decision_regions(X_train, y_train, classifier=model)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()

