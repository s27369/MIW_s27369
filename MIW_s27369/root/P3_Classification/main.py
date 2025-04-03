from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from functions import *

noise=0.4
n_samples = 10_000

# if __name__=="__main__":
#     print(type(range(1, 5)))
if __name__=="__main__":
    #1
    data = datasets.make_moons(n_samples, noise=noise)
    X, y = data
    print(len(X), len(y))
    plot_data("make_moons", X, y, "data.png")
    #2
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #3
    depth_range = range(1, 11)
    dtc_values_gini = evaluate_tree_depths(DecisionTreeClassifier, "gini", depth_range, X_train, y_train, X_test, y_test)
    dtc_values_entropy = evaluate_tree_depths(DecisionTreeClassifier, "entropy", depth_range, X_train, y_train, X_test, y_test)

    # plot_metric("DecisionTreeClassifier", "gini", dtc_gini_values, depth_range)
    # plot_metric("DecisionTreeClassifier", "entropy", dtc_entropy_values, depth_range)
    plot_metric_comparison("DecisionTreeClassifier", "gini", "entropy", dtc_values_gini, dtc_values_entropy, depth_range, "DTC_plot_comparison.png")

    #4
    rfc_values_gini = evaluate_tree_depths(RandomForestClassifier, "gini", depth_range, X_train, y_train, X_test, y_test)
    rfc_values_entropy = evaluate_tree_depths(RandomForestClassifier, "gini", depth_range, X_train, y_train, X_test, y_test)
    plot_metric("RandomForestClassifier", "gini", rfc_values_gini, depth_range, "RFC_plot_entropy.png")
    plot_metric("RandomForestClassifier", "gini", rfc_values_entropy, depth_range, "RFC_plot_gini.png")