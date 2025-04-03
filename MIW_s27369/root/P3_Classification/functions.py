from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def fit_and_eval(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

def evaluate_tree_depths(model_class, criterion:str, depth:range, X_train, y_train, X_test, y_test):
    metric_values = []
    depths = []
    for i in depth:
        model_kwargs = {"criterion": criterion} if "criterion" in model_class().get_params() else {}
        if "max_depth" in model_class().get_params():
            model_kwargs["max_depth"] = i
        model = model_class(**model_kwargs)
        depths.append(i)
        metric_values.append(fit_and_eval(model, X_train, y_train, X_test, y_test))
    return metric_values

def plot_metric(model_name:str, metric_name:str, metric_values:list, depths:list, output_name=None) ->None:
    plt.plot(depths, metric_values)
    plt.title(f"{metric_name} vs depth: {model_name}")
    plt.xlabel("Depth")
    plt.ylabel(metric_name)
    if output_name:
        plt.savefig(output_name)
    plt.show()

def plot_data(dataset_name, X:list, y:list, output_name=None) ->None:
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", alpha=0.3)
    plt.title(dataset_name)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    if output_name:
        plt.savefig(output_name)
    plt.show()


def plot_metric_comparison(model_name:str, metric_name_1:str, metric_name_2:str, metric_values_1:list, metric_values_2:list, depths:list, output_name=None):
    plt.plot(depths, metric_values_1, label=metric_name_1, marker="o")
    plt.plot(depths, metric_values_2, label=metric_name_2, marker="x")
    plt.title(f"{metric_name_1} & {metric_name_2} vs depth: {model_name}")
    plt.xlabel("Depth")
    plt.ylabel("Metrics")
    plt.legend()
    if output_name:
        plt.savefig(output_name)
    plt.show()
