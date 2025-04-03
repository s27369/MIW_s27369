from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

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
    # dtc_values_gini = evaluate_tree_depths(DecisionTreeClassifier, "gini", depth_range, X_train, y_train, X_test, y_test)
    # dtc_values_entropy = evaluate_tree_depths(DecisionTreeClassifier, "entropy", depth_range, X_train, y_train, X_test, y_test)
    #
    # plot_metric("DecisionTreeClassifier", "gini", dtc_values_gini, depth_range, "DTC_plot_gini.png")
    # plot_metric("DecisionTreeClassifier", "entropy", dtc_values_entropy, depth_range, "DTC_plot_entropy.png")
    # plot_metric_comparison("DecisionTreeClassifier", "gini", "entropy", dtc_values_gini, dtc_values_entropy, depth_range, "DTC_plot_comparison.png")

    #4
    rfc_values_gini = evaluate_tree_depths(RandomForestClassifier, "gini", depth_range, X_train, y_train, X_test, y_test)
    rfc_values_entropy = evaluate_tree_depths(RandomForestClassifier, "gini", depth_range, X_train, y_train, X_test, y_test)
    plot_metric("RandomForestClassifier", "gini", rfc_values_gini, depth_range, "RFC_plot_gini.png")
    plot_metric("RandomForestClassifier", "entropy", rfc_values_entropy, depth_range, "RFC_plot_entropy.png")
    plot_metric_comparison("RandomForestClassifier", "gini", "entropy", rfc_values_gini, rfc_values_entropy, depth_range, "RFC_plot_comparison.png")
    #
    # #5
    # model_logreg = LogisticRegression()
    # acc_logreg = fit_and_eval(model_logreg, X_train, y_train, X_test, y_test)
    # print_model_acc("LogisticRegression", acc_logreg)
    #
    # model_svc_soft = svm.SVC(probability=True) #for soft voting
    # acc_svc_soft = fit_and_eval(model_svc_soft, X_train, y_train, X_test, y_test)
    # print_model_acc("SVC_soft", acc_svc_soft)
    #
    # model_svc_hard = svm.SVC()
    # acc_svc_hard = fit_and_eval(model_svc_hard, X_train, y_train, X_test, y_test)
    # print_model_acc("SVC_hard", acc_svc_hard)
    #
    # best_depth_entropy = max(rfc_values_entropy)
    # best_depth_gini = max(rfc_values_gini)
    #
    # if best_depth_entropy>best_depth_gini:
    #     best_depth = rfc_values_entropy.index(best_depth_entropy)
    #     best_criterion = "entropy"
    # else:
    #     best_depth = rfc_values_gini.index(best_depth_gini)
    #     best_criterion = "gini"
    #
    # model_rfc = RandomForestClassifier(criterion=best_criterion, max_depth=best_depth)
    # acc_rfc = fit_and_eval(model_rfc, X_train, y_train, X_test, y_test)
    # print_model_acc("RandomForestClassifier", acc_rfc)
    #
    # #6
    # voting_clf_soft = VotingClassifier(
    #     estimators=[
    #         ("logreg", model_logreg),
    #         ("svc", model_svc_soft),
    #         ("rfc", model_rfc),
    #     ],
    #     voting="soft"
    # )
    # voting_clf_hard = VotingClassifier(
    #     estimators=[
    #         ("logreg", model_logreg),
    #         ("svc", model_svc_hard),
    #         ("rfc", model_rfc),
    #     ],
    #     voting="hard"
    # )
    #
    # acc_voting_soft = fit_and_eval(voting_clf_soft, X_train, y_train, X_test, y_test)
    # print_model_acc("VotingClassifier (soft)", acc_voting_soft)
    # acc_voting_hard = fit_and_eval(voting_clf_hard, X_train, y_train, X_test, y_test)
    # print_model_acc("VotingClassifier (hard)", acc_voting_hard)
    #
    # # LogisticRegression accuracy: 0.8336
    # # SVC_soft accuracy: 0.8672
    # # SVC_hard accuracy: 0.8672
    # # RandomForestClassifier accuracy: 0.8528
    # # VotingClassifier(soft) accuracy: 0.8628
    # # VotingClassifier(hard) accuracy: 0.8624
    #
    # #jak pokazują wyniki, ze składowych VotingClassifier najgorszy jest LogisticRegression.
    # #najlepszą skłądową jest SVC
    # #VotingClassifier dla tych danych lepiej radził sobie odrobinę stosując soft voting
    # #VotingClassifier pozwolił na niewielką poprawę wyników względem średniej wyników 3 modeli składowych
    # #"co 3 głowy to nie jedna"