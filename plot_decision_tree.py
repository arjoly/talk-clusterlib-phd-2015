import os
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn.datasets import make_circles
from sklearn.cross_validation import train_test_split

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def plot_boundary(fname, X, y, fitted_estimator=None, mesh_step_size=0.2):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.figure()

    if fitted_estimator is not None:
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(fitted_estimator, "decision_function"):
            Z = fitted_estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = fitted_estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        # plt.title("score = %s" % fitted_estimator.score(X, y))

    # Plot testing point
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.savefig("%s.pdf" % fname)
    plt.close()


def plot_error_curve(X, y, estimator, param_name, param_values, n_repetition=5):

    scores = np.empty((len(param_values), n_repetition), dtype=float)

    for iter_ in range(n_repetition):
        X, y = make_circles(n_samples=500, noise=0.1, random_state=iter_)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.5,
                                                            random_state=0)

        for i, value in enumerate(param_values):
            est = clone(estimator)

            est.set_params(**{param_name: value})

            try:
                est.set_params(random_state=iter_)
            except ValueError:
                pass

            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)
            scores[i, iter_] = accuracy_score(y_test, y_pred)

    scores_mean = scores.mean(axis=1)
    scores_std = scores.std(axis=1)

    plt.figure()
    plt.plot(param_values, scores_mean, "o-", color="g")
    plt.fill_between(param_values, scores_mean - scores_std,
                     scores_mean + scores_std, alpha=0.1, color="g")
    plt.xlabel("Random forest size")
    plt.ylabel('Accuracy')
    # best = np.argmax(scores_mean)
    # plt.title("Best %s = %s with score = %s"
    #           % (param_name, param_values[best], scores_mean[best]))
    plt.savefig("images/%s_accuracy_%s.pdf" %
                (estimator.__class__.__name__, param_name))
    plt.close()


if __name__ == "__main__":
    X, y = make_circles(n_samples=500, noise=0.1, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=0)

    est = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

    with open("images/decision_tree.dot", 'w') as fhandle:
        export_graphviz(est, out_file=fhandle)
    os.system("dot -Tpdf images/decision_tree.dot "
              "-o images/decision_tree_structure.pdf")

    # plot_boundary("images/decision_tree_boundary", X_test, y_test,
    #               est)
    #
    # plot_boundary("images/forest_pruning_nolearn", X_test, y_test)
    # for n_estimators in [1, 100, 1000]:
    #     est = RandomForestClassifier(random_state=0, n_estimators=n_estimators)
    #     est.fit(X_train, y_train)
    #     plot_boundary("images/forest_pruning_n_estimators_%s" % n_estimators,
    #                   X_test, y_test, est)
    #
    # plot_error_curve(X, y, RandomForestClassifier(), "n_estimators",
    #                  [2 ** i for i in range(12)], n_repetition=5)
