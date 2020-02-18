from sklearn.svm import SVC

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
X, y = mnist["data"], mnist["target"]
some_digit  = X[0]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])
svm_clf.predict([some_digit])
