import numpy as np
from os import path
from neurometrics.matlabcommand import matlab_command
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sknn.mlp import Classifier, Layer
from sklearn.grid_search import GridSearchCV, BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.scorer import check_scoring
from sklearn.cross_validation import check_cv, train_test_split
from sklearn.base import clone

def count_to_targets(count, classes=None):
    if classes is None:
        classes = np.unique(count)
    targets = np.zeros((len(count),len(classes)))
    for i,c in enumerate(classes):
        targets[count == c,i] = 1
    return targets

class RetrainCV(BaseSearchCV):
    def __init__(self, estimator, n, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        super(RetrainCV, self).__init__(
            estimator, scoring, fit_params, n_jobs, iid,
            refit, cv, verbose, pre_dispatch, error_score)
        self.n = n
                     
    def fit(self, X, y=None):
        return self._fit(X, y, [None]*self.n)


def retrain_estimator(estimator, X, y, n,
                      scoring=None,
                      fit_params=None,
                      split_params=None):
    scorer = check_scoring(estimator, scoring=scoring)
    split_params = split_params if split_params is not None else {}
    X_train, X_test, y_train, y_test = train_test_split(X,y,**split_params) #FIXME: shuffles
    fit_params = fit_params if fit_params is not None else {}
    estimators = [clone(estimator).set_params(nn__random_state=i).fit(X_train,y_train,**fit_params) for i in range(n)]
    scores = [scorer(e,X_test,y_test) for e in estimators]
    print(scores) #TODO: use logger
    return estimators[np.argmax(scores)]

class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        param_grid = [{'nn__hidden0__units': range(15,35,5)}]

        pipeline = Pipeline([
            ('scale', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('nn', Classifier(layers=[Layer("Sigmoid", units=15),
                                      Layer("Softmax")],
                              learning_rate=0.001,
                              n_iter=25))
            ])

        gs = GridSearchCV(
            pipeline,
            param_grid,
            scoring = None,
            cv = 5)        

        gs.fit(X,y)

        #self.estimator = retrain_estimator(gs.best_estimator_, X, y, n=10)
        self.estimator = gs.best_estimator_
        
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

class FeedForwardNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.tmp_dir = mkdtemp()

    def fit(self, X, y):
        targets = count_to_targets(y)
        tmp_file = path.join(self.tmp_dir,'net.mat')
        results = matlab_command(("net = fitNN(X,y);"
                                  "save(tmp_file,'net');"),
                                 X = X.T,
                                 y = targets.T,
                                 tmp_file = tmp_file)
        self.coef_ = self.sensitivity(X)
        return self

    def predict(self, X):
        tmp_file = path.join(self.tmp_dir,'net.mat')
        results = matlab_command(("load(tmp_file);"
                                  "y_pred = vec2ind(net(X));"),
                                 X = X.T,
                                 tmp_file = tmp_file)
        return (results['y_pred'].ravel() - 1) #dammit matlab...

    def predict_proba(self, X):
        tmp_file = path.join(self.tmp_dir,'net.mat')
        results = matlab_command(("load(tmp_file);"
                                  "y = net(X);"),
                                 X = X.T,
                                 tmp_file = tmp_file)
        return results['y'].T

    def confidence(self, X):
        tmp_file = path.join(self.tmp_dir,'net.mat')
        results = matlab_command(("load(tmp_file);"
                                  "output = net(X);"),
                                 X = X.T,
                                 tmp_file = tmp_file)
        return results['output']

    def sensitivity(self, X):
        tmp_file = path.join(self.tmp_dir,'net.mat')
        results = matlab_command(("load(tmp_file);"
                                  "Savg = ffnnSensitivityAnalysis(net,inputs);"),
                                 inputs = X.T,
                                 tmp_file = tmp_file)
        return results['Savg']

    def __del__(self):
        rmtree(self.tmp_dir)
