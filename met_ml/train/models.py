import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

class xgb_model():
    def __init__(self):
        self.params = {'objective': 'reg:squarederror'}
        self.eval_func = {'mse': mean_squared_error, 'r2': r2_score}
        
    def fit(self, X_train, y_train, X_test, y_test):
        # training data 
        dtrain = xgb.DMatrix(
            data=X_train.values,
            label=y_train.values
        )
        # val data 
        dtest = xgb.DMatrix(
            data=X_test.values,
            label=y_test.values
        )

        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        self.model = xgb.train(params=self.params, 
                               dtrain=dtrain, 
                               num_boost_round=200, 
                               evals=watchlist, 
                               early_stopping_rounds=10)

    def predict(X):
        dtest = xgb.DMatrix(data=X.values)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)
        
    def evaluate(X, y):
        y_pred = self.predict(X)
        scores = {}
        for k, func in self.eval_func.items():
            scores[k] = func(y, y_pred)
            
        return scores
