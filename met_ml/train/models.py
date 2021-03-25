import fsspec
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from met_ml.data import cat as met_ml_cat
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Attention, MultiHeadAttention, GRU, GlobalAveragePooling1D


class base_model():
    def __init__(self, variable, eval_func):
        self.label_name = variable
        self.name = f'{variable}_base_model'
        self.eval_func = eval_func
        print(f'Building {self.name} model')
        
    def fit(self, X_train, y_train, X_test, y_test):
        raise NotImplementedError('    fit function not implemented yet')

    def _predict(self, X):
        raise NotImplementedError('    predict function not implemented yet')
            
    def evaluate(self, X, y, labels, scalers):
        y_pred = self._predict(X, labels, scalers)
        out = pd.DataFrame()
        out['y_true'] = y
        out['y_pred'] = y_pred
        n = len(out)
        out = out.dropna(how='any')
        if len(out) < n:
            print(f'    dropping {n-len(out)} samples due to nans')
        scores = {}
        for k, func in self.eval_func.items():
            scores[k] = func(out.y_true.values, out.y_pred.values)
            
        return scores


class mtclim(base_model):
    def __init__(self, variable, eval_func):
        self.label_name = variable
        self.name = f'{variable}_mtclim'
        self.eval_func = eval_func
        print(f'Building {self.name} model')
        
    def fit(self, X_train, y_train, X_test, y_test, folder='./', overwrite=False):
        pass

    def _predict(self, X, labels, scalers):
        all_sites = np.unique(labels.sel(features='Site').values)
        metsim = []
        for site in all_sites:
            try:
                site_metsim = pd.read_csv(f'/srv/shared/data-jhamman/metsim/metsim_{site}_HH.csv', parse_dates=True, index_col=0)
                site_metsim.rename(columns={'shortwave': 'SW_IN', 
                                            'longwave': 'LW_IN', 
                                            'air_pressure': 'PA',
                                            'rel_humid': 'RH'
                                           }, inplace=True)
                out = site_metsim[[self.label_name]].resample("1D").mean()
                out['Site'] = site
                out.index.name = 'TIMESTAMP_START'
                metsim.append(out.reset_index())
            except FileNotFoundError:
                print(f'    MTCLIM data for site {site} not available, skipping')
        metsim = pd.concat(metsim)
        metsim['TIMESTAMP_START'] = metsim.TIMESTAMP_START.dt.strftime('%Y-%m-%d')
        labels_df = pd.DataFrame(labels.values, columns=['Site', 'TIMESTAMP_START'])
        y_pred = labels_df.merge(metsim, on=['Site', 'TIMESTAMP_START'], how='left')
        return y_pred.set_index(['Site', 'TIMESTAMP_START']).reindex(labels_df[['Site', 'TIMESTAMP_START']]).values

    
class era(base_model):
    def __init__(self, variable, eval_func):
        self.label_name = variable
        self.name = f'{variable}_era'
        self.eval_func = eval_func
        print(f'Building {self.name} model')
        
    def fit(self, X_train, y_train, X_test, y_test, folder='./', overwrite=False):
        self.folder = folder  
    
    def _predict(self, X, labels, scalers):
        all_sites = np.unique(labels.sel(features='Site').values)
        with fsspec.open(self.folder + 'daily_data.csv') as f:
            era = pd.read_csv(f)
        
        cols = ['Site', 'TIMESTAMP_START', f'{self.label_name}_ERA']
        rename_map = {f'{self.label_name}_ERA': self.label_name}
        era = era[cols].rename(columns=rename_map)
        
        labels_df = pd.DataFrame(labels.values, columns=['Site', 'TIMESTAMP_START'])
        y_pred = labels_df.merge(era, on=['Site', 'TIMESTAMP_START'], how='left')
        return y_pred.set_index(['Site', 'TIMESTAMP_START']).reindex(labels_df[['Site', 'TIMESTAMP_START']]).values   
    
class base_xgb(base_model):
    def __init__(self, params, variable, name, eval_func):
        base_params = {
            'objective': 'reg:squarederror', 
            'eval_metric': 'rmse', 
            'learning_rate': 0.15,
#             'max_depth': 10,
            'n_estimators': 400,
            'random_state': 1
        }
        base_params.update(params)
        self.params = base_params
        self.label_name = variable
        self.eval_func = eval_func
        self.name = name 
        self.extension = '.bin'
        self.model = None
        print(f'Building {self.name} model')
        
    def _load(self, folder, overwrite):
        model_filename = folder + self.name + self.extension
        if os.path.exists(model_filename) and not overwrite:
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_filename)
        
        elif overwrite:
            self.model = None
        
    def fit(self, X_train, y_train, X_test, y_test, folder='./', overwrite=False):
        self._load(folder=folder, overwrite=overwrite)
        
        if self.model:
            print('    model already exists, loading model')
            return

        print('    fitting')
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train.values, y_train.values, 
            eval_set=[(X_train.values, y_train.values), (X_test.values, y_test.values)], 
            early_stopping_rounds=10
        )
        self._save(folder=folder, overwrite=overwrite)
        
    def _save(self, folder, overwrite):
        print('    saving model')
        model_filename = folder + self.name + self.extension
        if not os.path.exists(model_filename) or overwrite:
            self.model.save_model(model_filename)

    def _predict(self, X, labels, scalers):
        raw = self.model.predict(X.values, ntree_limit=self.model.best_ntree_limit)
        return scalers[self.label_name].inverse_transform(raw.reshape(-1, 1))

    
class xgb_default(base_xgb):
    def __init__(self, variable, eval_func):
        params = {}
        super().__init__(
            params=params, 
            variable=variable,
            name=f'{variable}_xgb_default',
            eval_func=eval_func,
        )
        
class xgb_hist(base_xgb):
    def __init__(self, variable, eval_func):
        params = {'tree_method': 'hist'}
        super().__init__(
            params=params, 
            variable=variable,
            name=f'{variable}_xgb_hist',
            eval_func=eval_func,
        )

        
class xgb_gpu_hist(base_xgb):
    def __init__(self, variable, eval_func):
        params = {'tree_method': 'gpu_hist'}
        super().__init__(
            params=params, 
            variable=variable,
            name=f'{variable}_xgb_gpu_hist',
            eval_func=eval_func,
        )

    
# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression  (only for Keras tensors)
def mse(y_true, y_pred):
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res / (SS_tot + backend.epsilon())


def bias(y_true, y_pred):
    return backend.mean(y_pred) - backend.mean(y_true)


class base_nn(base_model):
    def __init__(self, variable, name, eval_func):
        self.name = name 
        self.eval_func = eval_func

        self.metrics = [rmse, mse, r_square, bias]
        self.batch_size = 512
        
        self.label_name = variable
        self.model = None
        print(f'Building {self.name} model')
        
    def _load(self, folder, overwrite):
        model_filename = f"{folder}{self.name}.h5"
        if os.path.exists(model_filename) and not overwrite:
            self.model = load_model(model_filename, compile=False)
        
        elif overwrite:
            self.model = None
        
    def fit(self, X_train, y_train, X_test, y_test, folder='./', overwrite=False):
        self._load(folder=folder, overwrite=overwrite)
        
        if self.model:
            print('    model already exists, loading model')
            return

        print('    fitting')

        mc = ModelCheckpoint(
            f"{folder}{self.name}.h5",
            monitor="val_loss",
            mode="min",
            verbose=0,
            save_best_only=True,
        )
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=20)
        
        X_train = X_train.unstack('flatten_features').transpose("samples", "lookback", "features").values
        X_test = X_test.unstack('flatten_features').transpose("samples", "lookback", "features").values
        
        self.model = self.get_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        self.model.fit(
            X_train,
            y_train.values,
            validation_data=(X_test, y_test.values,),
            batch_size=self.batch_size,
            epochs=500,
            shuffle=True,
            callbacks=[es, mc],
        )

    def _predict(self, X, labels, scalers):
        X = X.unstack('flatten_features').transpose("samples", "lookback", "features").values
        raw = self.model.predict(X)
        return scalers[self.label_name].inverse_transform(raw.reshape(-1, 1))


class lstm_1_layer(base_nn):        
    def __init__(self, variable, eval_func):
        super().__init__(
            variable=variable,
            name=f"{variable}_lstm_1_layer",
            eval_func=eval_func,
        )
    
    def get_model(self, input_shape):
        model = Sequential(name=self.name)
        model.add(LSTM(20, input_shape=input_shape, use_bias=True))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=self.metrics)
        return model

    
class lstm_2_layer(base_nn):        
    def __init__(self, variable, eval_func):
        super().__init__(
            variable=variable,
            name=f"{variable}_lstm_2_layer",
            eval_func=eval_func,
        )
    
    def get_model(self, input_shape):
        model = Sequential(name=self.name)
        model.add(LSTM(20, input_shape=input_shape, use_bias=True, return_sequences=True))
        model.add(LSTM(20))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=self.metrics)
        return model

    
class lstm_3_layer(base_nn):        
    def __init__(self, variable, eval_func):
        super().__init__(
            variable=variable,
            name=f"{variable}_lstm_3_layer",
            eval_func=eval_func,
        )
    
    def get_model(self, input_shape):
        model = Sequential(name=self.name)
        model.add(LSTM(20, input_shape=input_shape, use_bias=True, return_sequences=True))
        model.add(LSTM(20, return_sequences=True))
        model.add(LSTM(20))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=self.metrics)
        return model

    
class lstm_3_layer_wide(base_nn):        
    def __init__(self, variable, eval_func):
        super().__init__(
            variable=variable,
            name=f"{variable}_lstm_3_layer_wide",
            eval_func=eval_func,
        )
    
    def get_model(self, input_shape):
        model = Sequential(name=self.name)
        model.add(LSTM(40, input_shape=input_shape, use_bias=True, return_sequences=True))
        model.add(LSTM(40, return_sequences=True))
        model.add(LSTM(40))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=self.metrics)
        return model

    
class gru_1_layer(base_nn):        
    def __init__(self, variable, eval_func):
        super().__init__(
            variable=variable,
            name=f"{variable}_gru_1_layer",
            eval_func=eval_func,
        )
    
    def get_model(self, input_shape):
        model = Sequential(name=self.name)
        model.add(GRU(20, input_shape=input_shape, use_bias=True))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=self.metrics)
        return model
    

class gru_2_layer(base_nn):        
    def __init__(self, variable, eval_func):
        super().__init__(
            variable=variable,
            name=f"{variable}_gru_2_layer",
            eval_func=eval_func,
        )
    
    def get_model(self, input_shape):
        model = Sequential(name=self.name)
        model.add(GRU(20, input_shape=input_shape, use_bias=True, return_sequences=True))
        model.add(GRU(20))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=self.metrics)
        return model
    
    
class gru_3_layer_wide(base_nn):
    def __init__(self, variable, eval_func):
        super().__init__(
            variable=variable,
            name=f"{variable}_gru_3_layer_wide",
            eval_func=eval_func,
        )
    
    def get_model(self, input_shape):
        model = Sequential(name=self.name)
        model.add(GRU(40, input_shape=input_shape, use_bias=True, return_sequences=True))
        model.add(GRU(40, return_sequences=True))
        model.add(GRU(40))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=self.metrics)
        return model

    
class attention_1_layer(base_nn):        
    def __init__(self, variable, eval_func):
        super().__init__(
            variable=variable,
            name=f"{variable}_att_1_layer",
            eval_func=eval_func,
        )
    
    def get_model(self, input_shape):
        inputs = Input(shape=input_shape)
        # positional encoding 
        x = LSTM(64, input_shape=input_shape, use_bias=True, return_sequences=True)(inputs)
        x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=self.metrics)
        return model 
