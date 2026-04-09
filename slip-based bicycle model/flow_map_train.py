from __future__ import absolute_import, division, print_function
import os
import random as rn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Add, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import math

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        print(f"Folder '{folder}' already exists.")

def set_all_seeds(seed: int = 0):
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)

class FirstK(Layer):
    def __init__(self, k=6, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)

    def call(self, inputs):
        return inputs[:, :self.k]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"k": self.k})
        return cfg

def zscore_mse_factory(out_mu, out_std, eps=1e-8):
    """
    Compute MSE in standardized (z-scored) output space:
      loss = mean( ((y_true - out_mu)/out_std - (y_pred - out_mu)/out_std)^2 )
    """
    out_mu  = tf.constant(out_mu.reshape(1, -1), dtype=tf.float32)
    out_std = tf.constant((out_std.reshape(1, -1) + eps), dtype=tf.float32)

    def zscore_mse(y_true, y_pred):
        y_true_z = (y_true - out_mu) / out_std
        y_pred_z = (y_pred - out_mu) / out_std
        return tf.reduce_mean(tf.square(y_true_z - y_pred_z))
    return zscore_mse

def build_model(d_input, d_output, n_hidden, n_nodes, activation="tanh", lr=1e-3,
                use_residual_first6=True, loss="mse", out_mu=None, out_std=None):
    inputs = Input(shape=(d_input,), name="main_input")
    x = inputs
    for k in range(n_hidden):
        x = Dense(int(n_nodes[k]), activation=activation, name=f"dense_{k+1}")(x)
    x = Dense(d_output, name="out")(x)

    if use_residual_first6 and d_input >= 6 and d_output >= 6:
        first6_in = FirstK(k=6, name="first6_inputs")(inputs)
        outputs = Add(name="residual_add_first6_only")([x, first6_in])
    else:
        outputs = x

    model = Model(inputs=inputs, outputs=outputs, name="prior_net")

    # loss
    if isinstance(loss, str):
        loss_fn = loss
    else:
        # assume it's already a callable
        loss_fn = loss

    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss_fn)
    return model


class CyclicalLearningRate(Callback):
    """
    mode in {"triangular", "triangular2", "exp_range"}.
    step_size is in iterations
    """
    def __init__(self, base_lr, max_lr, step_size, mode="triangular2", gamma=1.0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.max_lr = float(max_lr)
        self.step_size = max(1.0, float(step_size))
        self.mode = mode
        self.gamma = float(gamma)
        self.iterations = 0
        self.history = {"lr": [], "iterations": [], "loss": []}

    def _scale_value(self, cycle):
        if self.mode == "triangular":
            s = 1.0
        elif self.mode == "triangular2":
            s = 2.0 ** (1.0 - float(cycle))
        elif self.mode == "exp_range":
            s = self.gamma ** float(self.iterations)
        else:
            s = 1.0
        if not np.isfinite(s):
            s = 1.0
        return float(np.clip(s, 0.0, 1.0))

    def clr(self):
        cycle = math.floor(1 + self.iterations / (2 * self.step_size))
        x = abs(self.iterations / self.step_size - 2 * cycle + 1)
        scale = max(0.0, 1.0 - x)
        lr_delta = (self.max_lr - self.base_lr) * scale
        s = self._scale_value(cycle)
        lr = self.base_lr + lr_delta * s
        return float(np.clip(lr, self.base_lr, self.max_lr))

    def _assign_lr(self, lr):
        opt = self.model.optimizer
        if hasattr(opt, "learning_rate") and hasattr(opt.learning_rate, "assign"):
            opt.learning_rate.assign(lr); return
        if hasattr(opt, "lr") and hasattr(opt.lr, "assign"):
            opt.lr.assign(lr); return
        if hasattr(opt, "learning_rate"):
            opt.learning_rate = lr
        else:
            opt.lr = lr

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.clr()
        self._assign_lr(lr)
        self.history["lr"].append(float(lr))
        self.history["iterations"].append(self.iterations)
        self.iterations += 1

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if "loss" in logs:
            try:
                self.history["loss"].append(float(logs["loss"]))
            except Exception:
                self.history["loss"].append(logs["loss"])

def train_prior(model, x_train, y_train, outdir, epochs, batch_size, clr_params=None):
    ckpt_path = os.path.join(outdir, "model-best.keras")
    cb_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        )
    ]

    if clr_params is not None:
        steps_per_epoch = int(np.ceil(len(x_train) / float(batch_size)))
        step_size = clr_params.get("step_size", 2 * steps_per_epoch)
        clr_cb = CyclicalLearningRate(
            base_lr=clr_params.get("base_lr", 1e-4),
            max_lr=clr_params.get("max_lr", 3e-3),
            step_size=step_size,
            mode=clr_params.get("mode", "triangular2"),
            gamma=clr_params.get("gamma", 0.99994),
        )
        cb_list.append(clr_cb)

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=cb_list,
        verbose=1
    )
    model.save(os.path.join(outdir, "model.keras"))

    # Plot loss
    plt.figure()
    plt.loglog(history.history["loss"], label="train")
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(outdir, "loss_semilog_prior.png"), bbox_inches="tight")
    plt.close()

    # Plot LR history if CLR is enabled
    if clr_params is not None:
        lrs = cb_list[-1].history["lr"]
        plt.figure()
        plt.plot(lrs)
        plt.title("Cyclical Learning Rate (per batch)")
        plt.ylabel("learning rate")
        plt.xlabel("iteration")
        plt.savefig(os.path.join(outdir, "clr_schedule.png"), bbox_inches="tight")
        plt.close()

    return model

### pred function 
def one_step_pred(model, x0, test_input, steps, d_of_x):
    n_test = x0.shape[0]
    pred = np.zeros((n_test, d_of_x * (steps+1)), dtype=np.float32)
    pred[:, 0:d_of_x] = x0
    for j in range(n_test):
        state = x0[j:j+1, :]
        for t in range(steps):
            param = test_input[t,:].reshape(1,-1)
            state = np.asarray(state, dtype=np.float32).reshape(1, -1)
            state = np.concatenate([state, param], axis=1)
            y_next = model.predict(state, verbose=0)
            pred[j, d_of_x*(t+1):d_of_x*(t+2)] = y_next
            state = y_next
    return pred

#load model 
def load_prior_model(model_dir, prefer_best=True, compile=False):
    candidates = []
    if prefer_best:
        candidates += ["model-best.keras", "model.keras", "model-best.h5", "model.h5"]
    path = None
    for fname in candidates:
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"No saved model found in {model_dir}. Tried: {', '.join(candidates)}")

    custom_objects = {"FirstK": FirstK}  # ensure custom layer is known
    try:
        model = tf.keras.models.load_model(path, compile=compile, custom_objects=custom_objects)
    except TypeError:
        model = tf.keras.models.load_model(path, compile=compile, custom_objects=custom_objects)
    except Exception as e:
        try:
            model = tf.keras.models.load_model(path, compile=compile, custom_objects=custom_objects, safe_mode=False)
        except TypeError:
            raise e
    return model


if __name__ == "__main__":
    N_seeds = 1
    d_of_x = 6
    n_params = 6
    d_input = d_of_x + n_params
    d_output = d_of_x
    n_hidden = 3
    n_nodes = [100, 100, 100]
    activation = "tanh"
    batch_size = 100
    n_epochs = 20000
    n_train = 100000

    # test
    n_test = 3
    time_pred = 1000
    Model_dir = "./"

    make_folder(Model_dir)
    print(Model_dir)

    # Load data
    prior_train = "./bike_bursts_T5_dt01_20pertraj_LF_5000_traj.mat"
    test_file = 'test_input_complex_bike_dt0.01_case3.mat'

    # Load data
    set_all_seeds(0)
    mat_input = sio.loadmat(prior_train)
    mat_output = sio.loadmat(prior_train)

    inputs_train = mat_input["inputs"][:n_train, :d_input].astype(np.float32)
    outputs_train = mat_output["outputs"][:n_train, :d_output].astype(np.float32)

    # shuffle
    perm = np.random.permutation(inputs_train.shape[0])
    inputs_train = inputs_train[perm, :]
    outputs_train = outputs_train[perm, :]

    out_mu  = outputs_train.mean(axis=0)
    out_std = outputs_train.std(axis=0)
    # avoid zero std
    out_std[out_std == 0] = 1.0

    # build model with Z-score loss
    zloss = zscore_mse_factory(out_mu, out_std)
    model = build_model(
        d_input, d_output, n_hidden, n_nodes,
        activation=activation, lr=1e-3,
        use_residual_first6=True,
        loss=zloss
    )

    # CLR settings 
    steps_per_epoch = int(np.ceil(len(inputs_train) / float(batch_size)))
    clr_params = {
        "base_lr": 1e-4,
        "max_lr": 3e-3,
        "step_size": 2 * steps_per_epoch,
        "mode": "triangular2",
        "gamma": 0.99994
    }

    ## uncomment for training 
    # model = train_prior(model, inputs_train, outputs_train, Model_dir, n_epochs, batch_size, clr_params=clr_params)

    ## Or reload best 
    model = load_prior_model(Model_dir, prefer_best=True, compile=False)

    ## test
    test_mat = sio.loadmat(test_file)
    test_signal = test_mat["test_input"].astype(np.float32)  # (T, 2)

    test_input = np.zeros((test_signal.shape[0], n_params), dtype=np.float32)
    test_input[:, :test_input.shape[1]] = test_signal

    # Initial condition (raw physical units, no normalization)
    x_test = np.array([[1., 1., 1., 1., 0., 0.]], dtype=np.float32)
    steps = min(time_pred, test_input.shape[0])
    # pred = one_step_pred(model, x_test, test_input,
    #                                   steps=steps, d_of_x=d_of_x)
    # sio.savemat(os.path.join(Model_dir, f"prior_pred.mat"),
    #             {"pred": pred})
    # print("saved prior prediction")

    #### MODEL CORRECTION ####
    def get_dense_layers(m):
        return [L for L in m.layers if isinstance(L, tf.keras.layers.Dense)]
    
    def freeze_all(m):
        for L in m.layers:
            L.trainable = False
    
    def make_callbacks(patience=100):
        return [tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=patience, restore_best_weights=True, verbose=1
        )]
        
    correction_train = 'bike_bursts_T5_dt01_20pertraj_HF_1000.mat'
    
    # load HF data 
    corr = sio.loadmat(correction_train)
    x_corr = corr["inputs"].astype(np.float32)
    y_corr      = corr["outputs"].astype(np.float32)

    # last layer
    model_last1 = model
    freeze_all(model_last1)
    
    dense_layers = get_dense_layers(model_last1)
    out_dense = dense_layers[-1]
    out_dense.trainable = True

    model_last1.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    print("Fine-tuning last layer only")
    model_last1.fit(x_corr, y_corr, batch_size=500, epochs=10000,
                    shuffle=True, validation_split=0.0, callbacks=make_callbacks(), verbose=1)
    
    
    # Save and prediction
    corrected_dir = os.path.join(Model_dir, "corrected_model_final")
    make_folder(corrected_dir)
    
    path_last1 = os.path.join(corrected_dir, "model_corrected_last_layer.keras")
    model_last1.save(path_last1)
    print(f"saved corrected (last1) model at {path_last1}")
    
    test_mat   = sio.loadmat(test_file)
    test_input = test_mat["test_input"].astype(np.float32)
    x_test     = np.array([[1., 1., 1., 1., 0., 0.]], dtype=np.float32)
    
    pred_last1 = one_step_pred(model_last1, x_test, test_input,
                                      steps=steps, d_of_x=d_of_x)
    
    sio.savemat(os.path.join(corrected_dir, f"corrected_pred_last_layer_T={time_pred}_ntest={n_test}_500_traj.mat"),
                {"pred": pred_last1})
    print("saved corrected (last layer) predictions")
    
    # test correction 2 layers
    model_last2 = model
    freeze_all(model_last2)
    
    dense_layers = get_dense_layers(model_last2)
    penult_dense = dense_layers[-2]
    out_dense    = dense_layers[-1]
    penult_dense.trainable = True
    out_dense.trainable    = True
    
    model_last2.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    print("Fine-tuning last 2 layers")
    model_last2.fit(x_corr, y_corr, batch_size=500, epochs=10000,
                    shuffle=True, validation_split=0.0, callbacks=make_callbacks(), verbose=1)
    
    # Save and prediction
    path_last2 = os.path.join(corrected_dir, "model_corrected_last_2_layers.keras")
    model_last2.save(path_last2)
    print(f"saved corrected (last 2 layers) model at {path_last2}")
    pred_last2 = one_step_pred(model_last2, x_test, test_input,
                                      steps=steps, d_of_x=d_of_x)
    sio.savemat(os.path.join(corrected_dir, f"corrected_pred_last_2_layers_T={time_pred}_ntest={n_test}_500_traj.mat"),
                {"pred": pred_last2})
    print("saved corrected (last 2 layers) predictions")
