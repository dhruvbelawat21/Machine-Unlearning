import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm
from Unlearner.DNNUnlearner import DNNUnlearner

N_ROWS = 32
N_CHANNELS = 3


class CNNUnlearner(DNNUnlearner):
    """
    CNN-based unlearner tailored for MNIST/Fashion-MNIST (28x28 grayscale)
    and safely loads weights from a poisoned model.
    """

    def __init__(self, train, test, valid, weight_path=None, lambda_=1e-5,
                 n_filters=None, pooling_type='max'):
        self.x_train, self.y_train = train
        self.x_test, self.y_test = test
        self.x_valid, self.y_valid = valid
        self.lambda_ = lambda_
        self.n = self.x_train.shape[0]
        self.N_ROWS = self.x_train.shape[1]
        self.N_CHANNELS = self.x_train.shape[3]

        # Default filter configuration
        self.n_filters = n_filters or [32, 32, 64, 64, 128, 128]
        self.pooling_type = pooling_type.lower()

        self.model = self.get_network(weight_path)

    def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
        conv_params = dict(activation='relu', kernel_size=3,
                           kernel_initializer='he_uniform', padding='same')

        model = Sequential()

        # First block
        model.add(Conv2D(filters=self.n_filters[0],
                         input_shape=(self.N_ROWS, self.N_ROWS, self.N_CHANNELS),
                         **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=self.n_filters[1], **conv_params))
        model.add(BatchNormalization())
        model.add(self._get_pooling_layer())
        model.add(Dropout(0.1))

        # Second block
        model.add(Conv2D(filters=self.n_filters[2], **conv_params))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=self.n_filters[3], **conv_params))
        model.add(BatchNormalization())
        model.add(self._get_pooling_layer())
        model.add(Dropout(0.1))

        # Third block
        if len(self.n_filters) > 4:
            model.add(Conv2D(filters=self.n_filters[4], **conv_params))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=self.n_filters[5], **conv_params))
            model.add(BatchNormalization())
            model.add(self._get_pooling_layer())
            model.add(Dropout(0.2))

        # Dense layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(units=10, activation='softmax'))

        # Compile
        if optimizer.lower() == 'adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate, amsgrad=True, epsilon=0.1),
                          loss=categorical_crossentropy, metrics=['accuracy'])
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate),
                          loss=categorical_crossentropy, metrics=['accuracy'])

        # Load weights if provided
        if weight_path is not None:
            try:
                model.load_weights(weight_path)
                print(f"[INFO] Weights loaded from {weight_path}")
            except ValueError as e:
                print(f"[WARNING] Could not load weights due to shape mismatch: {e}")
                print("[INFO] Model initialized with random weights instead.")

        model.summary()
        return model

    def _get_pooling_layer(self):
        return MaxPooling2D(pool_size=(2, 2)) if self.pooling_type == 'max' else AveragePooling2D(pool_size=(2, 2))
    def explain_prediction(self, x, y, deletion_size=1, **kwargs):
        """
        Computes influence scores for input x and label y using 
        the trained CNN model and the LiSSA approximation of
        the inverse Hessian-vector product.
        """
        assert self.N_ROWS % deletion_size == 0, "N_ROWS must be divisible by deletion_size"

        batch_size = kwargs.get('batch_size', 500)
        rounds = kwargs.get('rounds', 1)
        scale = kwargs.get('scale', 75000)
        damping = kwargs.get('damping', 1e-2)
        verbose = kwargs.get('verbose', False)

        relevances = np.zeros_like(x)

        # Gradient of loss wrt parameters for the test point(s)
        grad_x = self.get_gradients(x, y)

        # Compute approximate inverse Hessian-vector product
        H_inv_grad_x, diverged = self.get_inv_hvp_lissa(
            self.x_train,          # full training data
            self.y_train,          # full training labels
            grad_x,                # vector v
            batch_size=batch_size,
            scale=scale,
            damping=damping,
            iterations=-1,         # full pass unless limited
            verbose=verbose,
            rounds=rounds
        )

        print(f'Calculating influence for {self.N_ROWS / deletion_size} rows')

        for i in tqdm(range(0, self.N_ROWS, deletion_size)):
            for j in range(0, self.N_ROWS, deletion_size):
                # Define the square block to delete
                square_to_delete = np.array(
                    np.meshgrid(
                        range(i, i + deletion_size),
                        range(j, j + deletion_size)
                    )
                ).T.reshape(-1, 2)

                # Get relevant training indices that affect this block
                relevant_indices = list(self.get_relevant_indices(square_to_delete))
                if len(relevant_indices) == 0:
                    relevances[0, i:i + deletion_size, j:j + deletion_size] = 0
                    continue

                # Limit number of relevant indices for efficiency
                if len(relevant_indices) > 256:
                    relevant_indices = np.random.choice(relevant_indices, 256, replace=False)

                # Create zeroed version of training data for influence calculation
                x_train_zero = self.x_train[relevant_indices].copy()
                x_train_zero[:, i:i + deletion_size, j:j + deletion_size] = 0

                # Compute gradients before and after zeroing
                d_L_z = self.get_gradients(self.x_train[relevant_indices], self.y_train[relevant_indices])
                d_L_z_delta = self.get_gradients(x_train_zero, self.y_train[relevant_indices])

                # Compute difference and multiply by inverse Hessian-vector product
                diff = [dLd - dL for dLd, dL in zip(d_L_z_delta, d_L_z)]
                loss_diff_per_param = [np.sum(d * Hd) for d, Hd in zip(diff, H_inv_grad_x)]

                # Assign influence to the block
                relevances[0, i:i + deletion_size, j:j + deletion_size] = np.sum(loss_diff_per_param) / self.n

        return relevances, diverged
    def remove_influential_points(self, x_del, y_del, scale=1.0, damping=1e-2, batch_size=500, verbose=False):
        """
        Performs gradient-based unlearning of the provided deletion set (x_del, y_del)
        by adjusting the model weights using influence functions, without full retraining.

        Args:
            x_del: Samples to unlearn (shape: [N, H, W, C])
            y_del: Labels of samples to unlearn (one-hot, shape: [N, num_classes])
            scale: LiSSA scale parameter for inverse Hessian-vector approximation
            damping: LiSSA damping factor
            batch_size: batch size for gradient calculations
            verbose: Print progress
        """

        print(f"[INFO] Starting gradient-based unlearning for {len(x_del)} samples...")

        # 1. Compute gradient of loss w.r.t. model params for the deletion set
        grad_del = self.get_gradients(x_del, y_del)

        # 2. Approximate inverse Hessian-vector product (IHVP) on training set
        H_inv_grad_del, diverged = self.get_inv_hvp_lissa(
            self.x_train, self.y_train,
            grad_del,
            batch_size=batch_size,
            scale=scale,
            damping=damping,
            iterations=-1,
            verbose=verbose,
            rounds=1
        )

        if diverged:
            print("[WARNING] IHVP computation diverged; results may be inaccurate.")

        # 3. Update model parameters to remove influence
        weights = self.model.trainable_weights
        for w, h_inv in zip(weights, H_inv_grad_del):
            w.assign_sub(h_inv)  # θ_new = θ_old - IHVP

        print("[DONE] Gradient-based unlearning applied successfully.")


        # The remaining methods (get_data_copy, get_relevant_indices, explain_prediction)
        # remain unchanged, they automatically use self.x_train/self.y_train

    def unlearn_by_gradient_negation(self, forget_label=1, steps=5, lr=1e-4, batch_size=128, verbose=True):
        """
        Fast unlearning method based on gradient negation.
        Instead of computing influence functions, this performs 'anti-training'
        on all samples with the specified label.

        Args:
            forget_label (int): The label to unlearn (e.g., 1 for digit '1').
            steps (int): Number of unlearning (anti-training) iterations.
            lr (float): Learning rate for gradient negation steps.
            batch_size (int): Batch size for unlearning.
            verbose (bool): Print progress info.
        """
        import tensorflow as tf

        # Filter samples with the target label
        mask = np.argmax(self.y_train, axis=1) == forget_label
        x_forget = self.x_train[mask]
        y_forget = self.y_train[mask]

        if len(x_forget) == 0:
            print(f"[WARNING] No samples found for label {forget_label}. Nothing to unlearn.")
            return

        if verbose:
            print(f"[INFO] Starting gradient-negation unlearning for label {forget_label} ...")
            print(f"       Found {len(x_forget)} samples to unlearn.")

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                preds = self.model(x_batch, training=True)
                loss = tf.keras.losses.categorical_crossentropy(y_batch, preds)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.model.trainable_variables)

            # NEGATE gradients (anti-learning step)
            neg_grads = [-1.0 * g for g in grads]
            optimizer.apply_gradients(zip(neg_grads, self.model.trainable_variables))
            return loss

        # Perform unlearning steps
        for epoch in range(steps):
            indices = np.arange(len(x_forget))
            np.random.shuffle(indices)
            x_forget = x_forget[indices]
            y_forget = y_forget[indices]

            total_loss = 0
            for i in range(0, len(x_forget), batch_size):
                xb = x_forget[i:i + batch_size]
                yb = y_forget[i:i + batch_size]
                loss = train_step(xb, yb)
                total_loss += loss.numpy()

            if verbose:
                print(f"[Epoch {epoch+1}/{steps}] Unlearning loss: {total_loss / len(x_forget):.6f}")

        print("[INFO] Gradient-negation unlearning complete.")






# class CNNUnlearner(DNNUnlearner):
#     """
#     CNN-based unlearner that can adapt to different architectures
#     and safely load weights from a poisoned model.
#     """

#     def __init__(self, train, test, valid, weight_path=None, lambda_=1e-5,
#                  n_filters=None, pooling_type='max'):
#         self.x_train, self.y_train = train
#         self.x_test, self.y_test = test
#         self.x_valid, self.y_valid = valid
#         self.lambda_ = lambda_
#         self.n = self.x_train.shape[0]
#         self.dim = self.x_train.shape[1]

#         # Default filter configuration if not provided
#         self.n_filters = n_filters or [32, 32, 64, 64, 128, 128]
#         self.pooling_type = pooling_type.lower()

#         self.model = self.get_network(weight_path)

#     def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
#         conv_params = dict(activation='relu', kernel_size=3,
#                            kernel_initializer='he_uniform', padding='same')

#         model = Sequential()

#         # First block
#         model.add(Conv2D(filters=self.n_filters[0],
#                          input_shape=(N_ROWS, N_ROWS, N_CHANNELS),
#                          **conv_params))
#         model.add(BatchNormalization())
#         model.add(Conv2D(filters=self.n_filters[1], **conv_params))
#         model.add(BatchNormalization())
#         model.add(self._get_pooling_layer())
#         model.add(Dropout(0.1))

#         # Second block
#         model.add(Conv2D(filters=self.n_filters[2], **conv_params))
#         model.add(BatchNormalization())
#         model.add(Conv2D(filters=self.n_filters[3], **conv_params))
#         model.add(BatchNormalization())
#         model.add(self._get_pooling_layer())
#         model.add(Dropout(0.1))

#         # Third block
#         if len(self.n_filters) > 4:
#             model.add(Conv2D(filters=self.n_filters[4], **conv_params))
#             model.add(BatchNormalization())
#             model.add(Conv2D(filters=self.n_filters[5], **conv_params))
#             model.add(BatchNormalization())
#             model.add(self._get_pooling_layer())
#             model.add(Dropout(0.2))

#         # Dense layers
#         model.add(Flatten())
#         model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.3))
#         model.add(Dense(units=10, activation='softmax'))

#         # Compile
#         if optimizer.lower() == 'adam':
#             model.compile(optimizer=Adam(learning_rate=learning_rate, amsgrad=True, epsilon=0.1),
#                           loss=categorical_crossentropy, metrics=['accuracy'])
#         else:
#             model.compile(optimizer=SGD(learning_rate=learning_rate),
#                           loss=categorical_crossentropy, metrics=['accuracy'])

#         # Load weights safely
#         if weight_path is not None:
#             try:
#                 model.load_weights(weight_path)
#                 print(f"[INFO] Weights loaded from {weight_path}")
#             except ValueError as e:
#                 print(f"[WARNING] Could not load weights due to shape mismatch: {e}")
#                 print("[INFO] Model initialized with random weights instead.")

#         model.summary()
#         return model

#     def _get_pooling_layer(self):
#         return MaxPooling2D(pool_size=(2, 2)) if self.pooling_type == 'max' else AveragePooling2D(pool_size=(2, 2))

#     def get_data_copy(self, data_name, indices_to_delete, **kwargs):
#         assert data_name in ['train', 'test', 'valid']
#         affected_samples = kwargs.get('affected_samples', None)
#         if data_name == 'train':
#             data_cpy = self.x_train.copy()
#         elif data_name == 'test':
#             data_cpy = self.x_test.copy()
#         else:
#             data_cpy = self.x_valid.copy()

#         if len(indices_to_delete) > 0:
#             if affected_samples is not None:
#                 for idx in affected_samples:
#                     data_cpy[idx, indices_to_delete[:, 0], indices_to_delete[:, 1], :] = 0
#             else:
#                 data_cpy[:, indices_to_delete[:, 0], indices_to_delete[:, 1], :] = 0
#         return data_cpy

#     def get_relevant_indices(self, indices_to_delete):
#         relevant_rows = set()
#         x_train_channel_sum = np.sum(self.x_train, axis=3)
#         for coordinate in indices_to_delete:
#             nz = x_train_channel_sum[:, coordinate[0], coordinate[1]].nonzero()[0]
#             relevant_rows.update(nz)
#         return list(relevant_rows)

#     def explain_prediction(self, x, y, deletion_size=1, **kwargs):
#         assert N_ROWS % deletion_size == 0
#         batch_size = kwargs.get('batch_size', 500)
#         rounds = kwargs.get('rounds', 1)
#         scale = kwargs.get('scale', 75000)
#         damping = kwargs.get('damping', 1e-2)
#         verbose = kwargs.get('verbose', False)

#         relevances = np.zeros_like(x)

#         # v = gradient of loss wrt params for the test point(s)
#         grad_x = self.get_gradients(x, y)

#         # Correct call: pass training data + labels + v
#         H_inv_grad_x, diverged = self.get_inv_hvp_lissa(
#             self.x_train,          # full training data
#             self.y_train,          # full training labels
#             grad_x,                # vector v
#             batch_size=batch_size,
#             scale=scale,
#             damping=damping,
#             iterations=-1,         # use full pass unless you want to limit
#             verbose=verbose,
#             rounds=rounds
#         )

#         print(f'Calculating influence for {N_ROWS/deletion_size} rows')
#         for i in tqdm(range(0, N_ROWS, deletion_size)):
#             for j in range(0, N_ROWS, deletion_size):
#                 square_to_delete = np.array(
#     np.meshgrid(
#         range(i, i + deletion_size),
#         range(j, j + deletion_size)
#     )
# ).T.reshape(-1, 2)

#                 relevant_indices = self.get_relevant_indices(square_to_delete)
#                 if not relevant_indices:
#                     relevances[0, i:i + deletion_size, j:j + deletion_size] = 0
#                     continue

#                 if len(relevant_indices) > 256:
#                     relevant_indices = np.random.choice(relevant_indices, 256, replace=False)

#                 x_train_zero = self.x_train[relevant_indices].copy()
#                 x_train_zero[:, i:i + deletion_size, j:j + deletion_size] = 0

#                 d_L_z = self.get_gradients(self.x_train[relevant_indices],
#                                         self.y_train[relevant_indices])
#                 d_L_z_delta = self.get_gradients(x_train_zero,
#                                                 self.y_train[relevant_indices])

#                 diff = [dLd - dL for dLd, dL in zip(d_L_z_delta, d_L_z)]
#                 loss_diff_per_param = [np.sum(d * Hd) for d, Hd in zip(diff, H_inv_grad_x)]
#                 relevances[0, i:i + deletion_size, j:j + deletion_size] = (
#                     np.sum(loss_diff_per_param) / self.n
#                 )

#         return relevances, diverged

#     # def explain_prediction(self, x, y, deletion_size=1, **kwargs):
#     #     assert N_ROWS % deletion_size == 0
#     #     batch_size = kwargs.get('batch_size', 500)
#     #     rounds = kwargs.get('rounds', 1)
#     #     scale = kwargs.get('scale', 75000)
#     #     damping = kwargs.get('damping', 1e-2)
#     #     verbose = kwargs.get('verbose', False)

#     #     relevances = np.zeros_like(x)
#     #     grad_x = self.get_gradients(x, y)
#     #     H_inv_grad_x, diverged = self.get_inv_hvp_lissa(grad_x, batch_size, scale, damping, verbose, rounds)

#     #     print(f'Calculating influence for {N_ROWS/deletion_size} rows')
#     #     for i in tqdm(range(0, N_ROWS, deletion_size)):
#     #         for j in range(0, N_ROWS, deletion_size):
#     #             square_to_delete = np.array(np.meshgrid(range(i, i+deletion_size),
#     #                                                     range(j, j+deletion_size))).T.reshape(-1, 2)
#     #             relevant_indices = self.get_relevant_indices(square_to_delete)
#     #             if not relevant_indices:
#     #                 relevances[0, i:i+deletion_size, j:j+deletion_size] = 0
#     #                 continue

#     #             if len(relevant_indices) > 256:
#     #                 relevant_indices = np.random.choice(relevant_indices, 256, replace=False)

#     #             x_train_zero = self.x_train[relevant_indices].copy()
#     #             x_train_zero[:, i:i+deletion_size, j:j+deletion_size] = 0

#     #             d_L_z = self.get_gradients(self.x_train[relevant_indices], self.y_train[relevant_indices])
#     #             d_L_z_delta = self.get_gradients(x_train_zero, self.y_train[relevant_indices])
#     #             diff = [dLd - dL for dLd, dL in zip(d_L_z_delta, d_L_z)]
#     #             loss_diff_per_param = [np.sum(d * Hd) for d, Hd in zip(diff, H_inv_grad_x)]
#     #             relevances[0, i:i+deletion_size, j:j+deletion_size] = np.sum(loss_diff_per_param) / self.n

#     #     return relevances, diverged



class CNNUnlearnerMedium(CNNUnlearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
        n_filters = [64, 64, 128, 128, 256, 256]
        kernel_size = 3
        model = Sequential()
        model.add(Conv2D(filters=n_filters[0], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same', input_shape=(N_ROWS, N_ROWS, N_CHANNELS)))
        model.add(Conv2D(filters=n_filters[1], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Conv2D(filters=n_filters[2], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(Conv2D(filters=n_filters[3], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Conv2D(filters=n_filters[4], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(Conv2D(filters=n_filters[5], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_regularizer=L2(self.lambda_)))
        model.add(Dropout(0.6))
        # final dense layer
        model.add(Dense(units=10, activation='softmax', kernel_regularizer=L2(self.lambda_)))
        if optimizer == 'Adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
                          loss=categorical_crossentropy, metrics='accuracy')
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate), loss=categorical_crossentropy, metrics='accuracy')
        if weight_path is not None:
            model.load_weights(weight_path)
        # print(model.summary())
        return model


class CNNUnlearnerSmall(CNNUnlearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
        n_filters = [64, 64, 128, 128]
        kernel_size = 3
        model = Sequential()
        model.add(Conv2D(filters=n_filters[0], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same', input_shape=(N_ROWS, N_ROWS, N_CHANNELS)))
        model.add(Conv2D(filters=n_filters[1], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Conv2D(filters=n_filters[2], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(Conv2D(filters=n_filters[3], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=L2(self.lambda_)))
        model.add(Dropout(0.6))
        # final dense layer
        model.add(Dense(units=10, activation='softmax', kernel_regularizer=L2(self.lambda_)))
        if optimizer == 'Adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
                          loss=categorical_crossentropy, metrics='accuracy')
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate), loss=categorical_crossentropy, metrics='accuracy')
        if weight_path is not None:
            model.load_weights(weight_path)
        print(model.summary())
        return model


class CNNUnlearnerAvgPooling(CNNUnlearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_network(self, weight_path=None, optimizer='Adam', learning_rate=0.0001):
        n_filters = [64, 64, 128, 128]
        kernel_size = 3
        model = Sequential()
        model.add(Conv2D(filters=n_filters[0], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same', input_shape=(N_ROWS, N_ROWS, N_CHANNELS)))
        model.add(Conv2D(filters=n_filters[1], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Conv2D(filters=n_filters[2], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(Conv2D(filters=n_filters[3], kernel_size=kernel_size, activation='relu',
                         kernel_regularizer=L2(self.lambda_), padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=L2(self.lambda_)))
        model.add(Dropout(0.6))
        # final dense layer
        model.add(Dense(units=10, activation='softmax', kernel_regularizer=L2(self.lambda_)))
        if optimizer == 'Adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
                          loss=categorical_crossentropy, metrics='accuracy')
        else:
            model.compile(optimizer=SGD(learning_rate=learning_rate), loss=categorical_crossentropy, metrics='accuracy')
        if weight_path is not None:
            model.load_weights(weight_path)
        print(model.summary())
        return model
