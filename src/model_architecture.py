
"""
Neural network architecture definitions for glucose prediction.
"""

# Configure GPUs to grow memory as needed
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Custom loss function: adds an extra penalty for false negatives, pushing the model to focus more on recall.
def recall_focused_loss(y_true, y_pred):
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    # Regular binary cross-entropy
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    
    # False negative penalty (weighs missed positive examples more heavily)
    fn_weight = 2  # weight to control recall focus
    fn_penalty = fn_weight * y_true * (1 - y_pred) ** 2
    
    return bce + tf.keras.backend.mean(fn_penalty)

# Attention mechanism
def attention_block(x):
    attention = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    attention = BatchNormalization()(attention)
    attention = GlobalMaxPooling2D()(attention)
    attention = Dense(x.shape[-1], activation='sigmoid')(attention)
    attention = Reshape((1, 1, x.shape[-1]))(attention)
    attention = Multiply()([x, attention])
    return attention

# Efficient channel attention mechanism
def efficient_attention(x, reduction=16):
    channel_dim = x.shape[-1]
    # Squeeze and excitation pattern
    se = GlobalAveragePooling2D()(x)
    se = Dense(channel_dim // reduction, activation='relu')(se)
    se = Dense(channel_dim, activation='sigmoid')(se)
    se = Reshape((1, 1, channel_dim))(se)
    return Multiply()([x, se])
    
def create_model(matrix_shape0, dailyshape, individualshape):
    # Define the CNN Model with improved architecture
    input_matrix = Input(shape=(matrix_shape0, 288, 2), name='input_matrix')
    input_features_daily = Input(shape=(dailyshape,), name='input_features_daily')
    input_features_individual = Input(shape=(individualshape,), name='input_features_individual')
    
    # CNN pathway for wavelet scalograms
    conv1 = Conv2D(32, kernel_size=(5, 5), strides=(2, 2), activation='relu', 
                  kernel_regularizer=l2(0.001))(input_matrix)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = efficient_attention(conv1) 
    
    conv2 = Conv2D(64, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', 
                  kernel_regularizer=l2(0.001))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', 
                  kernel_regularizer=l2(0.001))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Flatten and process CNN features
    flat = Flatten()(conv3)
    dense_flat = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(flat)
    densed_fully = Dense(512, activation='relu')(dense_flat)
    densed_fully = Dropout(0.5)(densed_fully)
    
    # Process daily features with implicit NaN handling
    dense_daily = Dense(128, activation='relu')(input_features_daily)
    dense_daily = BatchNormalization()(dense_daily)
    dense_daily = Dropout(0.3)(dense_daily)
    
    # Process individual features with implicit NaN handling
    dense_individual = Dense(128, activation='relu')(input_features_individual)
    dense_individual = BatchNormalization()(dense_individual)
    dense_individual = Dropout(0.3)(dense_individual)
    
    # Combine contextual features
    ctx_features = concatenate([dense_daily, dense_individual])
    ctx_attention = Dense(256, activation='relu')(ctx_features)
    ctx_attention = Dense(256, activation='sigmoid')(ctx_attention)
    ctx_gated = Multiply()([ctx_features, ctx_attention])
    
    # Simple concatenation of all features (based on empirical evidence)
    concat_all = concatenate([densed_fully,dense_daily,dense_individual])
    
    # Final dense layers with dropout for regularization
    fc_combined = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(concat_all)
    fc_combined = Dropout(0.6)(fc_combined)
    
    fc_combined = Dense(128, activation='relu')(fc_combined)
    fc_combined = Dropout(0.7)(fc_combined)
    
    fc_combined = Dense(128, activation='relu')(fc_combined)
    
    # Output layer
    output = Dense(1, activation='sigmoid', dtype='float32')(fc_combined)
    
    # Define model
    model = Model(inputs=[input_matrix, input_features_daily, input_features_individual], 
                 outputs=output)
    
    # Compile with recall-focused loss
    model.compile(optimizer=Adam(learning_rate=2e-4), 
                 loss=recall_focused_loss, 
                 metrics=['accuracy', 'AUC', tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    
    return model


