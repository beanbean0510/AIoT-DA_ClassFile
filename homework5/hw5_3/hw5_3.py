import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.optimizers import SGD  # 修正導入
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def create_model(model_name='vgg19', dropout_rate=0.3, num_classes=10):
    # 選擇基礎模型
    if model_name == 'vgg19':
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # 凍結前20層
    for layer in base_model.layers[:20]:
        layer.trainable = False
    
    # 建立模型
    model = models.Sequential([
        # 預處理層
        layers.Input(shape=(32, 32, 3)),
        layers.Resizing(224, 224),  # 更新為新的API
        layers.Normalization(
            mean=[123.68, 116.779, 103.939],
            variance=[58.393 ** 2, 57.12 ** 2, 57.375 ** 2]
        ),
        
        # 基礎模型
        base_model,
        
        # 分類層
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def prepare_data():
    # 載入CIFAR10數據集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # 將像素值標準化到 [0, 1] 範圍
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # One-hot 編碼標籤
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # 分割驗證集
    val_size = int(0.2 * len(x_train))
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def train_model(model_name='vgg19', batch_size=16, epochs=5, learning_rate=0.001):
    # 準備數據
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data()
    
    # 創建模型
    model = create_model(model_name)
    
    # 編譯模型
    model.compile(
        optimizer=SGD(learning_rate=learning_rate, momentum=0.9),  # 使用導入的SGD
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 回調函數
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=2,
            verbose=1,
            mode='min'
        )
    ]
    
    # 訓練模型
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 評估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f'\nTest accuracy: {test_acc:.4f}')
    
    return model, history

if __name__ == '__main__':
    # 訓練VGG19模型
    print("Training VGG19 model...")
    model_vgg19, history_vgg19 = train_model('vgg19')
    
    # 訓練VGG16模型
    print("\nTraining VGG16 model...")
    model_vgg16, history_vgg16 = train_model('vgg16')