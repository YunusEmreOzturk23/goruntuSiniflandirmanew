# ===============================
# 1. KÜTÜPHANELER
# ===============================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt


# ===============================
# 2. SABİT DEĞERLER
# ===============================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = "fruits/train"
TEST_DIR = "fruits/test"


# ===============================
# 3. DATA AUGMENTATION
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_generator.num_classes


# ===============================
# 4. MODEL OLUŞTURMA FONKSİYONU
# ===============================
def create_model(filter_size=3, optimizer_name="adam"):
    model = Sequential()

    model.add(Conv2D(32, filter_size, activation="relu",
                     input_shape=(128, 128, 3)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    model.add(Conv2D(64, filter_size, activation="relu"))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    model.add(Conv2D(128, filter_size, activation="relu"))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=0.001)
    elif optimizer_name == "sgd":
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
    else:
        optimizer = RMSprop(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ===============================
# 5. DENEYLER
# ===============================
experiments = [
    {"filter": 3, "optimizer": "adam"},
    {"filter": 5, "optimizer": "adam"},
    {"filter": 3, "optimizer": "sgd"},
    {"filter": 3, "optimizer": "rmsprop"}
]

histories = {}


# ===============================
# 6. EĞİTİM
# ===============================
for exp in experiments:
    print("\n==============================")
    print(f"Model: Filter={exp['filter']} | Optimizer={exp['optimizer']}")
    print("==============================")

    model = create_model(
        filter_size=exp["filter"],
        optimizer_name=exp["optimizer"]
    )

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        verbose=1
    )

    key = f"filter={exp['filter']}_opt={exp['optimizer']}"
    histories[key] = history.history


# ===============================
# 7. PERFORMANS KARŞILAŞTIRMA
# ===============================
plt.figure(figsize=(10, 6))

for key in histories:
    plt.plot(histories[key]["val_accuracy"], label=key)

plt.title("Validation Accuracy Karşılaştırması")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
