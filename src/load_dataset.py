import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset():

    img_size=48
    batch_size=32
    
    path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")

    print("Path to dataset files:", path)

    train_dir = f"{path}/images/train"
    val_dir = f"{path}/images/validation"

    # data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_data = train_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_data, val_data

if __name__ == "__main__":
    # test loading of dataset
    train_data, val_data = load_dataset()
    print("Train batches:", len(train_data))
    print("Validation batches:", len(val_data))