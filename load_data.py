from pathlib import Path
import numpy as np

def load_data():
    data_path = "data/sisfall/Three Classes"
    N_CLASSES = 3   # "Non-fall", "Pre-fall", "Fall"
    CLASSES = ["Non-fall", "Pre-fall", "Fall"]

    N_CHANNELS = 6      # 3 from accelerometer and 3 from 
    WINDOW = 256 # duration of each 

    x_train = np.fromfile(data_path + "/x_train_3", dtype=np.float32)
    x_val   = np.fromfile(data_path + "/x_val_3", dtype=np.float32)
    x_test  = np.fromfile(data_path + "/x_test_3", dtype=np.float32)

    y_train = np.fromfile(data_path + "/y_train_3", dtype=np.int8)
    y_val   = np.fromfile(data_path + "/y_val_3", dtype=np.int8)
    y_test  = np.fromfile(data_path + "/y_test_3", dtype=np.int8)

    x_train = x_train.reshape((-1, WINDOW, N_CHANNELS)) # -1 so np auto cal  
    x_val   = x_val.reshape((-1, WINDOW, N_CHANNELS))
    x_test  = x_test.reshape((-1, WINDOW, N_CHANNELS))

    y_train = y_train.reshape((y_train.shape[0] // N_CLASSES , N_CLASSES))
    y_val   = y_val.reshape((y_val.shape[0] // N_CLASSES , N_CLASSES))
    y_test  = y_test.reshape((y_test.shape[0] // N_CLASSES), N_CLASSES)

    print(f"x_train's shape: {x_train.shape}")
    print(f"x_val's shape: {x_val.shape}")
    print(f"x_test's shape: {x_test.shape}")

    print(f"y_train's shape: {y_train.shape}")
    print(f"y_val's shape: {y_val.shape}")
    print(f"y_test's shape: {y_test.shape}")

    # x_train's shape: (77871, 256, 6)
    # x_val's shape: (20233, 256, 6)
    # x_test's shape: (18884, 256, 6)
    # y_train's shape: (77871, 3)
    # y_val's shape: (20233, 3)
    # y_test's shape: (18884, 3)

    # see the difference in each class
    for split, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        count = y.sum(axis=0).astype(int)
        print(f"{split} class count:", dict(zip(CLASSES, count)))
        
    return x_train, x_val, x_test, y_train, y_val, y_test
    
if __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()