"""Configuration for YOLOv11 passenger detection fine-tuning."""

# YOLO Model Configuration
YOLO_CONFIG = {
    "model": "yolo11m",  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    "task": "detect",  # detection task
    "device": 0,  # GPU device ID (0 = RTX 5070 Ti) - NOW SUPPORTED with PyTorch 2.11.0+cu128
}

# Training Configuration
TRAIN_CONFIG = {
    "epochs": 100,
    "imgsz": 640,  # image size
    "batch": 16,  # batch size
    "patience": 20,  # early stopping patience
    "device": 0,  # GPU device
    "workers": 4,  # DataLoader workers
    "optimizer": "SGD",  # optimizer: SGD, Adam, AdamW
    "lr0": 0.01,  # initial learning rate
    "lrf": 0.01,  # final learning rate ratio
    "momentum": 0.937,  # SGD momentum
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "hsv_h": 0.015,  # HSV-Hue augmentation
    "hsv_s": 0.7,    # HSV-Saturation augmentation
    "hsv_v": 0.4,    # HSV-Value augmentation
    "degrees": 0.0,  # rotation
    "translate": 0.1,  # translation
    "scale": 0.5,  # scale
    "flipud": 0.0,  # flip upside-down
    "fliplr": 0.5,  # flip left-right
    "mosaic": 1.0,  # mosaic augmentation
}

# Passenger Attributes
ATTRIBUTES = {
    "gender": {
        "classes": ["male", "female", "other"],
        "num_classes": 3,
    },
    "age_group": {
        "classes": ["0-2", "3-5", "6-12", "13-18", "19-30", "31-45", "46-60", "60+"],
        "num_classes": 8,
    },
    "height_range": {
        "classes": ["very_short", "short", "average", "tall", "very_tall"],
        "num_classes": 5,
    },
    "bmi_category": {
        "classes": ["underweight", "normal", "overweight", "obese"],
        "num_classes": 4,
    },
}

# Reinforcement Learning Configuration
RL_CONFIG = {
    "num_iterations": 10,
    "learning_rate_bounds": (1e-5, 1e-3),
    "momentum_bounds": (0.8, 0.99),
    "enable_ppo_clip": True,
    "ppo_clip_ratio": 0.2,
}

# Paths
DATA_DIR = "data/"
TRAIN_DATA = "data/train/"
VAL_DATA = "data/val/"
TEST_DATA = "data/test/"
MODELS_DIR = "data/models/"
RESULTS_DIR = "results/"

PATHS = {
    "data_dir": DATA_DIR,
    "train_data": TRAIN_DATA,
    "val_data": VAL_DATA,
    "test_data": TEST_DATA,
    "models_dir": MODELS_DIR,
    "results_dir": RESULTS_DIR,
}

# Inference
CONF_THRESHOLD = 0.5  # confidence threshold
IOU_THRESHOLD = 0.45  # IoU threshold
