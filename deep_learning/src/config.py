import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_folder = "F:\\PythonProject\\deep\\image_segmentation\\deep_learning\\data\\object_detection_segment\\segmentation"
data_folder = "F:\\PythonProject\\deep\\image_segmentation\\deep_learning\\data\\object_detection_segment\\object_detection/"
sr_data_folder = "../data/super_resolution/"
checkpoint = "../checkpoing/"
sr_checkpoint = "../data/chapter_three/sr.pth"

batch_size = 8
lr = 1e-3
epoch_lr = [(20,0.01),(10,0.001),(10,0.0001)]