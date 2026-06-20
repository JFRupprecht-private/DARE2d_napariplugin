import os
import shutil

# data_folder = '../data/3D_DivDetect/train_8_06_2023' # 105 img from 0.tif to 104.tif
# SPLIT with ratio 75% train  25% val : according to number of division per image
# nb_div    counts       val       train
# 1        69 img            17
# 2        30 img             7
# 3         6 img             2
# 5         1 img             0
#                 TOT         26      79
all_val_img = "2 3 7 9 12 17 19 22 30 31 36 40 45 47 52 57 59 63 64 67 73 80 84 87 96 99"

data_folder = os.path.join("..", "data")
data_output = os.path.join(data_folder, "3d")
data_folder_raw = os.path.join(data_folder, "3d", "train_8_06_2023")

if not os.path.exists(data_folder_raw):
    raise ValueError(
        f"The path specified for the training dataset does not exist: {data_folder_raw}")

os.makedirs(data_output, exist_ok=True)

list_val_img = list(map(int, all_val_img.split(" ")))
for folder in ['train_8_06_val', 'train_8_06_train']:
    os.makedirs(os.path.join(
        data_output, f'{folder}', 'div_location'), exist_ok=True)
    os.makedirs(os.path.join(
        data_output, f'{folder}', 'currimg'), exist_ok=True)
    os.makedirs(os.path.join(
        data_output, f'{folder}', 'previmg'), exist_ok=True)
    os.makedirs(os.path.join(
        data_output, f'{folder}', 'nextimg'), exist_ok=True)
    os.makedirs(os.path.join(
        data_output, f'{folder}', 'currlabel'), exist_ok=True)


for idx_img in range(105):
    if idx_img in list_val_img:
        folder = "train_8_06_val"
    else:
        folder = "train_8_06_train"

    shutil.copy2(os.path.join(data_folder_raw, f'./div_location/{str(idx_img)}.npy'),
                 os.path.join(data_output, f'{folder}/div_location'))
    shutil.copy2(os.path.join(data_folder_raw,
                 f't_img/{str(idx_img)}.tif'), os.path.join(data_output, f'{folder}/currimg'))
    shutil.copy2(os.path.join(data_folder_raw,
                 f't-1_img/{str(idx_img)}.tif'), os.path.join(data_output, f'{folder}/previmg'))
    shutil.copy2(os.path.join(data_folder_raw,
                 f't+1_img/{str(idx_img)}.tif'), os.path.join(data_output, f'{folder}/nextimg'))
    shutil.copy2(os.path.join(data_folder_raw,
                 f't+1_label/{str(idx_img)}.tif'), os.path.join(data_output, f'{folder}/currlabel'))
