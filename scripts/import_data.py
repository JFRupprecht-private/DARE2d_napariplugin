"""
Requirements

>>> pip install gdown
# Direct in bash or via python API
>>> gdown https://drive.google.com/drive/folders/1ZLF1isk-4_2CKgXdXUIBCkXVx5Jg0UA3 -O ./folder_Marc --folder --remaining-ok
"""

# Author: Romain and Marc

import os

import click
import gdown
import time

CELLS_DATABASE = {
    "2D_marc0": dict(url="https://drive.google.com/drive/folders/1ZLF1isk-4_2CKgXdXUIBCkXVx5Jg0UA3",
                     out_folder="2D/gastruloid/marc0"),
    "2D_marc1": dict(url="https://drive.google.com/drive/folders/1PsDMoneGp9GHcOT0SevOpWM4Rb9xh8bv",
                     out_folder="2D/gastruloid/marc1"),
    "2D_medhi_day2": dict(url="https://drive.google.com/drive/folders/1NcoEAcZGq4IPmWpT1ND0GtijlIUEFeyl",
                          out_folder="2D/gastruloid/medhi_day2"),  # film +
    "2D_medhi_day3": dict(url="https://drive.google.com/drive/folders/18Y-FTnvB27lZcxO1YtV0_xPk2UwkitQc",
                          out_folder="2D/gastruloid/medhi_day3"),  # day 1 et day 2 // film + split Marc
    "2D_corrected_Ro": dict(url="https://drive.google.com/drive/folders/19Nq0kqk3--W4rF2_TU-qtWqddeeL6IV1",
                            out_folder="2D/gastruloid/corrected_2D"),
}


def get_data_home(data_home=None):
    """Return the path of the dare2d data home directory.

    This folder is used by some large dataset loaders to avoid downloading
    data several times.

    By default, ``data_home`` is */dare2d/data/**

    Alternatively, it can be set by the **dare2d_DATA** environment variable or programmatically
    by giving an explicit folder path.

    Parameters
    ----------
    data_home : str | None
        The path to dare2d data dir.
    """
    this_filepath = __file__  # example dare2d_path = '/home/hcourtei/Projects/dare2d/dare2d/scripts'
    dare2d_root = os.path.dirname(
        os.path.dirname(this_filepath))  # sk_root = '/home/hcourtei/Projects/dare2d/dare2d'
    print("Root Dir for project: ", dare2d_root)
    if data_home is None:
        data_home = os.path.join(dare2d_root, "dataCells")
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


@click.command()
@click.option("--codename", default='2D_corrected_Ro', help="codename for dataset ")
@click.option("--quiet", default=False, help="True to supress verbosity")
@click.option("--data_home", default=None, help="rootdir/data or specify custom")
def fetch_cells_division_ds(codename, quiet=False, data_home=None):
    if codename not in CELLS_DATABASE:
        raise ValueError(f"codename {codename}  is not in dare2d database")
    else:
        info_dataset = CELLS_DATABASE[codename]
    data_home = data_home or get_data_home()
    dataset_dir = os.path.join(data_home, info_dataset["out_folder"])
    print(f"Dataset codename {codename} will be stored  : {dataset_dir}")
    if os.path.exists(dataset_dir):
        print(f"dataset {codename} already fetched in: {dataset_dir}")
    else:
        print(f"downloading {codename} in: {dataset_dir}")
        start = time.time()

        if "drive.google.com" in info_dataset["url"]:
            print("from google drive")
            gdown.download_folder(info_dataset["url"], output=dataset_dir, quiet=quiet, remaining_ok=True)
        else:
            print("from direct download link")
            os.system(f"""wget --directory-prefix={dataset_dir} {info_dataset['url']}""")
            print('Downloaded, now unzipping ')
            os.system(f"unzip -qq " + os.path.join(dataset_dir, info_dataset['url'].split('/')[-1]))
            print("deleting .zip")
            os.system(f"rm " + os.path.join(dataset_dir, info_dataset['url'].split('/')[-1]))
        duree = time.time() - start
        print(f"Dataset codename {codename} downladed in {round(duree / 60)} min in {dataset_dir}")
    return info_dataset


if __name__ == '__main__':
    info = fetch_cells_division_ds()
# python3 fetch_cells_division_ds.py --codename 2D_corrected_Ro
#   For debug
#   info = fetch_cells_division_ds('2D_corrected_Ro', quiet=False)

""" SIZE and TIME to DOWNLOAD

# 2D_corrected_Ro, 2 min, 105 items, totalling 337.7MB 
# 2D_marc0, 4 min, 224 items, totalling 960.8MB
# 2D_marc1, <1 min, 4 items, totalling 203.5MB
# 2D_medhi_day2, 6 min, 418 items, totalling 482.4MB
# 2D_medhi_day3, 9  min, 627 items, totalling 999.3MB

"""

""" dataCells directory
── 2D
│ └── gastruloid
│     ├── corrected_2D
│     │ ├── Exp_1
│     │ ├── Exp_1_val
│     │ ├── Exp_2
│     │ ├── Exp_2_val
│     │ ├── Exp_3
│     │ └── Exp_3_val
│     ├── marc0
│     │ ├── 2D TEST.rar
│     │ ├── Codes
│     │ ├── Image3_t1-100-Z9_contrast_bis_pred.tif
│     │ ├── Image3_t1-100-Z9_contrast_bis.tif
│     │ ├── test.h5
│     │ └── Train_set
│     ├── marc1
│     │ ├── Image 1-Z9.tif
│     │ ├── Image 3_t1-100-1.tif
│     │ ├── Image 3_t1-100-Z9.tif
│     │ └── MAX_Image 2-pos1-1-celldivisionlevel.tif
│     ├── medhi_day2
│     │ ├── ant
│     │ └── post
│     └── medhi_day3
│         ├── Ant
│         └── Post

"""
    └── gastruloid
        ├── alice10_03_2023
        │ ├── currimg
        │ ├── currlabel
        │ ├── div_location
        │ ├── nextimg
        │ ├── nextlabel
        │ ├── previmg
        │ └── prevlabel
        ├── alice13_03_2023
        │ ├── currimg
        │ ├── currlabel
        │ ├── div_location
        │ ├── nextimg
        │ └── previmg
        └── Deep_division3D
            ├── Codes
            ├── Compte Rendu.pdf
            └── Train_set

"""
