import random
from pathlib import Path
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def count_images(path):
    files = Path(path).glob("*") #all files
    c = 0

    for f in files:
        if ".jpg" in f.name or ".png" in f.name: #only img
            c += 1

    return c


def check_labels(img_path, label_path):
    imgs = list(Path(img_path).glob("*")) 
    missing = []

    for f in imgs:
        if ".jpg" in f.name or ".png" in f.name:
            txt = Path(label_path) / (f.stem + ".txt")

            if not txt.exists():
                missing.append(f.name) #no labels

    print("missing:", len(missing))


def random_img(path):
    files = list(Path(path).glob("*"))

    if len(files) == 0:
        return None

    return random.choice(files)

if __name__ == "__main__":
    print("train images:", count_images("data/images/train"))

