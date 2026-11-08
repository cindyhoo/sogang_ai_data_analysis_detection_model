"""데이터셋 레지스트리 (제안서 §4.2).

PoC에서는 CIFAR-100만 쓰지만, 데이터셋을 추가할 때 DatasetSpec만 등록하면 되도록 구조를 잡아 둔다.
fine(100클래스)과 coarse(20클래스)는 같은 원본 파일을 쓰므로 다운로드는 한 번이면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core import config
from app.core.fsutil import read_json
from app.core.i18n import AppError

# keras.datasets.cifar100은 클래스 이름을 돌려주지 않으므로 원본 meta 파일의 순서대로 상수로 둔다.
# (prepare 작업에서 내려받은 meta 파일과 일치하는지 다시 확인한다.)
CIFAR100_FINE = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)
CIFAR100_COARSE = (
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
    "medium_mammals", "non-insect_invertebrates", "people", "reptiles", "small_mammals", "trees",
    "vehicles_1", "vehicles_2",
)
assert len(CIFAR100_FINE) == 100 and len(CIFAR100_COARSE) == 20


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    display_name: str
    description: str
    num_classes: int
    label_mode: str
    input_shape: tuple
    class_names: tuple
    source: str = "keras.datasets.cifar100.load_data"

    def load(self):
        """(x_train, y_train), (x_test, y_test). x는 uint8 (N,32,32,3), y는 1차원 int64."""
        from keras.datasets import cifar100

        (xtr, ytr), (xte, yte) = cifar100.load_data(label_mode=self.label_mode)
        return (xtr, ytr.reshape(-1).astype("int64")), (xte, yte.reshape(-1).astype("int64"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "num_classes": self.num_classes,
            "label_mode": self.label_mode,
            "input_shape": list(self.input_shape),
            "source": self.source,
        }


DATASETS: dict[str, DatasetSpec] = {
    spec.id: spec
    for spec in (
        DatasetSpec(
            id="cifar100_fine",
            display_name="CIFAR-100 (fine, 100 classes)",
            description="100개 세부 클래스. 클래스당 train 500장 / test 100장.",
            num_classes=100,
            label_mode="fine",
            input_shape=(32, 32, 3),
            class_names=CIFAR100_FINE,
        ),
        DatasetSpec(
            id="cifar100_coarse",
            display_name="CIFAR-100 (coarse, 20 superclasses)",
            description="20개 상위 클래스(superclass). 클래스당 train 2,500장 / test 500장.",
            num_classes=20,
            label_mode="coarse",
            input_shape=(32, 32, 3),
            class_names=CIFAR100_COARSE,
        ),
    )
}


def get_dataset(dataset_id: str) -> DatasetSpec:
    try:
        return DATASETS[dataset_id]
    except KeyError:
        raise AppError("err.unknown_dataset", dataset=dataset_id) from None


def info_path(dataset_id: str):
    return config.DATASETS_DIR / dataset_id / "info.json"


def dataset_status(dataset_id: str) -> dict:
    spec = get_dataset(dataset_id)
    info = read_json(info_path(dataset_id))
    out = spec.to_dict()
    out["class_names"] = list(spec.class_names)
    out["prepared"] = bool(info)
    if info:
        out.update({k: v for k, v in info.items() if k not in out})
    return out


def split_train_val(x, y, val_split: float, seed: int, subset: float = 1.0):
    """원본 train에서 validation을 클래스 비율대로(stratified) 떼어 낸다. test set은 건드리지 않는다."""
    from sklearn.model_selection import train_test_split

    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=val_split, random_state=seed, stratify=y
    )
    if subset < 1.0:
        x_tr, _, y_tr, _ = train_test_split(
            x_tr, y_tr, train_size=subset, random_state=seed, stratify=y_tr
        )
    return x_tr, y_tr, x_val, y_val
