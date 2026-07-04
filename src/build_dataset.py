"""
Junta as amostras já convertidas de todas as fontes, remove duplicatas, separa
train/val(/test) e escreve o data.yaml final consumido pelo treino do YOLO.
"""

import os
import random
import shutil
from pathlib import Path

import yaml

from .dedup import dedup_samples
from .sources.base import YoloSample
from .sources.coco_baseline import COCOBaselineSource
from .sources.rdd2022_source import RDD2022Source
from .sources.roboflow_source import RoboflowSource
from .sources.tt100k_source import TT100KSource

random.seed(42)

ADAPTER_CLASSES = {
    "tt100k": TT100KSource,
    "rdd2022": RDD2022Source,
    "roboflow": RoboflowSource,
}

# chaves do sources.yaml que NÃO são definições de classe (não têm "sources: [...]")
RESERVED_KEYS = {"baseline_coco", "dataset"}


def _link_or_copy_image(src: Path, dst: Path) -> None:
    """
    Usa link simbólico em vez de copiar bytes. Cai pra cópia
    física só se o symlink falhar por algum motivo (ex: sistema de arquivos que
    não suporta, como certas montagens de rede).
    """
    try:
        os.symlink(os.path.realpath(src), dst)
    except OSError:
        shutil.copy(src, dst)


def load_config(config_dir: Path) -> tuple[dict, dict]:
    with open(config_dir / "classes.yaml") as f:
        classes_cfg = yaml.safe_load(f)
    with open(config_dir / "sources.yaml") as f:
        sources_cfg = yaml.safe_load(f)
    return classes_cfg, sources_cfg


def run_all_sources(work_dir: Path, classes_cfg: dict, sources_cfg: dict) -> list[YoloSample]:
    all_samples: list[YoloSample] = []

    # baseline COCO (não é uma "classe nova", tratado à parte)
    baseline_cfg = dict(sources_cfg["baseline_coco"])
    baseline_cfg["coco_classes"] = list(classes_cfg["coco_classes"].values())
    all_samples += COCOBaselineSource(work_dir, baseline_cfg).run()

    # Cada classe nova pode ter 1+ fontes (config/sources.yaml -> <classe>.sources).
    # Todas alimentam o mesmo target_class_id; duplicatas entre fontes da mesma
    # classe são removidas depois, no dedup global.
    for class_key, class_def in sources_cfg.items():
        if class_key in RESERVED_KEYS:
            continue

        target_class_id = class_def["target_class_id"]
        source_list = class_def.get("sources", [])

        for i, src_cfg in enumerate(source_list):
            adapter_name = src_cfg.get("adapter")
            adapter_cls = ADAPTER_CLASSES.get(adapter_name)
            if adapter_cls is None:
                print(f"AVISO: adapter '{adapter_name}' desconhecido para '{class_key}', pulando")
                continue

            cfg = dict(src_cfg)
            cfg["target_class_id"] = target_class_id
            instance_key = f"{class_key}_{i}"

            try:
                all_samples += adapter_cls(work_dir, cfg, instance_key=instance_key).run()
            except RuntimeError as e:
                print(f"AVISO: pulei fonte {i} de '{class_key}' — {e}")

    return all_samples


def split_and_write(
    samples: list[YoloSample],
    dataset_root: Path,
    val_split: float,
    test_split: float = 0.0,
) -> list[str]:
    """
    Separa em train/val (e opcionalmente test, se test_split > 0) e copia os
    arquivos pra estrutura final. Retorna a lista de splits efetivamente criados,
    pra write_data_yaml saber quais caminhos declarar.
    """
    splits = ["train", "val"] + (["test"] if test_split > 0 else [])
    for split in splits:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    shuffled = samples[:]
    random.shuffle(shuffled)

    n_val = int(len(shuffled) * val_split)
    n_test = int(len(shuffled) * test_split)

    val_ids = set(id(s) for s in shuffled[:n_val])
    test_ids = set(id(s) for s in shuffled[n_val:n_val + n_test])

    for i, sample in enumerate(shuffled):
        if id(sample) in val_ids:
            split = "val"
        elif id(sample) in test_ids:
            split = "test"
        else:
            split = "train"

        # prefixo com a fonte + índice evita colisão de nomes entre datasets
        stem = f"{sample.source_name}_{i:06d}"
        out_img = dataset_root / "images" / split / (stem + sample.image_path.suffix)
        out_lbl = dataset_root / "labels" / split / (stem + ".txt")
        _link_or_copy_image(sample.image_path, out_img)
        shutil.copy(sample.label_path, out_lbl)

    print(
        f"Split final: train={len(shuffled) - n_val - n_test}, "
        f"val={n_val}" + (f", test={n_test}" if test_split > 0 else "")
    )
    return splits


def write_data_yaml(dataset_root: Path, classes_cfg: dict, splits: list[str]) -> Path:
    names = {**classes_cfg["coco_classes"], **classes_cfg["new_classes"]}
    content = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): v for k, v in names.items()},
    }
    if "test" in splits:
        content["test"] = "images/test"

    out_path = dataset_root / "data.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)
    return out_path


def build(work_dir: Path, dataset_root: Path, config_dir: Path) -> Path:
    classes_cfg, sources_cfg = load_config(config_dir)

    samples = run_all_sources(work_dir, classes_cfg, sources_cfg)
    print(f"Total antes do dedup: {len(samples)}")

    hash_threshold = sources_cfg["dataset"]["dedup_hash_threshold"]
    samples = dedup_samples(samples, hash_threshold)

    val_split = sources_cfg["dataset"]["val_split"]
    test_split = sources_cfg["dataset"].get("test_split", 0.0)
    splits = split_and_write(samples, dataset_root, val_split, test_split)

    data_yaml_path = write_data_yaml(dataset_root, classes_cfg, splits)
    print(f"data.yaml gerado em {data_yaml_path}")
    return data_yaml_path
