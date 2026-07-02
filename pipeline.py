"""
Ponto de entrada único da pipeline.

Uso:
    python pipeline.py --steps download,convert,unify,dedup,split
    python pipeline.py --steps train
    python pipeline.py --steps evaluate
    python pipeline.py --steps all

"download,convert,unify,dedup,split" são tratados como um bloco único aqui porque,
na prática, cada adapter já faz download+convert+unify internamente (ver run())
e o dedup/split só fazem sentido depois que TODAS as fontes rodaram. Rodar essas
etapas separadamente exigiria persistir estado intermediário em disco entre
chamadas de CLI distintas — deixado como possível melhoria de v1.1 caso o dataset
cresça a ponto de valer a pena poder re-rodar só uma etapa.
"""

import argparse
from pathlib import Path

from src.build_dataset import build
from src.evaluate import evaluate
from src.train import train

ROOT = Path(__file__).parent
WORK_DIR = ROOT / "work"          # dados brutos e intermediários por fonte
DATASET_ROOT = ROOT / "dataset"   # dataset final unificado (train/val)
CONFIG_DIR = ROOT / "config"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="dataset,train,evaluate ou all (separados por vírgula)",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--weights", type=Path, default=None, help="para --steps evaluate")
    args = parser.parse_args()

    steps = set(args.steps.split(","))
    if "all" in steps:
        steps = {"dataset", "train", "evaluate"}

    data_yaml_path = DATASET_ROOT / "data.yaml"

    if "dataset" in steps or {"download", "convert", "unify", "dedup", "split"} & steps:
        data_yaml_path = build(WORK_DIR, DATASET_ROOT, CONFIG_DIR)

    if "train" in steps:
        model, results = train(
            data_yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            freeze=args.freeze,
        )

    if "evaluate" in steps:
        weights = args.weights or (ROOT / "runs/detect/traffic_train/weights/best.pt")
        evaluate(weights, data_yaml_path, ROOT / "reports/per_class_metrics.csv")


if __name__ == "__main__":
    main()
