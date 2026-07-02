"""
Treino com backbone congelado (mesma estratégia de transfer learning discutida
para o projeto de 82 classes: preserva as features genéricas do COCO, adapta
neck+head para as classes novas).
"""

from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: Path,
    model_base: str = "yolov8n.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 16,
    freeze: int = 10,
    lr0: float = 0.001,
    patience: int = 15,
    project: str = "runs/detect",
    name: str = "traffic_train",
):
    model = YOLO(model_base)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,
        lr0=lr0,
        patience=patience,
        project=project,
        name=name,
        exist_ok=True,
    )
    return model, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--freeze", type=int, default=10)
    args = parser.parse_args()

    train(args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, freeze=args.freeze)
