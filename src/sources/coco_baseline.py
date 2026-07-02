"""
Adapter que baixa uma amostra pequena das 80 classes originais do COCO, usada
exclusivamente para mitigar catastrophic forgetting durante o fine-tuning (o modelo
continua vendo exemplos das classes antigas enquanto aprende as novas).

Não representa uma classe "nova"; por isso não tem target_class_id fixo — os ids
já vêm corretos do próprio COCO (0-79), então aqui só copiamos direto.
"""

import shutil
from pathlib import Path

from .base import DatasetSource, YoloSample


class COCOBaselineSource(DatasetSource):
    name = "coco_baseline"

    def download(self) -> Path:
        import fiftyone as fo
        import fiftyone.zoo as foz

        coco_classes = self.config["coco_classes"]  # lista de 80 nomes, na ordem oficial
        n_per_class = self.config.get("images_per_class", 100)

        export_dir = self.work_dir / "export"
        if export_dir.exists() and any(export_dir.iterdir()):
            print(f"[{self.name}] cache local encontrado, pulando download")
            return export_dir

        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            max_samples=n_per_class * len(coco_classes),
            classes=coco_classes,
            label_types=["detections"],
        )
        dataset.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            classes=coco_classes,
        )
        return export_dir

    def to_yolo(self, raw_dir: Path) -> list[YoloSample]:
        img_dir = raw_dir / "images" / "train"
        lbl_dir = raw_dir / "labels" / "train"

        samples = []
        for img_path in sorted(img_dir.glob("*")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                samples.append(YoloSample(img_path, lbl_path, self.name))
        return samples
