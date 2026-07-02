"""
Avaliação por classe. Gera uma tabela com mAP50, mAP50-95, precisão e revocação
para cada uma das 85 classes, com destaque separado pras 5 classes novas — é a
comparação mais direta pra mostrar que o fine-tuning não destruiu o desempenho
nas classes antigas do COCO.
"""

from pathlib import Path

import yaml
from ultralytics import YOLO


def evaluate(weights_path: Path, data_yaml: Path, output_csv: Path) -> None:
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml))

    with open(data_yaml) as f:
        names = yaml.safe_load(f)["names"]

    rows = ["class_id,class_name,precision,recall,mAP50,mAP50-95"]
    # metrics.box.p, .r, .maps são arrays por classe, na mesma ordem de names
    for class_id, class_name in names.items():
        idx = int(class_id)
        try:
            precision = metrics.box.p[idx]
            recall = metrics.box.r[idx]
            map50 = metrics.box.ap50[idx]
            map50_95 = metrics.box.maps[idx]
        except IndexError:
            precision = recall = map50 = map50_95 = float("nan")

        rows.append(f"{idx},{class_name},{precision:.4f},{recall:.4f},{map50:.4f},{map50_95:.4f}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("\n".join(rows))
    print(f"Relatório por classe salvo em {output_csv}")

    # resumo rápido no console, separando classes antigas vs novas
    new_class_ids = {80, 81, 82, 83, 84}
    print("\n--- Resumo ---")
    print(f"mAP50-95 geral: {metrics.box.map:.4f}")
    print(f"mAP50 geral:    {metrics.box.map50:.4f}")

    new_maps = [metrics.box.maps[i] for i in new_class_ids if i < len(metrics.box.maps)]
    if new_maps:
        print(f"mAP50-95 médio (5 classes novas): {sum(new_maps) / len(new_maps):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/per_class_metrics.csv"))
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.out)
