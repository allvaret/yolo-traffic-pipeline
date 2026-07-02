"""
Adapter para o TT100K (Tsinghua-Tencent 100K), dataset de placas de trânsito chinesas.

Reaproveita a lógica de download+conversão que a própria Ultralytics já mantém
(TT100K.yaml), que baixa direto de cg.cs.tsinghua.edu.cn sem necessidade de login.
Aqui apenas orquestramos essa chamada e depois remapeamos as 221 subclasses originais
para o nosso id global unificado "traffic_sign".

Licença do dataset original: CC BY-NC 2.0 (uso não-comercial).
"""

import shutil
from pathlib import Path

from .base import DatasetSource, YoloSample


class TT100KSource(DatasetSource):
    name = "tt100k"

    def download(self) -> Path:
        """
        Usa o utilitário de download da própria Ultralytics, que lê o YAML oficial
        do TT100K e baixa+extrai o dataset (~18GB) automaticamente na primeira vez.
        Em execuções seguintes, reaproveita o cache local.
        """
        from ultralytics.data.utils import check_det_dataset

        data_cfg = check_det_dataset("TT100K.yaml")
        return Path(data_cfg["path"])

    def to_yolo(self, raw_dir: Path) -> list[YoloSample]:
        target_class_id = self.config["target_class_id"]

        samples = []
        for split in ["train", "val", "test"]:
            img_dir = raw_dir / "images" / split
            lbl_dir = raw_dir / "labels" / split
            if not img_dir.exists():
                continue

            out_split_dir = self.work_dir / split
            out_split_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(img_dir.glob("*.jpg")):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue

                # Remapeia TODAS as classes originais do TT100K para o id unificado
                # de "traffic_sign" (achatamento intencional; ver sources.yaml para
                # como manter granularidade se preferir)
                remapped_lines = []
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        _, x, y, w, h = parts
                        remapped_lines.append(f"{target_class_id} {x} {y} {w} {h}")

                if not remapped_lines:
                    continue

                out_img = out_split_dir / img_path.name
                out_lbl = out_split_dir / (img_path.stem + ".txt")
                if not out_img.exists():
                    shutil.copy(img_path, out_img)
                out_lbl.write_text("\n".join(remapped_lines))

                samples.append(YoloSample(out_img, out_lbl, self.name))

        # NÃO corta por max_images aqui — isso é feito de forma centralizada e
        # aleatória em DatasetSource.run() (ver src/sources/base.py:limit_samples),
        # pra não enviesar a amostra pegando só as primeiras imagens do diretório.
        return samples
