"""
Adapter para o RDD2022 (Road Damage Detection 2022), usado aqui só pela classe
"pothole" (D40 e variações próximas, dependendo do país).

Download direto via GitHub/FigShare, sem necessidade de login. Anotações originais
em PascalVOC XML; convertidas via src/convert/voc_to_yolo.py.
"""

import shutil
import zipfile
from pathlib import Path

import requests

from ..convert.voc_to_yolo import voc_xml_to_yolo_lines
from .base import DatasetSource, YoloSample

# Link espelhado no repositório oficial sekilab/RoadDamageDetector. Se o link mudar,
# confira a seção "releases" do repositório e atualize aqui.
RDD2022_URL = "https://github.com/sekilab/RoadDamageDetector/releases/download/v1.0/RDD2022.zip"


class RDD2022Source(DatasetSource):
    name = "rdd2022"

    def download(self) -> Path:
        zip_path = self.work_dir / "RDD2022.zip"
        extract_dir = self.work_dir / "raw"

        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"[{self.name}] cache local encontrado, pulando download")
            return extract_dir

        print(f"[{self.name}] baixando RDD2022.zip (isso pode demorar, dataset é grande)")
        with requests.get(RDD2022_URL, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        zip_path.unlink()  # economiza espaço em disco, já temos o extraído
        return extract_dir

    def to_yolo(self, raw_dir: Path) -> list[YoloSample]:
        target_class_id = self.config["target_class_id"]
        filter_classes = self.config.get("filter_original_classes", ["D40"])
        countries = self.config.get("countries")

        class_filter_map = {cls: target_class_id for cls in filter_classes}

        samples = []
        # Estrutura esperada: raw_dir / <País> / train / {images,annotations/xmls}
        for country_dir in sorted(raw_dir.glob("*")):
            if not country_dir.is_dir():
                continue
            if countries and country_dir.name not in countries:
                continue

            img_dir = country_dir / "train" / "images"
            xml_dir = country_dir / "train" / "annotations" / "xmls"
            if not img_dir.exists() or not xml_dir.exists():
                continue

            out_dir = self.work_dir / "yolo" / country_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for xml_path in sorted(xml_dir.glob("*.xml")):
                lines = voc_xml_to_yolo_lines(xml_path, class_filter_map)
                if not lines:
                    continue  # imagem sem instância da classe que nos interessa

                img_path = img_dir / (xml_path.stem + ".jpg")
                if not img_path.exists():
                    continue

                out_img = out_dir / img_path.name
                out_lbl = out_dir / (xml_path.stem + ".txt")
                if not out_img.exists():
                    shutil.copy(img_path, out_img)
                out_lbl.write_text("\n".join(lines))

                samples.append(YoloSample(out_img, out_lbl, self.name))

        # NÃO corta por max_images aqui — feito centralmente e de forma aleatória
        # em DatasetSource.run() (ver src/sources/base.py:limit_samples).
        return samples
