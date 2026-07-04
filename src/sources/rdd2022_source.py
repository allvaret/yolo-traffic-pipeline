"""
Adapter para o RDD2022 (Road Damage Detection 2022), usado aqui só pela classe
"pothole" (D40 e variações próximas, dependendo do país).

Download direto via S3 do sekilab, sem necessidade de login. Anotações originais
em PascalVOC XML; convertidas via src/convert/voc_to_yolo.py.

IMPORTANTE: baixamos só os países configurados em "countries" (sources.yaml), não o
RDD2022.zip inteiro (~seis países combinados). O mantenedor disponibiliza zips
separados por país, o que reduz bastante o download quando você não precisa de
todos. Tamanhos aproximados (train+test): Japan ~1.0GB, India ~502MB, Czech
~245MB, United_States ~424MB, China_MotorBike ~183MB, China_Drone ~153MB,
Norway ~9.9GB (bem mais pesado que os outros — pense duas vezes antes de incluir).
"""

import shutil
import zipfile
from pathlib import Path

import requests

from ..convert.voc_to_yolo import voc_xml_to_yolo_lines
from .base import DatasetSource, YoloSample

_BASE_URL = (
    "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/"
    "Country_Specific_Data_CRDDC2022"
)

# Se o link mudar, confira a seção "Links to download Country-specific data" em
# https://github.com/sekilab/RoadDamageDetector
COUNTRY_ZIP_URLS = {
    "Japan": f"{_BASE_URL}/RDD2022_Japan.zip",
    "India": f"{_BASE_URL}/RDD2022_India.zip",
    "Czech": f"{_BASE_URL}/RDD2022_Czech.zip",
    "Norway": f"{_BASE_URL}/RDD2022_Norway.zip",
    "United_States": f"{_BASE_URL}/RDD2022_United_States.zip",
    "China_MotorBike": f"{_BASE_URL}/RDD2022_China_MotorBike.zip",
    "China_Drone": f"{_BASE_URL}/RDD2022_China_Drone.zip",
}

_HEAVY_COUNTRIES = {"Norway"}  # ~9.9GB sozinho, avisa antes de baixar


class RDD2022Source(DatasetSource):
    name = "rdd2022"

    def download(self) -> Path:
        countries = self.config.get("countries") or list(COUNTRY_ZIP_URLS.keys())
        unknown = [c for c in countries if c not in COUNTRY_ZIP_URLS]
        if unknown:
            print(f"[{self.name}] AVISO: países desconhecidos ignorados: {unknown}")
            countries = [c for c in countries if c in COUNTRY_ZIP_URLS]

        extract_root = self.work_dir / "raw"
        extract_root.mkdir(parents=True, exist_ok=True)

        for country in countries:
            country_dir = extract_root / country
            if country_dir.exists() and any(country_dir.iterdir()):
                print(f"[{self.name}] {country}: cache local encontrado, pulando download")
                continue

            if country in _HEAVY_COUNTRIES:
                print(f"[{self.name}] AVISO: {country} é ~9.9GB sozinho, isso pode demorar bastante")

            url = COUNTRY_ZIP_URLS[country]
            zip_path = self.work_dir / f"RDD2022_{country}.zip"
            print(f"[{self.name}] baixando {country} ({url.rsplit('/', 1)[-1]})")

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Extrai num diretório temporário e depois normaliza a estrutura pra
            # <extract_root>/<country>/train/..., independente de como o zip do
            # país organiza as pastas internamente (com ou sem prefixo do país).
            tmp_extract = self.work_dir / f"_tmp_{country}"
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract)
            tmp_extract.mkdir(parents=True)

            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp_extract)
            zip_path.unlink()  # economiza espaço, já temos o extraído

            train_dir = next(tmp_extract.rglob("train"), None)
            if train_dir is None:
                print(f"[{self.name}] AVISO: não encontrei pasta 'train' dentro do zip de {country}, pulando")
                shutil.rmtree(tmp_extract)
                continue

            country_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(train_dir), str(country_dir / "train"))
            shutil.rmtree(tmp_extract)

        return extract_root

    def to_yolo(self, raw_dir: Path) -> list[YoloSample]:
        target_class_id = self.config["target_class_id"]
        filter_classes = self.config.get("filter_original_classes", ["D40"])

        class_filter_map = {cls: target_class_id for cls in filter_classes}

        samples = []
        # Estrutura: raw_dir / <País> / train / {images,annotations/xmls}
        # (o filtro por país já aconteceu em download() — só baixamos o que
        # estava em "countries" — então aqui processamos tudo que existe em disco)
        for country_dir in sorted(raw_dir.glob("*")):
            if not country_dir.is_dir():
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

                out_lbl = out_dir / (xml_path.stem + ".txt")
                out_lbl.write_text("\n".join(lines))

                samples.append(YoloSample(img_path, out_lbl, self.name))

        # NÃO corta por max_images aqui — feito centralmente e de forma aleatória
        # em DatasetSource.run() (ver src/sources/base.py:limit_samples).
        return samples
