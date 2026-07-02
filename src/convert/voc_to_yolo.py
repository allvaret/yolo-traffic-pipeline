"""
Conversor genérico de anotações PascalVOC (.xml) para o formato YOLO (.txt).

Usado pelo adapter do RDD2022, mas escrito de forma desacoplada porque PascalVOC é um
formato comum o suficiente pra reaparecer em futuras fontes de dados.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


def voc_xml_to_yolo_lines(xml_path: Path, class_filter_map: dict[str, int]) -> list[str]:
    """
    Lê um XML no formato PascalVOC e retorna linhas no formato YOLO
    ("<class_id> <x_center> <y_center> <w> <h>", tudo normalizado 0-1).

    class_filter_map: dict mapeando o nome da classe original (ex: "D40") para o id
    global unificado (ex: 85). Classes fora desse dict são ignoradas — é assim que
    filtramos, por exemplo, só "pothole" entre as 8 categorias de dano do RDD2022.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in class_filter_map:
            continue

        class_id = class_filter_map[name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Bounding boxes malformadas (fora da imagem, ou com w/h <= 0) acontecem em
        # datasets reais; melhor descartar a linha do que treinar com lixo.
        if w <= 0 or h <= 0 or not (0 <= x_center <= 1) or not (0 <= y_center <= 1):
            continue

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return lines
