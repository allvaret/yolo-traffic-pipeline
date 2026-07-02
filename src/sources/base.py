"""
Interface comum que toda fonte de dataset deve implementar.

O contrato é propositalmente pequeno: download() traz os dados brutos pro disco,
to_yolo() garante que eles terminem como pares (imagem, label .txt) já com o id de
classe global (unificado) escrito no label. Isso permite ao orquestrador
(build_dataset.py) tratar todas as fontes de forma idêntica, independente de quão
diferente é o formato original de cada uma.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class YoloSample:
    """Um par imagem+label já convertido, pronto pra ser copiado pro dataset final."""
    image_path: Path
    label_path: Path
    source_name: str  # usado para prefixar nomes de arquivo e evitar colisão


def limit_samples(
    samples: list[YoloSample], max_images: int | None, seed: int = 42
) -> list[YoloSample]:
    """
    Corta a lista para no máximo max_images, escolhendo aleatoriamente (não pegando
    as primeiras N). Isso importa porque muitos datasets têm imagens sequenciais
    (frames de vídeo próximos, capturas da mesma rua em sequência); pegar só as
    primeiras N enviesa a amostra. max_images=None mantém tudo.
    """
    if max_images is None or len(samples) <= max_images:
        return samples

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    return shuffled[:max_images]


class DatasetSource(ABC):
    """Toda fonte (TT100K, RDD2022, Roboflow, baseline COCO) implementa isso."""

    name: str = "base"

    def __init__(self, work_dir: Path, config: dict, instance_key: str | None = None):
        # instance_key isola o work_dir quando a MESMA classe usa múltiplas fontes
        # (ex: dois datasets Roboflow diferentes alimentando "crosswalk"). Sem isso,
        # a segunda fonte sobrescreveria o cache/download da primeira.
        self.instance_key = instance_key or "default"
        self.work_dir = Path(work_dir) / self.name / self.instance_key
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    @abstractmethod
    def download(self) -> Path:
        """Baixa (ou reaproveita cache local) os dados brutos. Retorna o diretório raiz."""
        raise NotImplementedError

    @abstractmethod
    def to_yolo(self, raw_dir: Path) -> list[YoloSample]:
        """Converte os dados brutos para pares (imagem, label YOLO) com id de classe
        global já mapeado. NÃO copia para o dataset final; só retorna os caminhos.
        NÃO precisa aplicar max_images aqui — isso é feito centralmente em run()."""
        raise NotImplementedError

    def run(self) -> list[YoloSample]:
        raw_dir = self.download()
        samples = self.to_yolo(raw_dir)
        total_before = len(samples)

        max_images = self.config.get("max_images")
        samples = limit_samples(samples, max_images)

        tag = f"{self.name}:{self.instance_key}"
        if max_images is not None and total_before > max_images:
            print(f"[{tag}] {total_before} disponíveis, amostra aleatória limitada a {max_images}")
        else:
            print(f"[{tag}] {len(samples)} amostras convertidas para YOLO")

        return samples
