"""
Remoção de duplicatas entre fontes diferentes.

Datasets públicos de trânsito frequentemente reaproveitam as mesmas imagens-base
(ex: crops do mesmo frame de vídeo aparecendo em dois datasets do Roboflow Universe
com anotações diferentes). Usamos perceptual hashing (pHash) em vez de hash exato,
porque a mesma imagem pode ter sido recomprimida/reescalada entre fontes.
"""

from pathlib import Path

import imagehash
from PIL import Image

from .sources.base import YoloSample


def dedup_samples(samples: list[YoloSample], hash_threshold: int = 5) -> list[YoloSample]:
    """
    Remove amostras cuja imagem é perceptualmente muito parecida com uma já vista.
    Mantém a primeira ocorrência (ordem de entrada = ordem de prioridade das fontes).

    hash_threshold: distância de Hamming máxima entre hashes pra considerar duplicata.
    Valores típicos: 0 = idêntico, 5 = bem parecido, 10+ = permissivo demais.
    """
    seen_hashes: list[imagehash.ImageHash] = []
    kept: list[YoloSample] = []
    dropped = 0

    for sample in samples:
        try:
            with Image.open(sample.image_path) as img:
                h = imagehash.phash(img)
        except Exception as e:
            print(f"AVISO: não consegui abrir {sample.image_path} ({e}), pulando")
            continue

        is_duplicate = any((h - seen_h) <= hash_threshold for seen_h in seen_hashes)
        if is_duplicate:
            dropped += 1
            continue

        seen_hashes.append(h)
        kept.append(sample)

    print(f"Dedup: {dropped} duplicatas removidas de {len(samples)} amostras totais")
    return kept


# NOTA (deferido para v1.1): essa implementação é O(n²) — cada imagem nova é
# comparada com todas as já vistas. Para o volume esperado aqui (na casa de
# 10-15 mil imagens) é aceitável, mas não escala bem além disso. Se o dataset
# crescer muito, trocar por um índice aproximado (ex: bucket por hash truncado,
# ou uma lib como `datasketch`) evita a explosão combinatória.
