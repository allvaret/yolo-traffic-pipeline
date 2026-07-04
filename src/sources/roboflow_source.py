"""
Adapter genérico para qualquer dataset do Roboflow Universe.

Diferente do TT100K e RDD2022 (fontes fixas), esse adapter é parametrizado:
o mesmo código serve pra qualquer classe (e pode inclusive ser reaproveitado mais
de uma vez para a MESMA classe, se ela tiver mais de uma fonte no Roboflow — ver
config/sources.yaml, campo "sources" como lista). O isolamento entre instâncias
diferentes é feito via instance_key (herdado de DatasetSource).

Requer variável de ambiente ROBOFLOW_API_KEY (cadastro gratuito único em
https://roboflow.com). A partir daí, o download é 100% programático.
"""

import os
from pathlib import Path

from .base import DatasetSource, YoloSample


def _normalize_class_name(name: str) -> str:
    """
    Remove espaço, hífen e underscore, e deixa minúsculo. Assim "traffic cone",
    "trafficcone", "Traffic-Cone" e "traffic_cone" são todos tratados como o
    mesmo nome — datasets diferentes do Universe raramente concordam na
    formatação exata do nome de uma classe.
    """
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


class RoboflowSource(DatasetSource):
    name = "roboflow"

    def download(self) -> Path:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY não definida. Crie uma conta gratuita em "
                "roboflow.com, gere uma API key, e exporte ROBOFLOW_API_KEY antes "
                "de rodar a pipeline."
            )

        workspace = self.config.get("workspace")
        project = self.config.get("project")
        version = self.config.get("version", 1)

        if workspace in (None, "PREENCHER") or project in (None, "PREENCHER"):
            raise RuntimeError(
                f"config/sources.yaml: preencha workspace/project para '{self.instance_key}' "
                f"com um dataset real do Roboflow Universe antes de rodar essa etapa."
            )

        from roboflow import Roboflow

        try:
            rf = Roboflow(api_key=api_key)
            rf_project = rf.workspace(workspace).project(project)
            dataset = rf_project.version(version).download(
                "yolov8", location=str(self.work_dir / "raw")
            )
        except RuntimeError:
            raise
        except Exception as e:
            # Cobre casos como projeto/versão inexistente, acesso negado (alguns
            # datasets do Universe exigem plano pago pra exportar em certos
            # formatos), download corrompido (zipfile.BadZipFile), ou instabilidade
            # de rede. Sem isso, a exceção original (às vezes uma BadZipFile ou
            # erro de rede) não é um RuntimeError, então NÃO seria pega pelo
            # try/except do orquestrador (run_all_sources em build_dataset.py) —
            # e derrubaria a pipeline inteira por causa de uma única fonte com
            # problema, depois de horas de processamento de outras fontes.
            raise RuntimeError(
                f"Falha ao baixar '{workspace}/{project}' versão {version} do "
                f"Roboflow ({type(e).__name__}: {e}). Confira se o projeto e a "
                f"versão existem e se você tem acesso, abrindo "
                f"https://universe.roboflow.com/{workspace}/{project} no navegador."
            ) from e

        return Path(dataset.location)

    def to_yolo(self, raw_dir: Path) -> list[YoloSample]:
        target_class_id = self.config["target_class_id"]
        source_class_names = self.config.get("source_class_names", [])
        normalized_targets = {_normalize_class_name(s) for s in source_class_names}

        # O Roboflow já exporta no formato YOLO, mas com o próprio esquema de ids
        # de classe do projeto original. Precisamos ler o data.yaml exportado pra
        # saber qual id local corresponde a qual nome de classe, e então remapear
        # só as classes que nos interessam para o nosso id global.
        import yaml

        data_yaml_path = raw_dir / "data.yaml"
        if not data_yaml_path.exists():
            raise FileNotFoundError(
                f"data.yaml não encontrado em {raw_dir}; download do Roboflow pode "
                f"ter falhado silenciosamente."
            )

        with open(data_yaml_path) as f:
            rf_config = yaml.safe_load(f)
        rf_names = rf_config["names"]  # lista de nomes, índice = id local do projeto

        # Comparação normalizada (ignora espaço/hífen/underscore/maiúsculas), pra
        # não depender de digitar EXATAMENTE "traffic cone" quando o projeto usa
        # "trafficcone" ou "Traffic-Cone". Reduz a chance de dar 0 amostras só
        # por causa de uma diferença cosmética no nome da classe.
        local_id_to_global = {
            i: target_class_id
            for i, n in enumerate(rf_names)
            if _normalize_class_name(n) in normalized_targets
        }

        if not local_id_to_global:
            print(
                f"[{self.name}:{self.instance_key}] AVISO: nenhuma classe do projeto "
                f"bateu com source_class_names={source_class_names} (mesmo com "
                f"normalização de espaço/hífen/underscore). Classes disponíveis no "
                f"projeto: {rf_names}. Ajuste config/sources.yaml."
            )

        samples = []
        for split in ["train", "valid", "test"]:
            img_dir = raw_dir / split / "images"
            lbl_dir = raw_dir / split / "labels"
            if not img_dir.exists():
                continue

            out_dir = self.work_dir / "yolo" / split
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(img_dir.glob("*")):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue

                remapped = []
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        local_id, x, y, w, h = parts
                        local_id = int(local_id)
                        if local_id not in local_id_to_global:
                            continue  # descarta classes que não nos interessam
                        remapped.append(f"{local_id_to_global[local_id]} {x} {y} {w} {h}")

                if not remapped:
                    continue

                out_lbl = out_dir / (img_path.stem + ".txt")
                out_lbl.write_text("\n".join(remapped))

                samples.append(YoloSample(img_path, out_lbl, f"{self.name}_{self.instance_key}"))

        return samples
