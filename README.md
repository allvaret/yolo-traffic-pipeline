# yolo-traffic-pipeline

Pipeline reprodutível de transfer learning para YOLOv8, estendendo o modelo pré-treinado
no COCO (80 classes) com 5 classes novas relacionadas a trânsito:

| id | classe            | fonte                        |
|----|--------------------|-------------------------------|
| 80 | traffic_sign       | TT100K                        |
| 81 | pedestrian_light   | Roboflow Universe             |
| 82 | crosswalk          | Roboflow Universe             |
| 83 | traffic_cone       | Roboflow Universe             |
| 84 | pothole            | RDD2022                       |

`pedestrian_light` como classe nova porque o COCO
não distingue semáforo de pedestre do semáforo de veículo — essa distinção é o valor
agregado do projeto.

## Por que essa arquitetura

Datasets de visão computacional raramente vêm prontos: cada um usa um formato de anotação
diferente (COCO JSON, PascalVOC XML, YOLO txt), IDs de classe próprios, e alguns exigem
cadastro manual pra download (Mapillary, BDD100K). Em vez de escrever um script único e
descartável, esse projeto usa um padrão de **adapter**: cada fonte de dados implementa a
mesma interface (`download()` + `to_yolo()`), e um orquestrador central (`pipeline.py`)
roda todas elas, unifica os IDs de classe via `config/classes.yaml`, remove duplicatas e
gera o dataset final.

Isso significa que adicionar uma 6ª classe no futuro, de uma nova fonte, é escrever um
adapter de ~30 linhas — não reescrever a pipeline inteira.

**Uma classe pode ter mais de uma fonte.** Em `config/sources.yaml`, cada classe define
uma lista `sources: [...]` em vez de uma fonte única — dá pra combinar, por exemplo,
RDD2022 + um dataset do Roboflow pra `pothole`, ganhando mais diversidade de ângulos e
países. Todas as fontes de uma classe são remapeadas pro mesmo `target_class_id`, e o
dedup (perceptual hash) roda no conjunto combinado, então imagens repetidas entre fontes
diferentes da mesma classe são removidas automaticamente. Veja o exemplo comentado em
`pothole` no próprio `sources.yaml`.

## Decisões de fonte de dados (e por quê)

- **TT100K**: a Ultralytics mantém um `.yaml` oficial com download e conversão automáticos.
  Zero fricção, licença CC BY-NC 2.0 (uso não-comercial — ok pra portfólio, mencione isso
  se for usar comercialmente). Alimenta `traffic_sign`.
- **RDD2022**: hospedado no GitHub/FigShare com link direto, sem login. Formato PascalVOC
  XML; o adapter filtra só as classes de buraco (`D40`/variações) entre as categorias de
  dano na via. Alimenta `pothole`.
- **Mapillary e BDD100K foram descartados** para as classes de nicho: ambos exigem
  aprovação manual de cadastro antes do download, o que quebraria o objetivo de pipeline
  100% reproduzível. Usamos Roboflow Universe no lugar.
- **Roboflow Universe**: cobre `pedestrian_light`, `crosswalk` e `traffic_cone` com
  datasets curados pela comunidade. Requer 1 cadastro gratuito e 1 API key (configuração
  única, depois é 100% automatizado via `roboflow` Python package).
- **Baseline COCO**: uma amostra pequena (~100 img/classe) das 80 classes originais é
  baixada via FiftyOne, para mitigar catastrophic forgetting durante o fine-tuning.

## Setup

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz (não versionar, já está no `.gitignore`):

```
ROBOFLOW_API_KEY=sua_chave_aqui
```

Preencha `config/sources.yaml` com os projetos do Roboflow Universe que você escolher
para `pedestrian_light`, `crosswalk` e `traffic_cone` (veja instruções dentro do arquivo).

## Uso

```bash
# Roda a etapa de dataset inteira: download -> conversão -> unificação -> dedup -> split -> data.yaml
python pipeline.py --steps dataset

# Treino
python pipeline.py --steps train

# Avaliação por classe
python pipeline.py --steps evaluate

# Ou tudo de uma vez
python pipeline.py --steps all
```

Cada etapa também pode ser chamada isoladamente via os módulos em `src/`, o que facilita
debugar uma fonte específica sem rodar a pipeline inteira.

## Estrutura

```
yolo-traffic-pipeline/
├── config/
│   ├── classes.yaml       # mapeamento unificado de classes (0-84)
│   └── sources.yaml       # qual adapter/fonte alimenta cada classe nova
├── src/
│   ├── sources/
│   │   ├── base.py            # interface abstrata DatasetSource
│   │   ├── coco_baseline.py   # amostra COCO via FiftyOne (anti-forgetting)
│   │   ├── tt100k_source.py
│   │   ├── rdd2022_source.py
│   │   └── roboflow_source.py
│   ├── convert/
│   │   └── voc_to_yolo.py     # conversor PascalVOC -> YOLO
│   ├── dedup.py                # remoção de duplicatas via perceptual hash
│   ├── build_dataset.py        # unifica, faz split train/val, gera data.yaml
│   ├── train.py
│   └── evaluate.py             # mAP/precision/recall por classe
├── pipeline.py                  # CLI orquestrador
├── notebooks/
│   └── colab_runner.ipynb       # roda o pipeline no Colab (GPU)
└── requirements.txt
```

## Licenciamento (importante mencionar no portfólio)

Cada fonte usada tem uma licença diferente:

- TT100K: CC BY-NC 2.0 (não-comercial)
- RDD2022: verificar termos atuais no repositório oficial (sekilab/RoadDamageDetector)
- Datasets do Roboflow Universe: CC BY 4.0