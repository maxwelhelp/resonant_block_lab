# Resonant Block Lab

Мини-лаборатория PyTorch для **безопасного ResonantBlock**, который можно вставлять в сеть параллельно attention/FFN блоку.

Главная схема:

```text
x ───────────────► обычный блок ─────────────► base_out
                       │
                       ▼
                ResonantBlock(base_out)
                       │
                       ▼
y = base_out + gate · resonant_delta
```

`gate` инициализирован почти в ноль. Поэтому сеть сначала работает почти как старая, а резонатор подключается постепенно.

## Зачем

Attention делает попарное смешивание токенов `O(L^2)`. `ResonantBlock` делает итеративное волновое уточнение через смесь replayable операторов:

- identity
- shift left/right
- local average
- global mean
- high-pass
- learned depthwise modes

Это не attention и не FFN, а третий тип sequence-mixer примитива.

## Быстрый запуск

```bash
cd resonant_block_lab
bash scripts/smoke.sh
```

## Как вставить в Transformer

```python
import torch.nn as nn
from resonant_block_lab import ResonantBlockConfig, ParallelResonantBlock

base_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
cfg = ResonantBlockConfig(dim=128, n_modes=8, micro_steps=3, gate_init=-5.0)
layer = ParallelResonantBlock(cfg, base_block=base_layer)

y = layer(x)  # x/y: [B, L, D]
```

## Replayable program

На каждом микрошаге controller выбирает программу:

```text
program_m = q0·identity + q1·shift_left + q2·shift_right + q3·local_avg + ...
```

Это реально исполняемая программа, потому что ровно эта смесь операторов использовалась в forward.

## Почему не MPS/MPO сразу

Для коротких и средних sequence динамические локальные операторы обычно быстрее на GPU. MPO/MPS имеет смысл для `L >= 128/256`, большого числа modes или компактных replayable matrix programs.
