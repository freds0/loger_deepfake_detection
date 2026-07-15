# PLAN v0.1 — Melhorias de Arquitetura e Pipeline de Treinamento (foco: NTIRE 2026)

Plano de implementação derivado da auditoria do código em 2026-07-06 (branch `main`,
commit `d2154f1` + suporte NTIRE não commitado). Cada item especifica: motivação,
arquivos e símbolos exatos a alterar, chaves de config com defaults, e critério de
verificação. Ordem de execução: P0 → P1 → P2 → P3. Itens dentro de uma fase são
independentes entre si, salvo quando indicado.

Convenções deste documento:
- "NTIRE root" = diretório com `shard_N/labels.csv` + `shard_N/images/`.
- Todos os novos parâmetros de config são **opt-in** com default igual ao
  comportamento atual (nenhuma run existente muda de resultado sem override).
- Todo item novo em `ForgeryDataModule.__init__` deve também ser adicionado a
  `configs/data/ntire.yaml` E `configs/data/faceforensics.yaml` (o datamodule é
  instanciado com `ForgeryDataModule(**cfg.data)` em `train.py:77` /
  `train_loger.py`, então chave faltante em qualquer yaml = TypeError).

---

## P0 — Correção e robustez (obrigatório antes de runs longas no NTIRE)

### P0.1 Carregamento robusto de imagens (truncadas/corrompidas)

**Problema.** `ForgeryFrameDataset.__getitem__` (`src/data/dataset.py:194`) chama
`Image.open(rec.path).convert("RGB")` sem tratamento. Dataset "in the wild" com
centenas de milhares de JPEGs tem probabilidade real de conter arquivos truncados;
um único arquivo ruim aborta a run inteira (potencialmente horas depois do início).

**Implementação.**
1. Em `src/data/dataset.py`, no topo do módulo, adicionar:
   ```python
   from PIL import ImageFile
   ImageFile.LOAD_TRUNCATED_IMAGES = True
   ```
   (JPEG truncado passa a decodificar com o que houver, em vez de lançar `OSError`.)
2. Criar `scripts/validate_dataset.py` (novo arquivo, CLI standalone):
   - Argumentos: `--root <path>` (obrigatório), `--shards 0 1 2 ...`
     (opcional; default = todos os `shard_*` encontrados), `--workers N`
     (default 16, via `multiprocessing.Pool`).
   - Para cada linha de cada `labels.csv`: verificar (a) arquivo existe,
     (b) `Image.open(p); img.verify()` não lança, (c) label ∈ {0, 1}.
   - Saída: imprime contagem por shard (`ok / missing / corrupt / bad_label`) e
     grava CSV `<root>/validation_report.csv` com colunas
     `shard,image_name,status` apenas para os problemáticos. Exit code 0 se tudo
     ok, 1 se houver qualquer problema.
3. Sem lógica de "skip on error" dentro do Dataset — arquivo com falha de decode
   completa (não apenas truncado) deve continuar lançando exceção com o path na
   mensagem; a política é sanear o dataset com o script ANTES do treino, não
   mascarar silenciosamente durante.

**Verificação.** Teste unitário em `tests/test_dataset_ntire.py`: gerar um JPEG
truncado (escrever bytes de um JPEG válido cortado em 60%), confirmar que
`ForgeryFrameDataset` o carrega sem exceção; rodar `validate_dataset.py` num
diretório sintético com 1 arquivo faltante + 1 corrompido e conferir o relatório.

### P0.2 Split treino/val determinístico intra-shard (substituir split por shard)

**Problema.** O split atual (`train_shards=[0..4]`, `val_shards=[5]`,
`test_shards=[6]` em `configs/data/ntire.yaml`) assume que os shards são i.i.d.,
o que não é verificável — se os shards do NTIRE agruparem geradores diferentes, a
validação mede generalização a geradores não vistos (pode ser desejado ou não), e
perde-se ~15% dos dados de treino à toa. Não há test set público separado de
verdade: shard 6 é apenas mais um shard de treino.

**Implementação.** Adicionar modo de split por fração, estratificado e estável:
1. `src/data/dataset.py` — nova função:
   ```python
   def split_records_by_hash(
       records: list[Record],
       val_fraction: float,
       split: str,                      # "train" | "val"
   ) -> list[Record]:
   ```
   Regra determinística SEM RNG (estável entre runs, máquinas e mudanças de
   `seed`): uma amostra vai para val sse
   `int(hashlib.md5(Path(r.path).name.encode()).hexdigest(), 16) % 10_000 < int(val_fraction * 10_000)`.
   `split="train"` retorna o complemento. (Hash do basename, não do path
   completo, para o split não mudar se o root mudar de máquina.)
2. `ForgeryDataModule.__init__` (`src/data/datamodule.py`): novo parâmetro
   `val_fraction: float | None = None`.
3. `ForgeryDataModule._load_records`, branch `ntire`: quando
   `val_fraction` não é `None`:
   - `train` → `split_records_by_hash(records_from_ntire(root, train_shards), val_fraction, "train")`
   - `val` e `test` → complemento de val sobre os MESMOS `train_shards`
     (`val_shards`/`test_shards` são ignorados; documentar no docstring).
   Quando `val_fraction is None`: comportamento atual (split por shard) intacto.
4. `configs/data/ntire.yaml`: adicionar `val_fraction: 0.05`, mudar
   `train_shards: [0, 1, 2, 3, 4, 5, 6]` (todos), manter `val_shards`/`test_shards`
   como estão (ignorados neste modo, mas documentados em comentário).

**Verificação.** Teste unitário: 10k records sintéticos → frações observadas de
val em [4.3%, 5.7%]; duas chamadas retornam exatamente o mesmo conjunto;
train ∩ val = ∅; train ∪ val = tudo.

### P0.3 Balanceamento de classes por sampler (em vez de duplicação de records)

**Problema.** O único mecanismo atual é `real_oversample` (duplicação física da
lista de records, `src/data/dataset.py:oversample_real`), pensado para o FF++
(1 real : 4 fakes). O balanceamento do NTIRE é desconhecido a priori e pode variar
por shard; duplicar records infla o "tamanho da época" e interage mal com
`max_steps`.

**Implementação.**
1. `ForgeryDataModule.__init__`: novo parâmetro `balanced_sampling: bool = False`.
2. `ForgeryDataModule.train_dataloader`: quando `balanced_sampling=True`,
   construir `torch.utils.data.WeightedRandomSampler` com
   `weights[i] = 1.0 / count(label_i)` (contagens sobre os records de treino),
   `num_samples = len(dataset)`, `replacement=True`, e passar
   `sampler=...`/`shuffle=False` ao DataLoader. Guard: se
   `balanced_sampling=True` e `real_oversample > 1`, lançar
   `ValueError("balanced_sampling and real_oversample>1 are mutually exclusive")`.
3. Atenção DDP: `WeightedRandomSampler` não é distribuído. Se
   `trainer.world_size > 1` for usado com `balanced_sampling`, Lightning
   substituiria o sampler; para não abrir esse escopo agora, documentar no
   docstring que `balanced_sampling` é suportado apenas em single-GPU nesta
   versão (assert em `setup()` se `trainer.world_size > 1`).
4. `configs/data/ntire.yaml`: `balanced_sampling: true`, `real_oversample: 1`.

**Verificação.** Teste unitário com dataset sintético 90/10: média da fração de
positivos em 50 batches ∈ [0.45, 0.55].

### P0.4 Precisão bf16 e flags de performance para B200

**Problema.** `configs/trainer/default.yaml` usa `precision: 16-mixed` (fp16 com
loss scaling) e `deterministic: true` global (`configs/loger_fsm_ntire.yaml:18`).
Em Hopper/Blackwell, `bf16-mixed` elimina loss-scaling e instabilidades; kernels
determinísticos custam throughput sem benefício num treino longo (seed já é
fixada para inicialização/dados).

**Implementação.** Somente config, sem código:
1. `configs/loger_fsm_ntire.yaml`: adicionar
   ```yaml
   deterministic: false
   trainer:
     precision: bf16-mixed
   ```
2. `src/utils/lightning_setup.py` (`build_trainer`): confirmar que `deterministic`
   é lido de `cfg.deterministic` e repassado ao `Trainer`; adicionar
   `benchmark=not cfg.deterministic` na construção do Trainer (cudnn autotune).

**Verificação.** Dry-run de 20 steps; conferir no stdout do Lightning
`precision=bf16-mixed` e ausência de mensagens de GradScaler; comparar
`train/epoch_time_s` antes/depois num subset fixo (esperado ≥ 5% mais rápido;
registrar números no PR).

### P0.5 Correção do custo O(n²) em `_limit_records`

**Problema.** `src/data/datamodule.py:_limit_records` faz
`remaining = [r for r in records if r not in selected]` — `Record` é dataclass
(comparação por valor), custo O(n·m). Com `max_records_per_split` sobre um
dataset de centenas de milhares de itens, isso trava o `setup()`.

**Implementação.** Trocar o teste de pertinência por identidade com `set` de
`id()`:
```python
selected_ids = {id(r) for r in selected}
remaining = [r for r in records if id(r) not in selected_ids]
```
(Os objetos vêm da mesma lista `records`, então `id()` é correto aqui.)

**Verificação.** Teste de tempo no pytest: 200k records sintéticos com
`max_records_per_split=1000` → `_limit_records` < 1 s.

---

## P1 — Qualidade de treino (implementar após P0, antes da primeira run completa)

### P1.1 Warmup de LR

**Problema.** Full fine-tune de um VFM pré-treinado com AdamW `lr=3e-5` sem
warmup arrisca degradar as features pré-treinadas nos primeiros steps.
`LOGERLightningModule.configure_optimizers` (`src/lightning/loger_module.py:249`)
JÁ suporta warmup linear via `scheduler.warmup_steps`; só não está habilitado.

**Implementação.** Somente config: em `configs/loger_fsm_ntire.yaml` adicionar
```yaml
scheduler:
  warmup_steps: 500
```
(500 steps ≈ 1.6% de `max_steps=30000`; `SequentialLR` já encadeia com o cosine.)

**Verificação.** Dry-run; conferir no W&B/TB que a curva `lr` sobe linearmente de
~3e-8 até 3e-5 nos primeiros 500 steps e depois segue o cosine.

### P1.2 EMA (Exponential Moving Average) dos pesos

**Problema.** Não há EMA. Para desafios de robustez, avaliar com pesos EMA
tipicamente rende +0.2–1.0 AUC/acc e curvas de val mais estáveis, a custo quase
nulo.

**Implementação.** Novo arquivo `src/utils/ema.py` com um callback Lightning:
```python
class EMACallback(lightning.Callback):
    def __init__(self, decay: float = 0.999) -> None: ...
```
Semântica exata:
- `on_fit_start`: cria `self._shadow = {k: v.detach().clone() for k, v in pl_module.model.state_dict().items()}`
  (somente `pl_module.model`, não o LightningModule inteiro).
- `on_train_batch_end`: para cada tensor float do state_dict,
  `shadow.mul_(decay).add_(param, alpha=1-decay)`; tensores não-float
  (num_batches_tracked etc.) são copiados diretamente.
- `on_validation_start` / `on_test_start`: guarda o state_dict corrente em
  `self._backup` e carrega `self._shadow` em `pl_module.model`
  (`load_state_dict(..., strict=True)`).
- `on_validation_end` / `on_test_end`: restaura `self._backup` e o descarta.
- `state_dict()` / `load_state_dict()`: expõe `self._shadow` para o checkpoint
  Lightning (resume preserva o EMA).
- Consequência intencional: `val/auc` monitorado pelo ModelCheckpoint passa a ser
  o AUC do modelo EMA, e `best.ckpt` (pesos salvos após a validação) contém os
  pesos EMA? **Não** — o checkpoint é salvo fora da janela de swap, então salva
  os pesos brutos + `_shadow` no estado do callback. Para inferência com EMA,
  `predict.py`/`test_loger.py` devem aceitar flag `use_ema: bool = true` que, ao
  carregar o ckpt, sobrescreve `model.state_dict` pelo shadow do callback
  (chave `callbacks/EMACallback` no ckpt). Implementar helper
  `load_ema_weights(ckpt_path, model) -> bool` em `src/utils/ema.py` e chamá-lo
  nos entry points de test/predict quando a flag estiver ativa.

Config: `configs/callbacks/loger.yaml` ganha bloco
```yaml
ema:
  enabled: true
  decay: 0.999
```
e `src/utils/lightning_setup.py:build_callbacks` instancia o callback quando
`enabled`.

**Verificação.** Teste unitário: modelo linear de brinquedo, 10 steps com pesos
conhecidos → shadow bate com o cálculo manual do EMA; teste de swap: dentro de
`on_validation_start` o `model.state_dict()` é o shadow, após `on_validation_end`
volta ao original (comparação exata de tensores). Smoke test integrado no
`tests/test_loger_smoke.py` com o callback ativo.

### P1.3 Calibração de threshold no val + uso na inferência

**Problema.** Toda a inferência usa threshold fixo 0.5
(`predict_step`, `src/lightning/loger_module.py:207`; `compute_metrics`,
`src/training/metrics.py:53`). Se o desafio pontuar acurácia balanceada/acc, o
threshold ótimo raramente é 0.5, sobretudo com classes desbalanceadas.

**Implementação.**
1. `src/training/metrics.py` — nova função:
   ```python
   def best_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
       """Varre os thresholds da ROC e retorna (threshold, acc) que maximiza a acurácia."""
   ```
   Implementação: `fpr, tpr, thr = roc_curve(...)`;
   `acc = tpr * pos_rate + (1 - fpr) * (1 - pos_rate)` vetorizado; argmax.
2. `LOGERLightningModule._finalise_eval`: quando `prefix == "val"`, calcular e
   logar `val/best_threshold` e `val/best_acc`, e guardar
   `self.best_threshold = float(t)` (atributo simples; entra no ckpt via
   `on_save_checkpoint` → `checkpoint["best_threshold"] = ...`, restaurado em
   `on_load_checkpoint`; default 0.5 se ausente).
3. `predict_step`: usar `getattr(self, "best_threshold", 0.5)` no lugar do 0.5
   hardcoded.

**Verificação.** Teste unitário de `best_accuracy_threshold` com distribuição
sintética assimétrica (threshold ótimo conhecido analiticamente ± tolerância);
conferir que um ckpt salvo/recarregado preserva `best_threshold`.

### P1.4 Métricas por shard na validação

**Problema.** `_finalise_eval` agrega tudo num único AUC. Sem breakdown por shard
não dá para detectar que o modelo vai mal num subconjunto (ex.: um gerador
específico concentrado num shard).

**Implementação.**
1. `_eval_step` (`src/lightning/loger_module.py:143`): além de scores/labels,
   acumular `self._val_paths: list[list[str]]` (batch["path"]).
2. `_finalise_eval`: derivar `shard = Path(p).parent.parent.name` para cada path
   (layout `<root>/shard_N/images/<name>` ⇒ `parent.parent.name == "shard_N"`;
   para FF++ o resultado é o nome da pasta de manipulação — breakdown igualmente
   útil, nenhum caso especial necessário). Para cada grupo com ≥ 2 classes e
   ≥ 50 amostras, logar `{prefix}/auc/{shard}` e `{prefix}/acc/{shard}`.
   Sob DDP (world_size > 1): pular o breakdown (strings não passam por
   `all_gather` sem complicação; logar apenas rank-0 seria enviesado) — anotar
   comentário explicando.
3. Limpar `self._val_paths` junto com os outros buffers.

**Verificação.** Smoke test com dataset sintético de 2 shards, um deles com
scores propositalmente invertidos → `val/auc/shard_1` ≈ 0 e `val/auc/shard_0` ≈ 1.

### P1.5 Eliminar forward duplo na avaliação multi-resolução

**Problema.** `_eval_step` roda `self.model(...)` para a loss e depois
`self._eval_logits(...)` roda o modelo de novo (incluindo a resolução nativa)
quando `len(eval_resolutions) > 1` — o custo da resolução base é pago 2×.

**Implementação.** Refatorar `LOGERModel.forward_multi_resolution`
(`src/models/loger.py:224`) para aceitar
`precomputed: tuple[int, torch.Tensor] | None = None` — logits já computados para
uma das resoluções — e pular essa resolução no loop. `_eval_step` passa
`precomputed=(pixel_values.shape[-1], out.logits)`.

**Verificação.** Teste unitário: com `eval_resolutions=[224, 336]` e input 224,
`forward_multi_resolution(x, [224, 336], precomputed=(224, logits224))` retorna o
mesmo tensor (atol 1e-6) que a versão sem `precomputed`, com 1 forward a menos
(contar via hook no backbone).

---

## P2 — Arquitetura (cada item entra atrás de flag, default off, e só é adotado se
melhorar `val/auc` num A/B com mesma seed e mesmos dados)

### P2.1 Pesos de fusão aprendíveis (global vs. local)

**Motivação.** A fusão atual é média fixa 0.5/0.5
(`LOGERModel.fusion_weights`, buffer em `src/models/loger.py:180`). No NTIRE
(imagens inteiras geradas, não face swap local), o prior do branch local (10% dos
patches manipulados) é menos válido; deixar o modelo aprender a ponderação é
barato (2 parâmetros).

**Implementação.**
1. `LOGERModel.__init__`: novo kwarg `learnable_fusion: bool = False`. Quando
   `True`: `self.fusion_logits = nn.Parameter(torch.zeros(2))` e NÃO registrar o
   buffer; no `forward`, `w = torch.softmax(self.fusion_logits, dim=0)` (garante
   w > 0, soma 1 — logits fundidos permanecem na mesma escala). Quando `False`:
   comportamento atual intacto.
2. `build_loger_model` (`src/lightning/loger_module.py:30`): repassar
   `learnable_fusion=_select(cfg, "model.learnable_fusion", False)`.
3. `configs/model/loger.yaml`: `learnable_fusion: false`.
4. Logar os pesos: em `on_train_epoch_end`, se o parâmetro existir, logar
   `fusion/w_global` e `fusion/w_local`.

**Verificação.** Teste unitário: com `learnable_fusion=true`, `fusion_logits`
recebe gradiente após um backward; com `false`, `state_dict` é idêntico ao atual
(compatibilidade de checkpoint preservada).

### P2.2 Gradient checkpointing no backbone

**Motivação.** Permite batch maior ou backbone large em full fine-tune, trocando
~25-30% de tempo por ~40-60% menos memória de ativação.

**Implementação.**
1. `configs/backbone/*.yaml`: nova chave `gradient_checkpointing: false`.
2. `build_backbone` (`src/models/backbones.py:245`): novo kwarg
   `gradient_checkpointing: bool = False`, repassado a cada classe. Em cada
   backbone (`SiglipLOGERBackbone`, `Dinov2Backbone`, `Dinov3Backbone`,
   `CLIPBackbone`): se `True`, chamar
   `self.model.gradient_checkpointing_enable()` (API HF, disponível nos 4 —
   para o SigLIP, chamar no `vision_model`; confirmar atributo com
   `hasattr` e lançar erro claro caso a versão do transformers não suporte).
3. `LOGERModel.__init__` → `build_backbone(..., gradient_checkpointing=...)`;
   `build_loger_model` lê `_select(cfg, "backbone.gradient_checkpointing", False)`.
4. Incompatibilidade conhecida: checkpointing exige `use_reentrant=False` com
   tokens que requerem grad; HF já usa non-reentrant por default nas versões
   pinadas. Nada a fazer além de smoke test.

**Verificação.** Dry-run com `data.batch_size` 2× o máximo sem checkpointing →
não dá OOM; `gpu/mem_alloc_GB` logado cai ≥ 30%; loss dos 20 primeiros steps bate
(atol 1e-4, mesma seed) com a run sem checkpointing.

### P2.3 Multi-resolução na avaliação final e TTA leve na predição

**Motivação.** `forward_multi_resolution` já existe e está plugado
(`model.eval_resolutions`); hoje configurado como `[224]` (single). O paper LOGER
reporta ganho de robustez com média multi-escala.

**Implementação.** Somente config + um flag:
1. Treino/val: manter `[224]` (val multi-res 3× mais lenta a cada 1000 steps não
   compensa).
2. `test_loger.py` / `predict_loger.py`: aceitar override já suportado pelo Hydra
   `model.eval_resolutions=[224,336,448]` — documentar no README a linha exata.
3. TTA horizontal-flip na predição: em `predict_step`, novo flag de config
   `model.tta_hflip: bool = false`; quando `true`,
   `logits = 0.5 * (self._eval_logits(x) + self._eval_logits(torch.flip(x, dims=[-1])))`.
   Adicionar a chave em `configs/model/loger.yaml`.

**Verificação.** Rodar `test_loger.py` no val split com `[224]` vs
`[224,336,448]` vs `+tta_hflip` e registrar a tabela de AUC/acc no PR; adotar a
combinação vencedora para a submissão.

### P2.4 Escada de backbones (ablação controlada)

**Motivação.** `siglip2_base` é o default; o repo já tem configs para
`siglip2_large`, `dinov2_base`, `dinov3_base`, `clip_base`. Para NTIRE, DINOv2/v3
(self-supervised, sem viés de texto) costumam ser fortes em detecção de artefatos.

**Implementação.** Nenhum código novo. Protocolo fixo de ablação (registrar em
`docs/ablations_ntire.md`, criado neste item):
1. Subset fixo: `data.max_records_per_split={train: 100000, val: 20000}`,
   `trainer.max_steps=8000`, `seed=0`, augmentation on, EMA on.
2. Rodar: `backbone=siglip2_base` (referência), `backbone=dinov2_base`,
   `backbone=dinov3_base` (requer transformers>=4.56 — upgrade só no env da
   ablação), `backbone=siglip2_large peft=loger_lora` (large em LoRA para caber
   no tempo).
   Atenção: ao trocar de família, sobrescrever também
   `data.normalization_mean/std` com os valores de
   `normalization_stats(backbone.name)` (`src/models/backbones.py:237`) — os
   yamls de data têm stats SigLIP hardcoded. Adicionar nota no yaml de cada
   backbone com os stats corretos.
3. Critério de adoção: melhor `val/auc` EMA; empate (< 0.2) decide pelo menor
   custo de inferência.

**Verificação.** Tabela preenchida em `docs/ablations_ntire.md` com AUC/acc/EER,
steps/s e memória por variante.

### P2.5 Fora de escopo desta versão (registrado para não virar dívida oculta)

- **Vetorizar `ForgeryStyleMixture._domain_shuffle`** (loop Python O(F) por step,
  `src/models/fsm.py:84`): FSM está desabilitado no NTIRE (sem domínios); otimizar
  só se o FF++ voltar ao foco.
- **`torch.compile`**: interage mal com multi-resolução (shapes dinâmicos →
  recompilação) e com checkpointing HF; adiar até P2.2/P2.3 estarem decididos.
- **Dedup do padding do DistributedSampler** no `_finalise_eval` (viés < 1 batch
  por rank nas métricas): aceitável; revisitar se DDP virar o modo padrão.
- **Mixup/CutMix de imagem inteira**: interfere com a semântica do branch local
  (MIL assume patches manipulados); exigiria adaptação da MILLoss — não fazer.

---

## P3 — Infra e throughput (qualquer momento; não bloqueia treino)

### P3.1 Dataloader tuning para DGX

`ForgeryDataModule._loader`: expor `prefetch_factor: int = 2` como parâmetro do
datamodule (passar ao DataLoader apenas quando `num_workers > 0`).
`configs/data/ntire.yaml`: `num_workers: 16`, `prefetch_factor: 4`.
**Verificação.** Comparar steps/s com 8/16/32 workers num dry-run de 200 steps;
fixar o melhor.

### P3.2 Batch-size finder opcional

`train_loger.py`: nova chave de config raiz `tune_batch_size: false`. Quando
`true`, antes do `fit`:
```python
from lightning.pytorch.tuner import Tuner
Tuner(trainer).scale_batch_size(model, datamodule=datamodule, mode="binsearch")
```
e imprimir o valor encontrado (o usuário então fixa `data.batch_size`
explicitamente — o tuner não deve rodar em runs de produção).
**Verificação.** Rodar com `tune_batch_size=true` num subset e conferir que o
valor impresso treina 50 steps sem OOM.

### P3.3 Guia multi-GPU

Sem código: adicionar seção no README com a linha DDP
(`python train_loger.py --config-name loger_fsm_ntire trainer.devices=8 trainer.strategy=ddp`),
explicando (a) batch efetivo = `batch_size × devices`, (b) escalar `lr`
linearmente com o batch efetivo, (c) `balanced_sampling` (P0.3) é single-GPU
nesta versão, (d) `val_check_interval` conta steps por rank.

---

## Roadmap e dependências

| Fase | Itens | Dependências | Estimativa |
|------|-------|--------------|-----------|
| P0 | P0.1–P0.5 | — | 1 dia |
| P1 | P1.1–P1.5 | P0 merged | 1–2 dias |
| Run 1 (baseline NTIRE) | treino completo com P0+P1 | P0, P1 | ~30k steps |
| P2 | P2.1–P2.4 (A/Bs) | Run 1 como referência | 2–3 dias + GPU |
| P3 | P3.1–P3.3 | — | 0.5 dia |

Critério global de aceite de cada PR: `pytest tests/` verde (incluindo os testes
novos exigidos por item), dry-run de 20 steps do `loger_fsm_ntire` sem erro, e
nenhuma mudança de comportamento com as flags novas em default.
