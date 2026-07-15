# PLAN v0.2 — Variações de Arquitetura para Avaliação Controlada (NTIRE 2026)

Plano derivado da análise de arquitetura em **[LOGER.md](LOGER.md)** (2026-07-14).
Diferente do PLAN_v0.1 (correções e infraestrutura, já implementado), este plano
é **experimental**: cada item é uma variação de arquitetura materializada como um
**config de experimento** para ser avaliada em A/B contra a baseline. Nada é
adotado sem vencer no protocolo da Seção 1.

Princípios (herdados do v0.1):

- Toda mudança de código entra atrás de **flag com default = comportamento
  atual** (nenhuma run existente muda de resultado sem override explícito).
- Variações que são **config-only** (Grupo 1) não exigem código novo e podem
  rodar imediatamente; as que exigem código (Grupo 2) listam o diff mínimo.
- Cada flag nova de código exige teste unitário correspondente em `tests/`.
- Este documento NÃO implementa nada — especifica o que criar e como avaliar.

---

## 1. Protocolo comum de avaliação

Todas as variantes compartilham um config base de ablação, para que a única
diferença entre runs seja a variação sob teste.

### 1.1 Config base — `configs/ntire_v2_base.yaml` (novo)

Compõe o experimento NTIRE existente e fixa o regime de ablação (Hydra permite
compor um config primário a partir de outro do mesmo diretório):

```yaml
# Base de ablação do PLAN v0.2: regime curto, dados fixos, EMA on.
defaults:
  - loger_fsm_ntire
  - _self_

seed: 0
resume:
  enabled: false            # ablações nunca retomam checkpoint de outra run

data:
  train_shards: [0, 1, 2, 3, 4, 5]
  max_records_per_split:
    train: 100000           # subset fixo => runs de ~8k steps comparáveis
    val: 20000

trainer:
  max_steps: 8000
  val_check_interval: 1000
  limit_val_batches: null

hydra:
  run:
    dir: outputs/ablation_v2/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

Cada variante é um yaml `configs/ntire_v2_<id>.yaml` com
`defaults: [ntire_v2_base, _self_]` + somente os overrides da variação.

### 1.2 Regras do A/B

| Regra | Valor |
|---|---|
| Baseline | `ntire_v2_base` sem overrides (1 run, reutilizada por todos os A/Bs) |
| Seed | 0 para todas as runs (mesma inicialização de heads e ordem de dados) |
| Métrica primária | `val/auc` (pesos EMA) no melhor checkpoint |
| Métricas secundárias | `val/acc` (threshold calibrado), `val/eer`, steps/s, VRAM pico |
| Critério de adoção | Δ AUC ≥ **+0.10 pp** sobre a baseline; empate → variante mais simples |
| Registro | tabela em `docs/ablations_ntire.md` (uma linha por run: config, AUC, acc, EER, steps/s, memória, link W&B) |
| Combinação final | vencedores combinados num config `loger_fsm_ntire_v3` e re-treinados a 30k steps completos antes de qualquer submissão |

Custo estimado por run: 8k steps ≈ 1h na RTX 3060 (bs 48, ~2.2 steps/s);
minutos na DGX B200. Grupo 1 completo ≈ 9 runs.

---

## 2. Grupo 1 — Variações config-only (rodar primeiro; zero código)

### V1 — Sweep do prior MIL top-k (`model.topk_ratio`)

**Hipótese.** O prior de 10% dos patches manipulados (LOGER, Eq. 1) modela
*face swap localizado*. No NTIRE a imagem inteira é sintética: a evidência
está em (quase) todos os patches, e um k pequeno joga fora sinal. Um único
override ajusta modelo E loss de forma consistente (`build_loger_model` e
`LOGERLoss` leem ambos `cfg.model.topk_ratio` —
`src/lightning/loger_module.py:47,76`).

**Configs.** `ntire_v2_topk05 | topk20 | topk40 | topk100`:

```yaml
defaults: [ntire_v2_base, _self_]
model:
  topk_ratio: 0.2   # 0.05 | 0.2 | 0.4 | 1.0 (1.0 = mean pooling, k = N)
```

**Leitura.** Se AUC crescer monotonicamente com o ratio, é evidência direta de
que o prior localizado é o gargalo do branch local no NTIRE — e motiva o V6
(pooling soft) do Grupo 2.

### V2 — Fusão aprendível global vs. local (`model.learnable_fusion`)

**Hipótese.** A média fixa ½/½ pressupõe branches igualmente confiáveis. Já
implementado (P2.1 do v0.1), nunca avaliado. Custa 2 parâmetros.

**Config.** `ntire_v2_learnfusion`:

```yaml
defaults: [ntire_v2_base, _self_]
model:
  learnable_fusion: true
```

**Leitura extra.** Logar/inspecionar os pesos finais (`softmax(fusion_logits)`):
se convergirem fortemente para o global (ex.: 0.8/0.2), reforça V1/V6.

### V3 — Remover a BCE do logit fundido (`loss.fused_weight=0`)

**Hipótese.** O termo `BCE(d, y)` é uma adição deste repo sem contrapartida no
artigo (que treina cada membro do ensemble isoladamente). Ele acopla os
gradientes dos dois branches e pode reduzir a diversidade entre eles — a
propriedade que faz ensembles funcionarem.

**Config.** `ntire_v2_nofused`:

```yaml
defaults: [ntire_v2_base, _self_]
loss:
  fused_weight: 0.0
```

### V4 — Capacidade das cabeças (`model.head_hidden_dim`)

**Hipótese.** As duas cabeças somam ~0.4% dos parâmetros; com backbone em full
fine-tuning, 256 de largura pode ser o fator limitante barato de testar.

**Configs.** `ntire_v2_head512` e `ntire_v2_head1024`:

```yaml
defaults: [ntire_v2_base, _self_]
model:
  head_hidden_dim: 512   # e 1024
```

### V5 — Escada de backbones (a escolha do artigo é DINOv3)

**Hipótese.** O artigo LOGER usa DINOv3 em 4 dos 5 membros do ensemble;
features auto-supervisionadas (sem alinhamento a texto) tendem a preservar
artefatos de baixo nível melhor que encoders contrastivos texto-imagem.

**Configs.** `ntire_v2_dinov2`, `ntire_v2_dinov3`, `ntire_v2_siglip2large`:

```yaml
# ntire_v2_dinov2.yaml — troca de grupo via defaults-override
defaults:
  - ntire_v2_base
  - override /backbone: dinov2_base
  - _self_
```

```yaml
# ntire_v2_siglip2large.yaml — large só cabe no orçamento em LoRA
defaults:
  - ntire_v2_base
  - override /backbone: siglip2_large
  - override /peft: loger_lora
  - _self_
```

Notas: normalização por família é derivada automaticamente de `backbone.name`
pelo entry point (nenhum override manual de stats); `dinov3_base` requer
`transformers>=4.56` — upgrade apenas no ambiente da ablação, com
`pytest tests/` verde antes da run.

### V6 — Train-low / infer-high + TTA (não requer novo treino)

**Hipótese.** O artigo treina membros a 224 e infere a 384; a infraestrutura
equivalente (`model.eval_resolutions`, `model.tta_hflip`) já existe e é
puramente de avaliação — reutiliza o checkpoint da baseline.

**Execução (overrides de test, sem config novo):**

```bash
python test_loger.py --config-name ntire_v2_base ckpt_path=<baseline_best.ckpt> \
  'model.eval_resolutions=[224,336]'            # depois [224,336,448], depois +tta
python test_loger.py --config-name ntire_v2_base ckpt_path=<baseline_best.ckpt> \
  'model.eval_resolutions=[224,336,448]' model.tta_hflip=true
```

Registrar a matriz resolução × TTA em `docs/ablations_ntire.md`; o vencedor
vira o modo de inferência default do config v3.

---

## 3. Grupo 2 — Variações que exigem código mínimo (flag off por default)

Implementar somente APÓS o Grupo 1, e somente os itens cuja hipótese o Grupo 1
não refutar. Cada item lista o diff mínimo e o teste exigido.

### V7 — Focal Loss nos termos de imagem (artigo LOGER, membro M3)

**Hipótese.** O artigo troca CE→Focal no seu modelo global M3 para focar os
exemplos difíceis (deepfakes de alta qualidade). Com balanced sampling, o
ganho esperado vem do foco em *dificuldade*, não do rebalanceamento.

**Código.** `src/losses/loger.py`:
- Nova classe `FocalLoss(nn.Module)` (binária, com logits): 
  `L = α·(1−p_t)^γ · BCE(logit, y)`, defaults `γ=2.0`, `α=0.25`.
- `LOGERLoss.__init__`: novos kwargs `focal: bool = False`,
  `focal_gamma: float = 2.0`, `focal_alpha: float = 0.25`. Quando `focal=True`,
  os termos `global` e `fused` usam Focal em vez de BCE (o `L_CE` local
  permanece BCE — no artigo o Focal é só do modelo global).
- `configs/loss/loger.yaml`: `focal: false`, `focal_gamma: 2.0`,
  `focal_alpha: 0.25`.
- `LOGERLightningModule.__init__`: repassar os 3 kwargs.

**Config.** `ntire_v2_focal`:

```yaml
defaults: [ntire_v2_base, _self_]
loss:
  focal: true
```

**Teste.** `tests/test_focal_loss.py`: com `γ=0, α=0.5`, Focal ≡ 0.5·BCE
(atol 1e-6); com `γ=2`, exemplos com `p_t→1` têm loss ≈ 0; `focal=false`
mantém `LOGERLoss` bit-idêntica à atual (mesma seed → mesmo total).

### V8 — Learning rates discriminativos (backbone vs. cabeças)

**Hipótese.** O artigo usa LRs separados. As cabeças são inicializadas do
zero e podem aprender ~10× mais rápido que o backbone pré-treinado sem
destruir features; hoje um único `lr=3e-5` atende aos dois ao mesmo tempo.

**Código.** `LOGERLightningModule.configure_optimizers`
(`src/lightning/loger_module.py`):
- Novo campo de config `optimizer.head_lr_multiplier: float = 1.0`
  (`configs/optimizer/adamw.yaml`; `1.0` = comportamento atual, um grupo só).
- Quando `!= 1.0`, construir dois param groups: parâmetros de
  `pl_module.model.backbone` com `lr`, todo o resto (global_head, local_head,
  fusion_logits, fsm — que não tem params) com `lr × multiplier`.

**Config.** `ntire_v2_headlr10`:

```yaml
defaults: [ntire_v2_base, _self_]
optimizer:
  head_lr_multiplier: 10.0
```

**Teste.** `tests/test_discriminative_lr.py`: com multiplier 10, o otimizador
tem 2 param groups com lrs 3e-5/3e-4 e a união dos groups cobre exatamente
`trainable_parameters()`; com 1.0, um único group (comportamento atual).
Verificar que o scheduler (SequentialLR) escala os dois groups.

### V9 — Pooling local soft (alternativa diferenciável ao top-k)

**Hipótese.** Se o V1 mostrar sensibilidade forte ao ratio, um pooling sem
corte duro remove o hiperparâmetro: log-sum-exp com temperatura interpola
suavemente entre max (τ→0) e mean (τ→∞), mantendo gradiente para *todos* os
patches.

**Código.** `src/models/loger.py`:
- Nova função `lse_pool(patch_diffs, temperature)`:
  `d_local = τ · logsumexp(d_i / τ) − τ·log(N)` (a correção `−τ·log N` mantém
  a escala comparável ao mean pooling).
- `LOGERModel.__init__`: novos kwargs `local_pool: str = "topk"`
  (`"topk" | "lse"`) e `pool_temperature: float = 1.0`; dispatch no forward.
- **Atenção à consistência com a MILLoss**: a MIL continua usando top-k
  internamente (supervisão dos k maiores). Para `local_pool=lse`, documentar
  que `topk_ratio` segue controlando apenas a loss MIL.
- `configs/model/loger.yaml`: `local_pool: topk`, `pool_temperature: 1.0`.

**Configs.** `ntire_v2_lse_t05 | lse_t10` (`pool_temperature: 0.5 | 1.0`).

**Teste.** `tests/test_lse_pool.py`: τ=100 ≈ mean(d_i) (atol 1e-3); τ=0.01 ≈
max(d_i); gradiente não-nulo em todos os patches; `local_pool=topk` mantém
saída bit-idêntica à atual.

### V10 — Single-Center Loss nas features globais (herança OSDFD)

**Hipótese.** O forward já expõe `scl_features` (penúltima da cabeça global)
sem qualquer supervisão, e `src/losses/single_center_loss.py` já existe no
repo (pipeline OSDFD). A SCL compacta as features reais em torno de um centro
e afasta as fakes — no OSDFD é o que sustenta a generalização open-set, e o
NTIRE ("in the wild", geradores não vistos) é exatamente esse regime.

**Código.**
- `LOGERLoss.__init__`: kwarg `scl_weight: float = 0.0`; quando `> 0`,
  instanciar `SingleCenterLoss` (dim = `head_hidden_dim`, margem 0.01 — os
  valores do OSDFD) e somar `scl_weight · SCL(scl_features, labels)`.
- `LOGERLoss.forward`: novo parâmetro `scl_features` (o Lightning module já
  tem o tensor no `LOGEROutput`; passar no `training_step`).
- `configs/loss/loger.yaml`: `scl_weight: 0.0`.
- Nota: a SCL tem estado próprio (centro aprendível) — vira submódulo da
  loss e entra no checkpoint automaticamente.

**Config.** `ntire_v2_scl`:

```yaml
defaults: [ntire_v2_base, _self_]
loss:
  scl_weight: 1.0   # λ=1 como no OSDFD
```

**Teste.** `tests/test_loger_scl.py`: `scl_weight=0` → total idêntico ao
atual; `scl_weight=1` → total = anterior + SCL calculada à mão num batch
sintético; centro recebe gradiente.

### V11 — Style mixture sem domínio (reativar FSM no NTIRE)

**Hipótese.** O FSM está morto no NTIRE por exigir ≥ 2 domínios de forgery, e
com ele morre toda a augmentation em espaço de features. Relaxar a restrição
de domínio (mistura de estatísticas entre fakes *quaisquer*, estilo MixStyle)
recupera o mecanismo no cenário em que os "domínios" (geradores) existem de
fato nos dados, só não são rotulados.

**Código.** `src/models/fsm.py`:
- `ForgeryStyleMixture.__init__`: novo kwarg
  `require_distinct_domains: bool = True`.
- Quando `False`, `_domain_shuffle` é substituído por um derangement simples
  (`torch.randperm` re-sorteado até `perm[i] != i`, ou shift circular
  aleatório — especificar shift circular: `perm = (arange + randint(1, F)) % F`,
  O(1) e sem laço).
- `configs/fsm/default.yaml`: `require_distinct_domains: true`;
  `build_loger_model` repassa.

**Config.** `ntire_v2_fsm_nodomain`:

```yaml
defaults: [ntire_v2_base, _self_]
fsm:
  enabled: true
  require_distinct_domains: false
```

**Teste.** `tests/test_fsm_nodomain.py`: com a flag, FSM altera tokens de
fakes mesmo com domínio único (todos domain=1); `perm[i] != i` para todo i;
reais intactos; com a flag default, comportamento atual bit-idêntico.

---

## 4. Grupo 3 — Extensão (somente após vencedores do Grupo 1/2 definidos)

### V12 — Mini-ensemble fiel ao artigo (especialistas global e local)

**Hipótese.** A maior divergência estrutural vs. o artigo é treinar os dois
branches acoplados num backbone só. Um mini-ensemble de 2–3 modelos
*especialistas*, treinados independentemente e fundidos por média de logits na
inferência (a formulação exata do artigo, Eq. 3), recupera diversidade sem
novo código de modelo.

**Configs.** Especialistas expressos só com pesos de loss/fusão:

```yaml
# ntire_v2_spec_global.yaml — só o branch global aprende
defaults: [ntire_v2_base, _self_]
model:
  fusion_weights: [1.0, 0.0]
loss: {ce_weight: 0.0, auc_weight: 0.0, mil_weight: 0.0, reg_weight: 0.0, fused_weight: 0.0}
```

```yaml
# ntire_v2_spec_local.yaml — só o branch local aprende
defaults: [ntire_v2_base, _self_]
model:
  fusion_weights: [0.0, 1.0]
loss: {global_weight: 0.0, fused_weight: 0.0}
```

Idealmente com backbones diferentes por especialista (ex.: global = vencedor
do V5, local = DINOv3), espelhando a heterogeneidade do artigo.

**Código (pequeno, fora do modelo).** `scripts/fuse_predictions.py`: lê 2+
`predictions.csv` (o `test_loger.py` já os escreve), alinha por `path`, faz a
média dos logits e recomputa a suíte de métricas
(`src/training/metrics.py:compute_metrics`). Nenhuma mudança em código de
treino/modelo.

**Avaliação.** Ensemble (média de logits) vs. melhor modelo único no mesmo
val split; adotar apenas se Δ AUC ≥ +0.15 pp (ensembles custam N× inferência).

### V13 — Registrado e explicitamente fora de escopo do v0.2

- **Features multi-camada para o branch local** (concat das últimas 4 hidden
  states): promissor, mas invasivo na abstração de backbone (4 famílias a
  tocar) — só considerar se V1/V9 indicarem que o branch local está limitado
  por *features*, não por *pooling*.
- **Attention-MIL (gated attention) na cabeça local**: subsumido pelo V9 com
  metade da complexidade; reavaliar se V9 vencer por margem grande.
- **Layer-wise LR decay**: generalização do V8; só se V8 vencer.
- **Mixup/CutMix de imagem**: conflita com a semântica MIL (já registrado no
  v0.1, mantido fora).
- **Backbones Huge (paridade de escala com o artigo)**: fora do orçamento de
  GPU atual; reavaliar com acesso sustentado à DGX.

---

## 5. Ordem de execução e orçamento

| Onda | Runs | Pré-requisito | Custo (3060 / B200) |
|---|---|---|---|
| 0 | baseline `ntire_v2_base` | configs criados | 1h / min |
| 1 | V1 (×4), V2, V3, V4 (×2), V5 (×3) — 10 runs config-only | onda 0 | ~10h / <1h |
| 1b | V6 (matriz de avaliação, sem treino) | ckpt da onda 0 | ~30min |
| 2 | V7–V11 — 6 runs, cada uma atrás de flag nova | código + testes verdes; hipóteses não refutadas na onda 1 | ~6h / <1h |
| 3 | V12 (2–3 treinos + fusão) | vencedores das ondas 1–2 | ~3h / <1h |
| Final | `loger_fsm_ntire_v3` = combinação dos vencedores, 30k steps completos, shards [0..5] | ondas 1–3 registradas em `docs/ablations_ntire.md` | ~12h / ~1h |

Riscos e controles:

- **Interações entre vencedores** (ex.: V1 topk=0.4 + V9 LSE são mutuamente
  exclusivos; V7 Focal + V10 SCL mudam a escala da loss total): o config v3
  combina no máximo um vencedor por eixo (pooling, objetivo, LR, backbone,
  FSM) e re-valida a combinação numa run de 8k antes dos 30k finais.
- **Subset de 100k pode inverter ranking vs. dataset completo**: os 2 melhores
  de cada eixo com diferença < 0.3 pp são desempatados numa run de 30k.
- **Regressão de comportamento**: `pytest tests/` verde é pré-condição de
  qualquer run da onda 2+ (todas as flags novas têm teste de equivalência com
  default off).
