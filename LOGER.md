# LOGER — Arquitetura do Modelo, Comparação com os Artigos e Processo de Treinamento

Este documento descreve a arquitetura implementada neste repositório
(`src/models/loger.py` e módulos relacionados), compara-a em detalhe com os
três artigos em `docs/` e explica o processo de treinamento. As propostas de
melhoria derivadas desta análise estão em **[PLAN_v0.2.md](PLAN_v0.2.md)**.

Artigos de referência (pasta `docs/`):

| Artigo | Papel neste repo |
|---|---|
| **LOGER: Local–Global Ensemble for Robust Deepfake Detection in the Wild** (arXiv:2604.03558) | Arquitetura de detecção: branch global + branch local com MIL top-k, fusão em logits |
| **Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method with Forgery Style Mixture** (arXiv:2408.12791, "OSDFD") | Módulo FSM (augmentation em espaço de features), receita de treino (lr, batch, steps, oversampling) e modos PEFT |
| **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features** | Backbone default (`google/siglip2-base-patch16-224`) |

---

## 1. Visão geral da arquitetura

O modelo é um detector binário (real vs. fake) de **duas cabeças sobre um
único backbone**, com um módulo de augmentation em espaço de features (FSM)
entre o backbone e as cabeças:

```
                       pixel_values (B, 3, H, W)
                                 │
                ┌────────────────▼─────────────────┐
                │  VFM backbone (token sequence)   │  siglip2 │ dinov2 │ dinov3 │ clip
                └────────────────┬─────────────────┘
                        tokens (B, T, D)
                                 │
                ┌────────────────▼─────────────────┐
                │  Forgery Style Mixture (FSM)     │  só treino · só amostras fake
                └────────────────┬─────────────────┘
                   ┌─────────────┴──────────────┐
       pool(tokens)│                            │patches(tokens)
                   ▼                            ▼
        pooled (B, D)                 patch tokens (B, N, D)
                   │                            │
    ┌──────────────▼─────────┐   ┌──────────────▼───────────────┐
    │ BRANCH GLOBAL          │   │ BRANCH LOCAL                 │
    │ fc(D→256)→ReLU→fc(→2)  │   │ fc(D→256)→ReLU→fc(→2) /patch │
    │ d_global = l_f − l_r   │   │ d_i = l_i^fake − l_i^real    │
    └──────────────┬─────────┘   │ MIL top-k: k = ⌊0.1·N⌋       │
                   │             │ d_local = (1/k) Σ_{i∈Sk} d_i │
                   │             └──────────────┬───────────────┘
                   └─────────────┬──────────────┘
                                 ▼
                d = α_g·d_global + α_l·d_local     P(fake) = σ(d)
```

Escala típica (SigLIP2-base, full fine-tuning): **93.3M parâmetros totais**,
todos treináveis; as duas cabeças somam ~0.4M. Com LoRA r=8 sobre backbone
congelado: ~0.84M treináveis.

### 1.1 Backbone (abstração VFM)

`src/models/backbones.py` define um contrato único para qualquer Vision
Foundation Model, escolhido por `backbone.name`:

```python
tokens  = backbone(pixel_values)   # (B, T, D)  sequência completa de tokens
pooled  = backbone.pool(tokens)    # (B, D)     embedding global da imagem
patches = backbone.patches(tokens) # (B, N, D)  apenas os patch tokens
```

| Família | Checkpoint default | Pooling global | Tokens removidos por `patches()` |
|---|---|---|---|
| `siglip2` | `google/siglip2-base-patch16-224` | **MAP head** (attention pooling) | nenhum (não há `[CLS]`) |
| `dinov2` | `facebook/dinov2-base` | token `[CLS]` | 1 (`[CLS]`) |
| `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | token `[CLS]` | 1 + register tokens |
| `clip` | `openai/clip-vit-base-patch16` | token `[CLS]` | 1 (`[CLS]`) |

O backbone **nunca é modificado estruturalmente**: FSM, pooling e seleção de
patches são pós-processamento sobre a sequência de tokens. Modos PEFT
(`peft.mode`): `full` (default, segue o artigo LOGER), `lora` (LoRA r=8 em
q/k/v da atenção, backbone congelado) e `frozen` (só as cabeças).

### 1.2 Forgery Style Mixture (FSM)

`src/models/fsm.py` — implementação fiel às Eqs. 6–9 do artigo OSDFD:

1. Ativo **só em treino**, com probabilidade 0.5 por forward; amostras reais
   passam intactas.
2. Cada amostra fake é pareada com uma fake de **domínio de forgery
   diferente** (Deepfakes ↔ Face2Face etc.); se houver menos de dois domínios
   no batch, é no-op.
3. As estatísticas AdaIN (média/desvio por canal, sobre a dimensão de tokens)
   dos dois ordenamentos são misturadas com peso `δ ~ Beta(0.1, 0.1)`
   (bimodal: quase sempre δ≈0 ou δ≈1, ou seja, "troca de estilo" quase pura):

   ```
   γ_mix = δ·σ(F) + (1−δ)·σ(F̃)          η_mix = δ·μ(F) + (1−δ)·μ(F̃)
   F'    = γ_mix · (F − μ(F)) / σ(F) + η_mix
   ```

O objetivo é diversificar os *estilos de forgery* vistos em treino, melhorando
a generalização a métodos de manipulação não vistos. Como o FSM age sobre a
sequência de tokens **antes** do pooling e das cabeças, ambos os branches veem
as estatísticas misturadas.

**Limitação estrutural relevante para o NTIRE**: FSM exige rótulos de domínio
(≥ 2 tipos de manipulação). O NTIRE só tem rótulo binário, então
`fsm.enabled=false` no config `loger_fsm_ntire` — a augmentation em espaço de
features fica totalmente inativa nesse dataset (ver proposta F1 no
PLAN_v0.2.md).

### 1.3 Branch global

`backbone.pool(tokens)` produz o embedding da imagem (MAP head no SigLIP2,
`[CLS]` nos demais). O `ImageClassifier` (`fc(D→256) → ReLU → fc(256→2)`,
dropout 0.1) emite dois logits; o score do branch é a diferença
`d_global = l_fake − l_real`. A feature penúltima (256-d) é exposta como
`scl_features` — hoje **não supervisionada** (reservada para uma loss de
feature como a SCL do OSDFD; ver proposta B3).

### 1.4 Branch local (MIL top-k)

Cada patch token passa pelo `PatchClassifier` compartilhado
(`fc(D→256) → ReLU → fc(256→2)`), gerando evidência por patch
`d_i = l_i^fake − l_i^real`, shape `(B, N)` (N = 196 para 224px/patch16). A
agregação usa **MIL top-k pooling**:

```
k = max(1, ⌊ topk_ratio · N ⌋)        # default 0.1 → k = 19
d_local = (1/k) · Σ_{i ∈ S_k} d_i     # S_k = índices dos k maiores d_i
```

Racional (hipótese de múltiplas instâncias): um rosto manipulado contém *ao
menos alguns* patches fortemente forjados; a média dos top-k torna o score
sensível a manipulações localizadas e ignora o fundo. **Nota crítica**: esse
prior de 10% assume forgeries localizadas (face swap). Em síntese de imagem
inteira (NTIRE, difusão/GAN), todos os patches são "fake" — o prior é
subótimo e é um dos eixos de ablação do PLAN_v0.2 (propostas A2/A3).

### 1.5 Fusão

`d = α_g·d_global + α_l·d_local` **em espaço de logits** (não de
probabilidades), com pesos uniformes ½/½ por default — a formulação de
ensemble do artigo LOGER (`d̄ = Σ α_m d_m`, `p = σ(d̄)`), aplicada aqui aos dois
branches de um único backbone. Alternativas já implementadas via config:
pesos fixos customizados (`model.fusion_weights=[a,b]`) ou aprendidos
(`model.learnable_fusion=true`, 2 parâmetros via softmax).

### 1.6 Inferência

- **Multi-resolução**: `forward_multi_resolution` redimensiona (bicúbico) a
  cada resolução de `model.eval_resolutions`, roda o modelo com FSM off e
  faz a média dos logits fundidos (com cache da resolução nativa para não
  pagar o forward base duas vezes).
- **TTA horizontal-flip** (`model.tta_hflip`): média de `logits(x)` e
  `logits(hflip(x))` no `predict_step`.
- **Threshold calibrado**: o threshold de decisão que maximiza acurácia no
  val é calculado a cada validação e salvo no checkpoint (`best_threshold`);
  a predição usa esse valor em vez de 0.5 fixo.

---

## 2. Objetivo de treinamento

`src/losses/loger.py`, pesos em `configs/loss/loger.yaml`:

```
L = 1.0 · BCE(d_global, y)          supervisão do branch global
  + 1.0 · BCE(d, y)                 supervisão do logit fundido
  + 1.0 · BCE(d_local, y)           L_CE  do branch local
  + 0.5 · L_AUC(d_local, y)         surrogate pairwise de AUC
  + 0.5 · L_MIL({d_i}, y)           supervisão por patch (MIL)
  + 1e-4 · L_reg({d_i})             L2 nos logits de patch
```

O artigo LOGER define o objetivo local como
`L_local = L_CE + 0.5·L_AUC + 0.5·L_MIL + L_reg`, **sem dar formas fechadas**
para os três últimos termos. As formulações implementadas (escolhas padrão):

- **L_AUC** — squared hinge pairwise: para cada par (fake i, real j) do batch,
  `mean max(0, margin − (s_i − s_j))²`, margem 1.0. Otimiza diretamente o
  ranqueamento que o AUC mede; retorna 0 em batches de classe única.
- **L_MIL** — supervisão em nível de patch: imagens reais têm *todos* os `d_i`
  puxados para 0 (BCE); imagens fake têm os *top-k* `d_i` puxados para 1
  (mesmo k do pooling).
- **L_reg** — `mean(d_i²)`, impede que os logits de patch explodam sob a
  pressão do hinge/MIL.

Dois termos são **adições deste repo** sem contrapartida no artigo (que treina
cada modelo do ensemble separadamente): a BCE do logit fundido e a BCE do
branch global co-treinada com o local no mesmo backbone.

---

## 3. Processo de treinamento

Loop: PyTorch Lightning 2.x (`src/lightning/loger_module.py`), entry point
`train_loger.py`, configuração 100% via Hydra.

### 3.1 Dados

1. **Descoberta de records** — FF++ (árvore `<split>/<classe>/**`, domínio =
   pasta de manipulação), NTIRE (shards `shard_N/labels.csv + images/`) ou
   manifest CSV arbitrário. No NTIRE, val é um hold-out determinístico de 5%
   por hash MD5 do filename (sem shard de teste real no release).
2. **Balanceamento** — FF++: reais duplicados ×4 (convenção OSDFD); NTIRE:
   `WeightedRandomSampler` com pesos inversos à frequência de classe.
3. **Augmentation** (só treino) — degradações com probabilidades
   independentes: hflip, resize aleatório, recompressão JPEG, blur gaussiano,
   motion blur, ruído gaussiano, brilho/contraste/color jitter, random crop,
   random erasing. Val/test: resize + normalize apenas. No FF++ segue o OSDFD
   (sem augmentation de pixel); no NTIRE está ligada (robustez "in the wild").
4. **Normalização** — estatísticas por família de backbone, derivadas
   automaticamente de `backbone.name`.

### 3.2 Otimização

| Aspecto | Valor | Origem |
|---|---|---|
| Otimizador | AdamW, lr 3e-5, wd 1e-2, β=(0.9, 0.999) | OSDFD (Adam 3e-5, batch 48) / LOGER (AdamW, wd 1e-2) |
| Schedule | cosseno por step, `t_max=30000`, warmup linear 500 steps (NTIRE) | warmup é adição deste repo |
| Batch | 48 (por GPU) | OSDFD |
| Duração | 30k steps, val a cada 1000 | OSDFD |
| Precisão | fp16 AMP (FF++) / bf16 (NTIRE) | — |
| Grad clipping | 1.0 (norma L2) | LOGER |
| EMA | decay 0.999 sobre `model.state_dict()`, avaliação e `best.ckpt` com pesos EMA via swap | adição deste repo |

### 3.3 Passo de treino e avaliação

```
training_step:
  out  = model(x, is_fake=label, domains=domain, apply_fsm=True)   # FSM ON
  loss = LOGERLoss(out.logits, out.global_logits, out.local_logits,
                   out.patch_diffs, label)
validação/test:  FSM OFF; scores σ(d) bufferizados por época, gather entre
  ranks DDP; métricas: acc, AUC (primária), F1, precision, recall, AP, EER,
  FPR, FNR; breakdown de AUC/acc por shard (NTIRE) ou por manipulação (FF++);
  calibração de threshold; figuras de confusão/ROC/PR no test.
checkpointing:  best.ckpt (max val/auc) + last.ckpt; auto-resume do último
  last.ckpt do experimento; config Hydra completo embutido no checkpoint.
```

---

## 4. Comparação com os artigos

### 4.1 vs. LOGER (arXiv:2604.03558)

O artigo é fundamentalmente um **ensemble de cinco modelos independentes**;
este repo destila o *design* (global + local + fusão de logits) em **um único
backbone com duas cabeças co-treinadas**.

| Dimensão | Artigo LOGER | Este repo |
|---|---|---|
| Estrutura | 5 modelos independentes: M1/M2 = DINOv3-Huge (global), M3 = MetaCLIP2-Huge (global), M4/M5 = DINOv3-Large (local); fusão de logits só na inferência | 1 backbone (SigLIP2-base default) com cabeça global + cabeça local, treinadas em conjunto |
| Escala | Huge/Large (≥ 300M–1B params por membro) | Base (93M) — ordem de magnitude menor |
| Cabeça local | Idêntica: classificador por patch, `d_i = l_fake − l_real`, MIL top-k `k=⌊0.1·N⌋`, `L_CE + 0.5·L_AUC + 0.5·L_MIL + L_reg` | Igual (formas fechadas dos 3 termos são escolhas padrão, artigo não especifica) |
| Cabeça global | MLP 2-logits sobre embedding pooled | Igual (D→256→2, ReLU, dropout 0.1) |
| Fusão | `d̄ = Σ α_m d_m`, α uniforme, `p = σ(d̄)` — entre 5 modelos | Mesma fórmula — entre 2 branches; termo extra de BCE no logit fundido (não existe no artigo) |
| Fine-tuning | **Full fine-tuning**; rejeita explicitamente LoRA ("partial adaptation tends to underfit forgery-relevant cues") | `peft.mode=full` default; `lora`/`frozen` disponíveis como ablações |
| Loss global | M3 usa **Focal Loss** (troca CE→Focal após 20% do treino) | BCE em todos os termos de imagem |
| Learning rates | **Discriminativos** (backbone < cabeças) | LR único para tudo |
| Resolução | Train-low/infer-high por membro (ex.: M4 treina 224, infere 384) | `eval_resolutions` faz média multi-resolução de um único modelo |
| TTA | hflip em M1/M2/M4/M5 | `model.tta_hflip` disponível (default off) |
| Sampler | `WeightedRandomSampler` | Igual no NTIRE; FF++ usa oversample ×4 |
| L_MIL em reais | Supervisão descrita só nos top-k | Aqui *todos* os patches de imagens reais são supervisionados a 0 (mais forte) |

**Leitura crítica**: as maiores lacunas em relação ao artigo — em provável
ordem de impacto — são (1) escala/diversidade do ensemble, (2) Focal Loss no
global, (3) LRs discriminativos, (4) train-low/infer-high. Todas são
endereçáveis sem reescrever a arquitetura e são a espinha dorsal do
PLAN_v0.2.md.

### 4.2 vs. OSDFD / FSM (arXiv:2408.12791)

| Dimensão | Artigo OSDFD | Este repo (pipeline LOGER+FSM) |
|---|---|---|
| FSM | Eqs. 6–9: fake-only, pareamento cross-domain, `δ~Beta(0.1,0.1)`, prob. 0.5, só treino, após os blocos transformer | **Idêntico** (`src/models/fsm.py`) |
| Adaptação | Backbone congelado + LoRA q/k/v (r=8) + adapter CDC no FFN | LOGER usa full FT por default; LoRA disponível; CDC só no pipeline OSDFD preservado |
| Loss | `BCE + λ·SCL` (Single-Center Loss, margem 0.01, λ=1, sobre features penúltimas) | LOGER usa o objetivo da Seç. 2 — **SCL não é usada** (embora `scl_features` já esteja exposta no forward) |
| Receita | Adam 3e-5, batch 48, 30k steps, reais ×4, sem augmentation de pixel, faces 224² com margem 1.3 | Mesma base (AdamW), com adições: warmup, EMA, augmentation (NTIRE), balanced sampler |
| Backbone | ImageNet-21K ViT-B / CLIP ViT-L | SigLIP 2 (o propósito desta re-implementação) |

O papel do OSDFD neste repo é fornecer o **FSM** e a **receita de treino**; a
arquitetura de detecção vem do LOGER.

### 4.3 vs. SigLIP 2 (backbone)

Por que SigLIP 2 como default:

- **MAP head (attention pooling)** em vez de token `[CLS]`: o embedding
  global é uma média ponderada aprendida de *todos* os patch tokens — casa
  bem com um detector cuja evidência pode estar espalhada (síntese de imagem
  inteira) ou localizada (face swap).
- **Pré-treino denso**: além da loss sigmoid contrastiva do SigLIP original,
  o SigLIP 2 adiciona objetivos de predição densa/auto-distilação e
  captioning, melhorando explicitamente *localization* e *dense features* —
  exatamente o que o branch local consome (features por patch de qualidade).
- **Sem `[CLS]`**: a sequência inteira de tokens é usada pelos dois branches
  sem descarte, e o FSM mistura estatísticas sobre todos os tokens.

Contraste com as alternativas plugáveis: DINOv2/v3 são auto-supervisionados
puros (sem alinhamento texto-imagem — features mais "fotométricas", boas para
artefatos de baixo nível; o artigo LOGER usa DINOv3 em 4 dos 5 membros);
CLIP é o baseline histórico (features semânticas, patch tokens mais fracos).
A escolha do artigo LOGER (DINOv3 dominante no ensemble) sugere que a família
DINO é forte para esta tarefa — motivação direta da ablação de backbones
(proposta D1).

---

## 5. Propostas de melhoria (resumo)

Análise completa, com configs de avaliação e protocolo A/B, em
**[PLAN_v0.2.md](PLAN_v0.2.md)**. Em síntese, os eixos identificados:

1. **Pooling local ciente do tipo de forgery** — o prior top-k de 10% é feito
   para face swap; para síntese de imagem inteira (NTIRE) avaliar
   `topk_ratio` maior e pooling soft (LSE/softmax com temperatura).
2. **Objetivo global mais próximo do artigo** — Focal Loss no termo global
   (o artigo LOGER a usa no M3); opcionalmente SCL nas `scl_features` já
   expostas (herança OSDFD de custo quase nulo).
3. **LRs discriminativos backbone vs. cabeças** — o artigo os usa; cabeças
   inicializadas do zero pedem LR maior que o backbone pré-treinado.
4. **Train-low/infer-high e TTA** — replicar a estratégia de resolução do
   artigo com um único modelo (config-only).
5. **Escada de backbones** — DINOv3 (a escolha do artigo) vs. SigLIP2 vs.
   large em LoRA, sob protocolo fixo.
6. **Estilo-mixture sem domínio para NTIRE** — variante do FSM que mistura
   estatísticas entre fakes sem exigir domínios distintos (MixStyle-like),
   reativando a augmentation de features no dataset que hoje não a usa.
7. **Capacidade das cabeças** — largura/profundidade dos MLPs (0.4% dos
   parâmetros; margem barata a explorar).
8. **Mini-ensemble fiel ao artigo** — treinar especialistas global-only e
   local-only e fundir logits na inferência, aproximando a formulação
   original de ensemble com o orçamento disponível.
