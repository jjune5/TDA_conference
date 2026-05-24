---
title: Persistent Homology가 Link Prediction에 정말 도움 되나?
subtitle: 도메인별 분석과 Adaptive Topology Gating
author: 박준영
date: 2026-06-21
marp: true
theme: gaia
paginate: true
---

<!-- _class: lead -->

# Persistent Homology가 Link Prediction에 정말 도움 되나?

### 도메인별 분석과 Adaptive Topology Gating

박준영 · TDA 학회 · 2026-06-21

---

# 오늘의 질문

> "Topology가 link prediction에 도움이 된다"
> — TLC-GNN (ICML 2021)

**진짜?**

우리가 검증해본 결과: **도메인 따라 다르다.** 어떤 데이터엔 오히려 **해롭다.**

---

# 1. TDA란 무엇인가

**Topological Data Analysis** — 데이터의 "모양"을 분석

- 거리/연결성에 robust → noise에 강함
- 도구: **persistent homology**

![bg right:40% 95%](https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Persistent_homology_filtration.svg/440px-Persistent_homology_filtration.svg.png)

---

# Persistent Homology

그래프 (또는 데이터)를 **점점 키우면서** topological feature 추적:
- 0-dim feature = **연결 component** (몇 개?)
- 1-dim feature = **loop / hole** (몇 개?)

각 feature: **birth** (생긴 시점) → **death** (사라지는 시점)

오래 사는 feature = robust, 빨리 죽는 feature = noise

---

# Persistent Diagram (PD)

각 feature를 (birth, death) 점으로 plot:

```
death
 │
 │     • robust loop
 │  •
 │•           • noise
 │
 └─────────────── birth
        diagonal (=noise)
```

대각선에서 멀수록 robust.

---

# Persistent Image (PI)

PD를 **5×5 grid에 Gaussian으로 흐려서** vector화 (25-dim)

```
[0.0 0.1 0.3 0.1 0.0]
[0.0 0.2 0.5 0.3 0.0]
[0.0 0.1 0.4 0.2 0.0]
[0.0 0.0 0.1 0.0 0.0]
[0.0 0.0 0.0 0.0 0.0]
```

ML 모델의 입력 feature로 사용 가능.

---

# 2. TLC-GNN (ICML 2021)

GCN + PI를 link prediction에 결합:

```python
emb_u = GCN(u의 feature)
emb_v = GCN(v의 feature)
PI(u, v) = persistent_image(vicinity_subgraph)
features = concat[|emb_u − emb_v|², PI(u, v)]
prob = MLP(features)
```

논문 주장: PubMed/Photo/Computers에서 SOTA.

---

# PDGNN (NeurIPS 2022)

문제: PD 계산이 **너무 느림** (Squirrel: 80시간)

해결: **GNN으로 PD를 근사**

- Input: graph + filter values
- Output: per-edge (birth, death)
- 학습: dionysus가 만든 ground-truth PD로 지도학습

**속도: 100× 빠름.** 정확도: 비슷한가? (놀라움 있음)

---

# 3. 우리가 한 일

1. **재현** — TLC-GNN + PDGNN, PyTorch 2.1에서 동작하도록 modernize
2. **9개 도메인으로 확장** — homo (3) + hetero (5) + drug (1)
3. **Ablation** — PI on/off 비교
4. **SBM density × heterophily sweep** — 합성 그래프로 인과 측정
5. **Adaptive PI Gating** — 자동 의사결정 방법 제안

GitHub: **github.com/jjune5/TDA_conference**

---

# 결과: 3-way 비교

| | TLC-GNN | PDGNN | No PI |
|---|---|---|---|
| **Photo** | 0.9825 | **0.9860** | — |
| **PubMed** | 0.9635 | **0.9669** | — |
| **Computers** | 0.9680 | **0.9830** | — |
| **Chameleon** | 0.943 | 0.945 | **0.969** |
| **Texas** | 0.571 | 0.540 | **0.594** |
| **ChChMiner** | 0.903 | 0.963 | **0.965** |

3 가지 도메인이 다른 패턴 보임.

---

# 발견 1: 도메인이 결정

- **Homophilic Citation/Amazon** (Photo / PubMed / Computers): **PI 도움** ✓
- **Heterophilic Wiki/Web** (Chameleon / Squirrel / WebKB): **PI 무용 또는 유해**
- **Drug Interaction** (ChChMiner): **PI 명확히 해로움** (−6%p)

→ Paper의 "topology helps LP" 주장은 **도메인 의존적**이다.

---

# 발견 2: PDGNN의 의외성

| | TLC-GNN exact | PDGNN approx | Δ |
|---|---|---|---|
| Photo | 0.9825 | 0.9860 | **+0.35%p** |
| PubMed | 0.9635 | 0.9669 | **+0.34%p** |
| Computers | 0.9680 | 0.9830 | **+1.50%p** |

**Neural approximation이 dionysus exact PD보다 더 좋다.**

해석: smoothing 효과로 high-frequency noise 제거 → 더 robust한 feature.

---

# 발견 3: Density × Heterophily 패턴 (B)

(SBM 5×5 sweep 결과 — 실험 완료 후 채움)

![bg right:50%](../docs/figures/sbm_heatmap.png)

3-panel heatmap:
1. TLC-GNN AUC
2. PDGNN AUC
3. **PI hurt magnitude** = no-PI − TLC-GNN

→ density × heterophily가 클수록 PI 해로움 정량적 패턴.

---

# 우리 제안: Adaptive PI Gating

```python
gate = sigmoid(GatingNet(edge_features))   # ∈ [0, 1]
features = concat[|emb_u − emb_v|², gate × PI(u, v)]
prob = MLP(features)
```

GatingNet 입력: clustering coeff, embedding distance.

**모델이 자동으로 결정**:
- Homo edge → gate ~1 (PI on)
- Hetero edge → gate ~0 (PI off)

---

# Adaptive Gating 결과 (TBD)

| | TLC-GNN | Gated | No-PI | Mean gate |
|---|---|---|---|---|
| Photo (homo) | 0.9825 | TBD | — | TBD |
| Chameleon (hetero) | 0.943 | TBD | 0.969 | TBD |
| Texas (hetero) | 0.571 | TBD | 0.594 | TBD |
| ChChMiner (drug) | 0.903 | TBD | 0.965 | TBD |

(실험 완료 후 채움)

---

# 4. 전망

- 🧬 **Drug discovery** — OGBL-DDI, BIOSNAP scale-up (PDGNN 가속 필수)
- 👥 **Social network** — heterophily 강한 도메인 (adaptive gating fit)
- 🧠 **Brain connectivity** — TDA의 sweet spot
- 🔗 **Heterogeneous KG** — drug-protein-disease multi-relation

→ Topology는 만능 아님. **언제 / 어디서** 쓸지 적응적 결정 필요.

---

# 정리

1. PI가 link prediction에 **무조건 도움 ≠ 사실**
2. **도메인** (density × heterophily) 의존
3. **PDGNN approximation**은 의외로 더 robust
4. **Adaptive gating**: 자동 적응 방법 제안

---

<!-- _class: lead -->

# 질문?

GitHub: **github.com/jjune5/TDA_conference**

박준영 · jjune5@naver.com
