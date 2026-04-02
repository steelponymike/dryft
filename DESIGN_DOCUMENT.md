# Dryft: Ecological Memory Architecture for AI Systems
## Design Document — Prior Art Record

**Author:** Mike Kozlowski
**Date:** April 2, 2026
**Version:** 1.0
**Status:** Working implementation, open source release

---

## 1. Summary of Invention

Dryft is a memory management architecture for AI systems that applies ecological principles — fitness emergence, proximity bonding, predator-prey dynamics, decomposition, and multi-type decay — to create a self-regulating memory population. Unlike existing approaches that treat memory as static storage with manual or rule-based curation, Dryft memories behave as living entities in an ecosystem: they strengthen through use, bond through co-activation, weaken through disuse, and die when they can no longer justify their existence. The system self-regulates without human intervention.

## 2. Problem Statement

Current AI memory systems share a fundamental architectural limitation: they only accumulate. Memories are added through extraction or summarization and persist indefinitely unless manually deleted. This creates several compounding problems:

- **Memory bloat**: Unbounded growth degrades retrieval quality over time
- **Equal weighting**: A trivial fact and a critical decision have identical persistence
- **No relational structure**: Flat vector search cannot represent relationships between concepts that emerge from usage patterns
- **No self-regulation**: The system never removes what it no longer needs
- **No lifecycle**: Memories have no birth, maturation, aging, or death

## 3. Core Innovations

### 3.1 Fitness-Based Memory Lifecycle

Each memory object carries a fitness score (0.0 to 1.0) that changes dynamically based on activation. When a memory is retrieved in response to a query, its fitness increases. Between activations, fitness decays at a rate determined by memory type:

- **Episodic memories** (events, conversations): decay rate 0.015 per query cycle
- **Semantic memories** (facts, knowledge): decay rate 0.005 per query cycle
- **Procedural memories** (processes, how-to): decay rate 0.0 (no passive decay)

This differential decay mirrors biological memory consolidation, where episodic memories fade faster than semantic knowledge, and procedural skills persist indefinitely once learned.

### 3.2 Proximity Bonding Through Co-Activation

When two or more memories are activated in response to the same query, their mutual bond score increases. Bond strength accumulates over repeated co-activations, progressing through relationship stages:

- **Strangers** (default): no bond
- **Acquainted**: 3+ co-activations, bond score >= 0.15
- **Bonded**: 6+ co-activations, bond score >= 0.30
- **Dating**: 10+ co-activations, bond score >= 0.50

Bonds that are not reinforced drift apart at a rate of 0.008 per query cycle. This creates an emergent relational graph without entity extraction, knowledge graph construction, or manual mapping. The relationships represent actual usage patterns, not inferred semantic similarity.

When a memory is activated, its bonded partners receive a retrieval boost proportional to bond strength, enabling associative recall across topics that co-occur in practice.

### 3.3 Predator-Prey Dynamics

Memories below a fitness threshold (0.1) become "prey eligible." A grace period (20 query cycles) protects young memories. After 5 consecutive cycles of prey eligibility, the memory is culled.

The predator mechanism is the critical differentiator. No other memory system in the current landscape implements automatic removal based on ecological fitness. This prevents unbounded growth, improves retrieval signal-to-noise ratio over time, and creates evolutionary pressure that sharpens the memory population.

**Suspicious death flagging**: The system flags culls that meet criteria suggesting potential value (3+ prior activations, bonded relationships, or rehydration history) for human review, creating an audit trail without blocking the cull.

### 3.4 Decomposition and Substrate Cycling

When a memory is culled through natural predation (not conflict resolution), its content is analyzed for reusable patterns. If the content meets a richness threshold, it decomposes into the grass layer — a substrate of generalized knowledge fragments. These fragments can later be synthesized into new memories or influence signal extraction.

This mirrors nutrient cycling in grassland ecosystems: death feeds new growth. The system's knowledge doesn't just shrink when memories die; it transforms.

**Humane dispatch** is an alternative death path for memories culled through conflict resolution. Because wrong or contradictory information has zero substrate value, humane dispatch skips decomposition entirely. The memory is removed without feeding the grass.

### 3.5 Multi-Herd Architecture (Mob Grazing)

The system maintains two separate herds running the same engine:

- **Main herd (cattle)**: Operational memories — facts, events, decisions, procedures. Starting fitness 0.5.
- **Evaluation herd (sheep)**: Evaluative memories — opinions, preferences, assessments, recommendations. Starting fitness 0.35.

Both herds share the same grass layer but maintain separate bond graphs. An evaluation gate controls access to the eval herd, opening only for explicit references to evaluated entities. This prevents evaluative memories from contaminating factual recall.

This architecture is directly inspired by mob grazing in regenerative agriculture, where different livestock species graze the same pasture in sequence, each contributing differently to soil health.

### 3.6 Six-Layer Architecture

1. **Foundational layer**: Permanent core knowledge that never decays (the "cowbirds" — they live with the herd but aren't part of it)
2. **Grass layer**: Substrate of decomposed memory nutrients, both directly inscribed and emergently synthesized
3. **Main herd**: Active operational memories under full ecological dynamics
4. **Evaluation herd**: Active evaluative memories with separate fitness thresholds
5. **Dormancy staging**: Incubator for new signals — signals must mature before entering the herd
6. **Temporal layer**: Time-aware reasoning including supersession detection (similarity > 0.80, age gap > 50 queries) and generational tracking via parent_ids

### 3.7 Conflict Detection and Resolution

An LLM-based classifier identifies contradictory memories across four categories. Detected conflicts enter a resolution queue with annoyance throttling (max 1 per session, 2 per 24 hours). Resolution persistence ensures resolved conflicts never resurface.

Resolution paths:
- **User confirms one memory is wrong**: Humane dispatch (immediate cull, no decomposition)
- **User confirms both are partially right**: Merge into updated memory
- **User defers**: Conflict tracked but not resurfaced within the same session

### 3.8 Rehydration

A culled memory can spontaneously recover if it becomes relevant again. When a query strongly matches a recently culled memory, the memory re-enters the herd with its fitness reset. This was an emergent behavior observed during testing — not explicitly designed — and was preserved because it mirrors biological recovery patterns.

### 3.9 Temporal Inference (Carbon Dating)

The system maintains a query-to-calendar mapping that allows temporal inference from ecological metadata. A memory's created_at field (query number at time of creation) can be mapped to approximate calendar dates. This enables:

- Supersession advisory: identifying when a newer memory likely replaces an older one
- Temporal context in conflict resolution
- Generational tracking through parent_ids

## 4. Implementation Status

As of April 2, 2026:
- Core engine: fully implemented (Python)
- Telegram bot integration: live, running 24/7 on VPS
- Vector similarity scoring: implemented (all-MiniLM-L6-v2, pre-computed embeddings)
- Conflict detection and resolution: implemented with persistence
- Temporal layer: implemented
- Custom benchmark (56 queries): 83% weighted score
- LOCOMO standardized benchmark: 50% (vs Mem0 66.9%, OpenAI Memory 53%)
- Active memory population: ~290 memories in main herd, 26 in eval herd
- Daily use period: multiple weeks of continuous operation

## 5. Differentiation from Existing Systems

| Capability | Dryft | Mem0 | OpenAI Memory | Zep | Letta |
|-----------|-------|------|--------------|-----|-------|
| Fitness-based lifecycle | Yes | No | No | No | No |
| Proximity bonding | Yes | No | No | No | No |
| Automatic predator culling | Yes | No | No | No | No |
| Decomposition/substrate cycling | Yes | No | No | No | No |
| Differential decay by type | Yes | No | No | Partial | No |
| Conflict detection | Yes | Partial (UPDATE/DELETE ops) | No | No | No |
| Multi-herd separation | Yes | No | No | No | No |
| Rehydration | Yes | No | No | No | No |
| Temporal supersession | Yes | No | No | Partial | No |

## 6. Design Methodology

The architecture is derived from direct observation of ecological systems over 15 years of working with livestock, soil, and plant communities in Western Canada. Key analogies:

- **Fitness emergence** ← natural selection in grazing populations
- **Proximity bonding** ← herd behavior and social bonding in cattle
- **Predator-prey** ← predator pressure maintaining herd health
- **Decomposition** ← nutrient cycling in grassland soil
- **Mob grazing** ← rotational grazing with multiple species
- **Grass layer** ← soil substrate that feeds new plant growth
- **Dormancy staging** ← seed dormancy and germination conditions
- **Rehydration** ← perennial regrowth from dormant root systems

These are not metaphors applied after the fact. They are the design process: each feature was conceived by asking "how does nature solve this problem?" and translating the ecological mechanism into a computational analog.

## 7. Licensing

Released under MIT License. This document establishes prior art and authorship for the ecological memory architecture described herein.

---

**Signed digitally by inclusion in git repository and email timestamp.**
**Author: Mike Kozlowski**
**Date: April 2, 2026**
