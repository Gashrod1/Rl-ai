# 🎮 Configuration Rapide - Modes de Jeu

## Configuration Actuelle ✅

```python
# bot.py - lignes ~106-107
spawn_opponents = False  # Mode 1v0
team_size = 1

# bot.py - ligne ~163
load_checkpoint = False  # Démarrage frais
```

**Observation Space**: 231 dimensions (compatible tous modes)

---

## 🔄 Basculer Entre Modes

### 📝 Copier-Coller ces Configurations

#### 1v0 - Entraînement Solo (ACTUEL)
```python
spawn_opponents = False
team_size = 1
load_checkpoint = False  # ou True pour continuer
```

#### 1v1 - Duel
```python
spawn_opponents = True
team_size = 1
load_checkpoint = True  # Continuer depuis 1v0
```

#### 2v2 - Équipe de 2
```python
spawn_opponents = True
team_size = 2
load_checkpoint = True  # Continuer depuis 1v1
```

#### 3v3 - Standard Compétitif
```python
spawn_opponents = True
team_size = 3
load_checkpoint = True  # Continuer depuis 2v2
```

---

## ⚡ Actions Rapides

### Démarrer Nouveau 1v0
1. Vérifier: `load_checkpoint = False`
2. Vérifier: `spawn_opponents = False`, `team_size = 1`
3. `python bot.py`

### Passer en 1v1 (depuis 1v0)
1. Changer: `spawn_opponents = True`
2. Changer: `load_checkpoint = True`
3. `python bot.py`

### Passer en 2v2 (depuis 1v1)
1. Changer: `team_size = 2`
2. Garder: `spawn_opponents = True`, `load_checkpoint = True`
3. `python bot.py`

### Recommencer de Zéro
1. Changer: `load_checkpoint = False`
2. `python bot.py`

---

## 📊 Progression Suggérée

| Étape | Mode | Timesteps | Objectif |
|-------|------|-----------|----------|
| 1 | 1v0 | 5-10M | >30 touches/partie |
| 2 | 1v1 | 10-20M | Win rate >40% |
| 3 | 2v2 | 20-40M | Teamplay basique |
| 4 | 3v3 | 40M+ | Compétitif |

---

## 🎯 Checkpoint de Départ

Votre bot va créer un nouveau checkpoint dans:
```
data/checkpoints/rlgym-ppo-run_<timestamp>/
```

Ce checkpoint fonctionnera pour **TOUS les modes** (1v0, 1v1, 2v2, 3v3) grâce à AdvancedObs!

---

Questions? Voir `GAME_MODES_GUIDE.md` pour plus de détails.
