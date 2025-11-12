# Guide des Modes de Jeu - Configuration Universelle

## ✅ Configuration Actuelle

Votre bot utilise maintenant **AdvancedObs** qui fonctionne avec TOUS les modes:
- **Observation size: 231** (fixe pour tous les modes)
- Pas besoin de retrain quand vous changez de mode!

---

## 🎮 Comment Changer de Mode de Jeu

Dans `bot.py`, lignes ~95-105:

### Mode 1v0 (Entraînement de base)
```python
spawn_opponents = False
team_size = 1
```
👉 Le bot apprend seul: toucher la balle, marquer, bases aériennes

### Mode 1v1 (Combat singulier)
```python
spawn_opponents = True
team_size = 1
```
👉 Le bot apprend: défendre, attaquer, positionner contre 1 adversaire

### Mode 2v2 (Jeu d'équipe)
```python
spawn_opponents = True
team_size = 2
```
👉 Le bot apprend: passes, rotations, jouer avec coéquipier

### Mode 3v3 (Compétitif standard)
```python
spawn_opponents = True
team_size = 3
```
👉 Le bot apprend: rotations complexes, jeu d'équipe avancé

---

## 🔄 Progression d'Entraînement Recommandée

### Phase 1: 1v0 (Actuellement configuré ✅)
**Durée**: Jusqu'à ce que le bot touche la balle régulièrement (~5-10M timesteps)

**Objectifs**:
- Toucher la balle (>50 touches par partie)
- Marquer des buts (>5 buts toutes les 10 parties)
- Boost management basique

**Réglage actuel**:
```python
load_checkpoint = False  # Repartir de zéro
spawn_opponents = False
team_size = 1
```

### Phase 2: 1v1
**Durée**: 10-20M timesteps

**Objectifs**:
- Jouer contre adversaire
- Défendre son but
- Gagner >40% des matchs

**Changements à faire**:
```python
load_checkpoint = True   # Continuer l'entraînement 1v0
spawn_opponents = True   # Ajouter adversaire
team_size = 1
```

### Phase 3: 2v2
**Durée**: 20-40M timesteps

**Objectifs**:
- Jouer avec coéquipier
- Faire des passes
- Rotations basiques

**Changements à faire**:
```python
load_checkpoint = True   # Continuer depuis 1v1
spawn_opponents = True
team_size = 2           # Passer en 2v2
```

### Phase 4: 3v3
**Durée**: 40M+ timesteps

**Objectifs**:
- Rotations complexes
- Jeu d'équipe avancé
- Niveau compétitif

**Changements à faire**:
```python
load_checkpoint = True   # Continuer depuis 2v2
spawn_opponents = True
team_size = 3           # Passer en 3v3
```

---

## 🔧 Démarrer l'Entraînement

### Pour Commencer (Actuellement configuré ✅)
```bash
python bot.py
```

Vous verrez:
```
🆕 Starting fresh training from scratch (no checkpoint loaded)
```

### Pour Continuer un Checkpoint Existant
1. Dans `bot.py`, changez:
```python
load_checkpoint = True  # Au lieu de False
```

2. Lancez:
```bash
python bot.py
```

Vous verrez:
```
📁 Loading checkpoint: data/checkpoints/rlgym-ppo-run_XXX/YYYY
```

---

## 📊 Observation Space (AdvancedObs)

**Taille fixe: 231 dimensions**

Contenu (pour CHAQUE agent):
- Position du joueur (3)
- Vitesse linéaire (3)
- Vitesse angulaire (3)
- Matrice de rotation (9)
- Boost (1)
- Position balle (3)
- Vitesse balle (3)
- Vitesse angulaire balle (3)
- Dernière touche balle (1)
- Données coéquipiers (jusqu'à 2 max)
- Données adversaires (jusqu'à 3 max)
- États boost pads (34)
- Données temporelles et contextuelles

**Avantage**: Taille TOUJOURS 231, peu importe le mode!

---

## 🎯 Conseils par Mode

### 1v0
- Récompenses: Focus sur touches et buts
- Durée: Ne passez PAS trop de temps (bot devient "paresseux")
- Objectif: Dès que >30 touches/partie → passez en 1v1

### 1v1
- Ajustez rewards: Augmentez pénalité concede
- Le bot va sembler "pire" au début (normal!)
- Surveillez win rate (objectif >40%)

### 2v2
- Ajoutez reward pour passes
- Pénalisez "ball chasing" (tout le monde sur la balle)
- Formation défensive importante

### 3v3
- Rotations = clé du succès
- Reward pour positioning
- Très long à maîtriser (plusieurs semaines)

---

## ⚙️ Configuration Actuelle

```python
# Dans bot.py
spawn_opponents = False  # Mode 1v0
team_size = 1           # 1 joueur par équipe
load_checkpoint = False # Démarrage frais
```

**Prochaine étape**: Quand le bot touche bien la balle:
1. Changez `spawn_opponents = True`
2. Changez `load_checkpoint = True`
3. Relancez l'entraînement!

---

## 🚀 Commandes Rapides

### Voir les statistiques
```bash
# WandB dashboard - ouvrez dans votre navigateur
```

### Tester le bot
```bash
# Mettez render=True dans bot.py
# Ouvrez RocketSimVis pour visualiser
```

### Backup checkpoint important
```bash
# PowerShell
Copy-Item "data/checkpoints/rlgym-ppo-run_XXX" "backups/checkpoint_1v0_good" -Recurse
```

---

## ❓ FAQ

**Q: Puis-je passer directement de 1v0 à 3v3?**
R: Oui techniquement, mais déconseillé. Le bot sera perdu. Progression graduelle recommandée.

**Q: Combien de temps par phase?**
R: 
- 1v0: 1-2 jours
- 1v1: 3-5 jours  
- 2v2: 1 semaine
- 3v3: 2-4 semaines

**Q: Le checkpoint 1v0 marche en 2v2?**
R: OUI! Grâce à AdvancedObs (231-dim constant). Mais le bot devra s'adapter.

**Q: Je veux recommencer de zéro?**
R: Mettez `load_checkpoint = False` et lancez.

**Q: Mon ancien checkpoint DefaultObs marche encore?**
R: NON. DefaultObs = 70-dim, AdvancedObs = 231-dim. Incompatibles. Repartez de zéro.

---

Bon entraînement! 🚗⚽
