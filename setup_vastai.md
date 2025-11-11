# Guide: Transférer votre bot Rocket League sur Vast.ai

## 1. Préparer les fichiers à transférer

Créez un dossier avec uniquement les fichiers nécessaires :
- bot.py
- rewards.py
- collision_meshes/ (tout le dossier)
- data/checkpoints/rlgym-ppo-run-XXXXX/65M/ (votre dernier checkpoint)

NE PAS transférer :
- wandb/ (sera recréé)
- __pycache__/ (sera recréé)

## 2. S'inscrire sur Vast.ai

1. Allez sur https://vast.ai
2. Créez un compte
3. Ajoutez 15€ de crédit (Billing → Add Credit)

## 3. Louer une machine

1. Cliquez sur "Search" dans le menu
2. Filtres recommandés :
   - GPU: RTX 3060 ou RTX 3070
   - VRAM: >= 8 GB
   - Disk Space: >= 20 GB
   - Prix: <= 0.30 $/h

3. Template: "pytorch/pytorch:latest" ou "python:3.11"

4. Cliquez sur "RENT" sur une machine pas chère

## 4. Se connecter à la machine

Une fois la machine lancée :
1. Cliquez sur "CONNECT"
2. Copiez la commande SSH (ressemble à: ssh root@ssh4.vast.ai -p 12345)
3. Dans votre terminal PowerShell, connectez-vous

## 5. Installer les dépendances

```bash
pip install rlgym-ppo rlgym-sim rocketsim wandb numpy torch
```

## 6. Transférer vos fichiers

Option A - Via SCP (depuis votre PC) :
```powershell
# Adapter le port et l'adresse de votre machine Vast.ai
scp -P 12345 -r "e:\rl AI\bot.py" root@ssh4.vast.ai:/workspace/
scp -P 12345 -r "e:\rl AI\rewards.py" root@ssh4.vast.ai:/workspace/
scp -P 12345 -r "e:\rl AI\collision_meshes" root@ssh4.vast.ai:/workspace/
scp -P 12345 -r "e:\rl AI\data" root@ssh4.vast.ai:/workspace/
```

Option B - Via GitHub (plus simple) :
```bash
# Sur la machine Vast.ai
cd /workspace
git clone https://github.com/Gashrod1/Rl-ai.git
cd Rl-ai
```

## 7. Configurer WandB

```bash
wandb login
# Collez votre clé API WandB
```

## 8. Modifier bot.py pour GPU

Le bot.py utilisera automatiquement CUDA si disponible.
Vérifiez avec:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 9. Lancer l'entraînement

```bash
# Désactiver le render (pas de X11 sur serveur)
python bot.py
```

## 10. Monitoring

- Surveillez sur WandB: https://wandb.ai
- Pour voir les logs en temps réel: laissez le terminal ouvert
- Pour détacher la session: utilisez `screen` ou `tmux`

```bash
# Avec screen (recommandé)
screen -S training
python bot.py
# Appuyez sur Ctrl+A puis D pour détacher
# Pour revenir: screen -r training
```

## 11. Récupérer le modèle entraîné

Quand vous voulez arrêter :
```powershell
# Depuis votre PC
scp -P 12345 -r root@ssh4.vast.ai:/workspace/data/checkpoints ./
```

## 💰 Coûts estimés (RTX 3060 à 0.15€/h)

- 24h = 3.6€ → ~300M steps supplémentaires
- 48h = 7.2€ → ~600M steps supplémentaires
- 72h = 10.8€ → ~900M steps supplémentaires

## ⚠️ IMPORTANT

- N'oubliez pas de DESTROY la machine quand vous avez fini !
- Vast.ai facture à l'heure, même si vous n'utilisez pas
- Sauvegardez régulièrement vos checkpoints
