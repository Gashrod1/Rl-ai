"""
Script pour transférer un checkpoint 1v0 vers le self-play (1v1)
Charge les poids existants et adapte la première couche pour 2 joueurs
"""
import torch
import os
import shutil

def transfer_checkpoint_to_selfplay(checkpoint_path, output_path):
    """
    Transfère un checkpoint 1v0 (obs_size=70) vers 1v1 (obs_size=89)
    
    Args:
        checkpoint_path: Chemin vers le checkpoint 1v0 (ex: data/checkpoints/run/403396680)
        output_path: Chemin où sauvegarder le nouveau checkpoint adapté
    """
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Charger les fichiers du checkpoint (noms utilisés par rlgym-ppo)
    policy_path = os.path.join(checkpoint_path, "PPO_POLICY.pt")
    critic_path = os.path.join(checkpoint_path, "PPO_VALUE_NET.pt")
    
    if not os.path.exists(policy_path) or not os.path.exists(critic_path):
        raise FileNotFoundError(f"Checkpoint files not found in {checkpoint_path}")
    
    # Charger les state dicts
    policy_state = torch.load(policy_path, map_location='cpu')
    critic_state = torch.load(critic_path, map_location='cpu')
    
    print("Original policy input layer shape:", policy_state['model.0.weight'].shape)
    print("Original critic input layer shape:", critic_state['model.0.weight'].shape)
    
    # Anciennes dimensions (1 joueur)
    old_obs_size = 70  # DefaultObs pour 1 joueur
    new_obs_size = 89  # DefaultObs pour 2 joueurs (1v1)
    
    # POLICY: Adapter la première couche
    old_policy_weight = policy_state['model.0.weight']  # Shape: [512, 70]
    old_policy_bias = policy_state['model.0.bias']      # Shape: [512]
    
    # Créer une nouvelle première couche avec initialisation Xavier
    new_policy_weight = torch.zeros(old_policy_weight.shape[0], new_obs_size)
    torch.nn.init.xavier_uniform_(new_policy_weight)
    
    # Copier les poids des 70 premières features (info du joueur principal)
    new_policy_weight[:, :old_obs_size] = old_policy_weight
    
    # Les 19 nouvelles features (info de l'adversaire) sont initialisées aléatoirement
    # mais avec une petite échelle pour ne pas perturber le modèle
    new_policy_weight[:, old_obs_size:] *= 0.1
    
    policy_state['model.0.weight'] = new_policy_weight
    # Le biais reste le même
    
    # CRITIC: Même chose
    old_critic_weight = critic_state['model.0.weight']
    old_critic_bias = critic_state['model.0.bias']
    
    new_critic_weight = torch.zeros(old_critic_weight.shape[0], new_obs_size)
    torch.nn.init.xavier_uniform_(new_critic_weight)
    new_critic_weight[:, :old_obs_size] = old_critic_weight
    new_critic_weight[:, old_obs_size:] *= 0.1
    
    critic_state['model.0.weight'] = new_critic_weight
    
    print("New policy input layer shape:", policy_state['model.0.weight'].shape)
    print("New critic input layer shape:", critic_state['model.0.weight'].shape)
    
    # Créer le dossier de sortie
    os.makedirs(output_path, exist_ok=True)
    
    # Sauvegarder les nouveaux state dicts avec les bons noms
    torch.save(policy_state, os.path.join(output_path, "PPO_POLICY.pt"))
    torch.save(critic_state, os.path.join(output_path, "PPO_VALUE_NET.pt"))
    
    # Copier les autres fichiers du checkpoint s'ils existent
    for filename in os.listdir(checkpoint_path):
        if filename not in ['PPO_POLICY.pt', 'PPO_VALUE_NET.pt']:
            src = os.path.join(checkpoint_path, filename)
            dst = os.path.join(output_path, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
    
    print(f"\n✅ Checkpoint adapté sauvegardé dans {output_path}")
    print("Vous pouvez maintenant l'utiliser pour l'entraînement en self-play !")
    return output_path


if __name__ == "__main__":
    # Trouver le dernier checkpoint automatiquement
    checkpoint_base = os.path.join("data", "checkpoints")
    
    if os.path.isdir(checkpoint_base):
        run_dirs = [d for d in os.listdir(checkpoint_base) 
                   if d.startswith("rlgym-ppo-run") and os.path.isdir(os.path.join(checkpoint_base, d))]
        
        if run_dirs:
            latest_run = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(checkpoint_base, d)))
            latest_run_dir = os.path.join(checkpoint_base, latest_run)
            
            checkpoint_subdirs = [d for d in os.listdir(latest_run_dir) 
                                 if d.isdigit() and os.path.isdir(os.path.join(latest_run_dir, d))]
            
            if checkpoint_subdirs:
                latest_checkpoint = max(checkpoint_subdirs, key=int)
                checkpoint_path = os.path.join(latest_run_dir, latest_checkpoint)
                
                print(f"📁 Checkpoint trouvé: {checkpoint_path}")
                print(f"📊 Steps: {latest_checkpoint}")
                
                # Créer un nouveau dossier pour le checkpoint adapté
                output_path = os.path.join(checkpoint_base, "selfplay_transfer", latest_checkpoint)
                
                transfer_checkpoint_to_selfplay(checkpoint_path, output_path)
                
                print(f"\n📝 Pour utiliser ce checkpoint:")
                print(f"   Modifiez bot.py ligne ~165 pour pointer vers:")
                print(f'   latest_checkpoint_dir = r"{output_path}"')
            else:
                print("❌ Aucun checkpoint trouvé dans le run directory")
        else:
            print("❌ Aucun run directory trouvé")
    else:
        print(f"❌ Le dossier {checkpoint_base} n'existe pas")
