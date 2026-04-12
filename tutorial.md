# MARL Formation Control & Collision Avoidance — Projektplan
 
## Ziel
Baue ein Multi-Agent Reinforcement Learning System, in dem 5 UAV-Agenten lernen:
1. Eine Dreiecks-Formation (V-Formation) zu halten
2. Gemeinsam in eine Zielrichtung zu fliegen
3. Kollisionen untereinander zu vermeiden
4. Dynamischen Hindernissen auszuweichen
 
## Architektur: CTDE (Centralized Training, Decentralized Execution)
- **Training:** Zentralisierter Critic mit globalem State (Positionen aller Agenten)
- **Execution:** Jeder Agent hat einen eigenen Actor, der nur lokale Beobachtungen nutzt
- **Algorithmus:** MAPPO (Multi-Agent Proximal Policy Optimization)
 
---
 
## Schritt 1: Custom PettingZoo Environment
 
Erstelle ein Custom Parallel Environment (`ParallelEnv`) mit folgenden Specs:
 
### Welt
- 2D-Raum, kontinuierlich, Größe 100x100
- 5 Agenten (UAVs), dargestellt als Punkte mit Radius 1.0
- 2-3 dynamische Hindernisse, die sich zufällig linear bewegen
- Zielrichtung: Alle Agenten sollen sich gemeinsam nach rechts (+x) bewegen
 
### Observation Space (pro Agent, nur lokal)
- Eigene Position (x, y)
- Eigene Geschwindigkeit (vx, vy)
- Relative Positionen der N nächsten Nachbarn (z.B. 2-3 Nachbarn)
- Relative Positionen sichtbarer Hindernisse (innerhalb Sensorradius)
- Abweichung von der Soll-Position in der Formation
- **Shape:** Box, continuous, ca. 14-20 Werte
 
### Action Space (pro Agent)
- Continuous Box: (ax, ay) — Beschleunigung in x und y
- Bereich: [-1, 1] für beide Achsen
- Max-Geschwindigkeit begrenzen (z.B. 2.0 pro Step)
 
### Soll-Formation
- V-Formation mit definiertem Leader (Agent 0 vorne)
- Relative Soll-Positionen der Follower zum Formationsschwerpunkt
- Formation darf rotieren, aber Abstände müssen stimmen
 
### Reward Function (pro Agent, pro Step)
```
reward = (
    w1 * formation_reward      # Belohnung für Nähe zur Soll-Position in Formation
  + w2 * direction_reward       # Belohnung für Bewegung in Zielrichtung (+x)
  + w3 * collision_penalty      # Strafe bei Kollision mit anderem Agent
  + w4 * obstacle_penalty       # Strafe bei Kollision mit Hindernis
  + w5 * smoothness_reward      # Belohnung für sanfte Aktionen (wenig Ruck)
)
```
Vorgeschlagene Gewichte: w1=1.0, w2=0.5, w3=-5.0, w4=-5.0, w5=0.1
 
### Episode
- Max 500 Steps pro Episode
- Terminated wenn: Kollision (optional: Episode weiter, nur Strafe)
- Truncated nach 500 Steps
 
### Physik (simpel)
- Euler-Integration: v_new = v_old + a * dt, pos_new = pos_old + v * dt
- dt = 0.1
- Optional: leichtes Rauschen auf Aktionen (simuliert Wind)
 
---
 
## Schritt 2: MAPPO Implementierung
 
### Option A: EPyMARL (empfohlen für schnellen Start)
- Installiere EPyMARL: `git clone https://github.com/uoe-agents/epymarl`
- Wrape das Custom Environment für EPyMARL-Kompatibilität
- Nutze den eingebauten MAPPO-Algorithmus
- Config anpassen: lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, epochs=10
 
### Option B: Eigene Implementierung (mehr Kontrolle, lehrreicher)
- Actor-Netzwerk: MLP [obs_dim → 128 → 128 → action_dim], Tanh output
- Critic-Netzwerk: MLP [global_state_dim → 128 → 128 → 1]
- Global State für Critic = Konkatenation aller Agenten-Observations
- PPO mit Clipping, GAE für Advantage Estimation
- Shared Parameters zwischen allen Agents (Parameter Sharing)
- Training: ~2000 Episodes, Batch Size 32, 10 PPO-Epochs pro Update
 
### Wichtige Details
- **Parameter Sharing:** Alle Agenten teilen sich die gleichen Netzwerk-Gewichte (sie sind homogen)
- **Action Masking:** Optional — maskiere Aktionen die zur Kollision führen würden
- **Reward Normalization:** Running Mean/Std für stabileres Training
 
---
 
## Schritt 3: Training & Evaluation
 
### Training Loop
```
for episode in range(num_episodes):
    obs = env.reset()
    for step in range(max_steps):
        actions = {agent: actor.get_action(obs[agent]) for agent in agents}
        next_obs, rewards, dones, infos = env.step(actions)
        buffer.store(obs, actions, rewards, dones)
        obs = next_obs
    # PPO Update nach jeder Episode (oder nach N Steps)
    actor.update(buffer)
    critic.update(buffer)
```
 
### Metriken zum Tracken
- **Formation Error:** Mittlerer Abstand aller Agenten von ihrer Soll-Position
- **Collision Rate:** Anzahl Kollisionen pro Episode
- **Distance Traveled:** Wie weit der Schwarm in +x gekommen ist
- **Reward Curve:** Gesamtreward pro Episode über Training
 
### Evaluation
- Teste trainierte Policy über 100 Episodes ohne Exploration
- Visualisiere Trajektorien aller Agenten in 2D Plot
- Zeige Formation-Error über die Zeit
 
---
 
## Schritt 4: Visualisierung
 
Baue eine einfache Pygame oder Matplotlib-Visualisierung:
- Agenten als farbige Punkte
- Soll-Formation als gestrichelte Linien/Kreise
- Hindernisse als rote Kreise
- Trajektorien als Spuren hinter den Agenten
- Live-Anzeige: Formation Error, Step Counter, Total Reward
 
---
 
## Projektstruktur
```
uav-swarm-marl/
├── env/
│   ├── __init__.py
│   ├── formation_env.py        # Custom PettingZoo Environment
│   └── formation_config.py     # Hyperparameter & Formation-Definition
├── algo/
│   ├── __init__.py
│   ├── mappo.py                # MAPPO Algorithmus
│   ├── actor.py                # Actor-Netzwerk
│   ├── critic.py               # Critic-Netzwerk
│   └── buffer.py               # Replay/Rollout Buffer
├── train.py                    # Training Script
├── evaluate.py                 # Evaluation & Metriken
├── visualize.py                # Pygame/Matplotlib Visualisierung
├── requirements.txt            # Dependencies
└── README.md
```
 
## Dependencies
```
torch>=2.0
pettingzoo>=1.24
gymnasium
numpy
matplotlib
pygame          # für Live-Visualisierung
tensorboard     # für Training-Logs
```
 
---
 
## Hinweise für die Implementierung
- Fang OHNE Hindernisse an, füge sie erst hinzu wenn Formation + Collision Avoidance funktioniert (Curriculum Learning)
- Teste zuerst mit 3 Agenten, skaliere dann auf 5
- Wenn Training nicht konvergiert: Reward-Gewichte anpassen, vor allem Collision Penalty nicht zu hoch (sonst lernen Agenten nur stillzustehen)
- Logge alles mit Tensorboard
 



