# Guide capture vidéo plan B — démo soutenance

## Quand

À J-3 minimum (15 mai 2026 ou avant).
Refaire à J-1 si du code a changé entre temps.

## Outil recommandé

- OBS Studio (gratuit, qualité pro) ou ShareX (Windows simple)
- Résolution : 1920x1080 minimum
- FPS : 30 (suffisant pour un dashboard, économise la taille)
- Format : MP4 (compatible partout, facile à lire en soutenance)

## Pré-vérifications avant capture

1. Run `python scripts/sanity_check_full.py` — DOIT être 9/9 OK
2. `docker compose ps` — tous les services healthy
3. Aucune fenêtre parasite, mode plein écran
4. Désactiver les notifications système
5. Curseur visible (option OBS)

## Script de la capture

Suivre exactement `DEMO_PLAN.md`, acte par acte.
Durée cible : 8-10 minutes.

- Acte 1 (1 min) : architecture
- Acte 2 (2 min) : onglet 1 — montrer 3 nœuds, MD5, click "+ Connecter"
- Acte 3 (2 min) : onglet 2 — KPI, scheduler, click "Lancer training mock"
- Acte 4 (2 min) : onglet 3 — table 8x6, radar, status pills
- Acte 5 (1 min) : onglet 4 monitoring + sélecteur scénario

## Voix off (optionnel)

Si tu enregistres ta voix : parle clairement, pas trop vite, en français.
Énonce les hashes MD5 oralement pour souligner la preuve cryptographique.

## Stockage

- Sauvegarde locale : `~/dev/qi-fl-ids-iot/demo_video.mp4`
- Backup clé USB
- Backup Google Drive personnel (au cas où)
- Ne PAS committer dans git (>> 50 MB)

## Plan B du plan B

Si le jour de la soutenance :

- Docker plante : lancer la vidéo
- Vidéo plante : montrer `DEMO_PLAN.md` + screenshots
- Tout plante : raconter la démo en mots avec les screenshots dans tes slides
