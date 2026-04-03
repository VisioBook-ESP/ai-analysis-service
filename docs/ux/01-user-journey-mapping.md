# User Journey Mapping - VisioBook Mobile

## Vue d'ensemble

Ce document presente les parcours utilisateur (User Journeys) pour l'application mobile VisioBook. Chaque journey est decrit selon les phases, points de contact, emotions et opportunites d'amelioration.

## Journey 1: Nouvel Utilisateur - Premiere Experience

### Scenario
Marie, enseignante, decouvre VisioBook et souhaite creer sa premiere animation a partir d'un livre pour ses eleves.

### Journey Map

```mermaid
journey
    title Parcours Nouvel Utilisateur - Premiere VisioBook
    section Decouverte
      Telecharger l'app: 5: Marie
      Ouvrir l'app: 5: Marie
      Voir le splash screen: 4: Marie
    section Onboarding
      Parcourir les slides: 4: Marie
      Comprendre le concept: 5: Marie
      Cliquer sur Commencer: 5: Marie
    section Inscription
      Voir l'ecran login: 3: Marie
      Choisir inscription: 4: Marie
      Remplir le formulaire: 3: Marie
      Valider l'email: 3: Marie
      Se connecter: 4: Marie
    section Premiere utilisation
      Arriver sur Dashboard: 5: Marie
      Cliquer sur Scanner: 5: Marie
      Choisir un fichier PDF: 4: Marie
      Attendre l'analyse: 3: Marie
      Voir le resume: 5: Marie
    section Generation
      Configurer le style: 4: Marie
      Lancer la generation: 4: Marie
      Attendre la generation: 2: Marie
      Voir le resultat: 5: Marie
    section Partage
      Regarder la video: 5: Marie
      Partager par lien: 5: Marie
```

### Detail par phase

#### Phase 1: Decouverte (Pre-app)

| Etape | Point de contact | Emotion | Action | Opportunite |
|-------|-----------------|---------|--------|-------------|
| Decouverte | Store (App Store/Play Store) | Curieuse | Recherche "livre animation" | SEO, mots-cles pertinents |
| Evaluation | Page store | Interessee | Lit les avis et description | Screenshots attractifs |
| Telechargement | Bouton installer | Enthousiaste | Telecharge l'app | Taille d'app reduite |

#### Phase 2: Onboarding

| Etape | Ecran | Emotion | Action | Opportunite |
|-------|-------|---------|--------|-------------|
| Splash | Splash screen | Neutre | Attend | Animation engageante < 2s |
| Slide 1 | Onboarding | Curieuse | Swipe | Message clair sur la valeur |
| Slide 2 | Onboarding | Interessee | Swipe | Demo visuelle du resultat |
| Slide 3 | Onboarding | Motivee | Tap "Commencer" | CTA visible et attractif |

#### Phase 3: Inscription

| Etape | Ecran | Emotion | Action | Pain Point | Solution |
|-------|-------|---------|--------|------------|----------|
| Choix methode | Login | Hesitante | Choisit email | Trop d'options | Social login en priorite |
| Formulaire | Register | Frustrée | Remplit champs | Formulaire long | Minimum de champs |
| Validation | Email | Impatiente | Verifie email | Delai reception | Lien magique ou code |
| Connexion | Login | Soulagee | Se connecte | - | Connexion auto apres validation |

#### Phase 4: Premiere utilisation

| Etape | Ecran | Emotion | Action | Pain Point | Solution |
|-------|-------|---------|--------|------------|----------|
| Dashboard vide | Dashboard | Perdue | Cherche quoi faire | Ecran vide intimidant | CTA prominent + guide |
| Import | Input/Scanner | Curieuse | Choisit fichier | Format non supporte | Liste claire des formats |
| Upload | Input/Scanner | Impatiente | Attend upload | Temps d'attente | Progress bar + estimation |
| Analyse | Detail | Interessee | Voit le resume | Resume incorrect | Option d'edition |

#### Phase 5: Generation

| Etape | Ecran | Emotion | Action | Pain Point | Solution |
|-------|-------|---------|--------|------------|----------|
| Configuration | Detail | Excitee | Choisit style | Trop d'options | Defaults intelligents |
| Lancement | Detail | Enthousiaste | Clique generer | Cout/credits | Premier gratuit |
| Attente | Detail | Anxieuse | Attend | Temps long | Notifications + temps estime |
| Resultat | Player | Emerveillée | Voit le resultat | - | Celebration UI |

#### Phase 6: Partage

| Etape | Ecran | Emotion | Action | Pain Point | Solution |
|-------|-------|---------|--------|------------|----------|
| Visionnage | Player | Ravie | Regarde | Qualite video | Options qualite |
| Partage | Player | Fiere | Partage lien | Processus complexe | Share sheet native |
| Feedback | - | Satisfaite | Recommande | - | Demande d'avis |

---

## Journey 2: Utilisateur Regulier - Creation rapide

### Scenario
Thomas utilise VisioBook regulierement pour creer des histoires pour ses enfants avant le coucher.

### Journey Map

```mermaid
journey
    title Parcours Utilisateur Regulier - Usage quotidien
    section Retour dans l'app
      Ouvrir l'app: 5: Thomas
      Login automatique: 5: Thomas
      Dashboard avec historique: 5: Thomas
    section Nouveau projet
      Cliquer Scanner: 5: Thomas
      Scanner une page livre: 4: Thomas
      OCR rapide: 4: Thomas
      Apercu du texte: 5: Thomas
    section Configuration rapide
      Style favori pre-selectionne: 5: Thomas
      Ajuster duree: 4: Thomas
      Valider generation: 5: Thomas
    section Attente active
      Notification push: 4: Thomas
      Continuer autre tache: 5: Thomas
      Recevoir notification: 5: Thomas
    section Utilisation
      Ouvrir le resultat: 5: Thomas
      Lire avec les enfants: 5: Thomas
      Sauvegarder en favori: 4: Thomas
```

### Points cles pour l'utilisateur regulier

```yaml
Attentes:
  - Login automatique (biometrie)
  - Acces rapide au scanner
  - Preferences memorisees
  - Historique organise
  - Notifications fiables

Optimisations UX:
  - Shortcut direct vers scanner depuis home screen
  - Templates de style favoris
  - Generation en arriere-plan
  - Hors-ligne pour consultation historique
```

### Metriques cibles

| Metrique | Objectif | Mesure |
|----------|----------|--------|
| Time-to-scan | < 10 secondes | Du lancement au debut scan |
| Time-to-generate | < 30 secondes | De la validation au lancement IA |
| Retention J7 | > 40% | Utilisateurs revenant a J+7 |
| Sessions/semaine | > 3 | Moyenne par utilisateur actif |

---

## Journey 3: Utilisateur Premium - Usage professionnel

### Scenario
Sophie, auteure, utilise VisioBook pour creer des teasers visuels de ses romans pour les reseaux sociaux.

### Journey Map

```mermaid
journey
    title Parcours Utilisateur Premium - Usage professionnel
    section Connexion Pro
      Ouvrir l'app: 5: Sophie
      Badge Premium visible: 5: Sophie
      Options avancees accessibles: 5: Sophie
    section Import pro
      Importer fichier complet: 4: Sophie
      Selectionner passages: 5: Sophie
      Previsualiser chapitres: 5: Sophie
    section Personnalisation avancee
      Choisir style artistique: 5: Sophie
      Ajuster parametres fins: 4: Sophie
      Personnaliser personnages: 4: Sophie
      Choisir voix narrateur: 5: Sophie
    section Generation HD
      Lancer en haute qualite: 4: Sophie
      Suivre progression detaillee: 4: Sophie
      Preview intermediaires: 5: Sophie
    section Export pro
      Telecharger sans filigrane: 5: Sophie
      Choisir format export: 5: Sophie
      Integrer a workflow: 4: Sophie
```

### Fonctionnalites Premium differenciantes

| Fonctionnalite | Standard | Premium |
|----------------|----------|---------|
| Resolution export | 720p | 4K |
| Filigrane | Oui | Non |
| Styles disponibles | 5 | 20+ |
| Personnalisation personnages | Non | Oui |
| Voix narrateur | 2 | 10+ |
| Generations/mois | 5 | Illimite |
| Support | Email | Prioritaire |

---

## Journey 4: Parcours d'erreur - Gestion des echecs

### Scenario
Un utilisateur rencontre une erreur pendant la generation.

### Journey Map

```mermaid
journey
    title Parcours d'erreur - Generation echouee
    section Detection
      Generation en cours: 3: User
      Erreur detectee: 1: User
      Message d'erreur affiche: 2: User
    section Comprehension
      Lire le message: 2: User
      Comprendre la cause: 3: User
      Voir les options: 4: User
    section Resolution
      Choisir reessayer: 4: User
      Ajuster parametres: 4: User
      Relancer generation: 4: User
    section Succes
      Generation reussie: 5: User
      Confiance restauree: 4: User
```

### Matrice des erreurs et solutions UX

| Type d'erreur | Message utilisateur | Action proposee | Prevention |
|---------------|---------------------|-----------------|------------|
| Timeout generation | "La generation prend plus de temps que prevu" | Reessayer / Notifier quand pret | Estimation realiste du temps |
| Contenu trop long | "Le texte depasse la limite" | Raccourcir / Decoouper | Avertir avant upload |
| Format non supporte | "Ce format n'est pas pris en charge" | Voir formats acceptes | Liste visible a l'import |
| Quota atteint | "Vous avez utilise vos credits" | Passer Premium / Attendre | Compteur visible |
| Erreur serveur | "Un probleme est survenu" | Reessayer plus tard | Monitoring proactif |
| Connexion perdue | "Connexion internet perdue" | Reconnecter | Mode offline partiel |

---

## Journey 5: Parcours complet MVP - De A a Z

### Diagramme du parcours complet

```mermaid
flowchart TD
    subgraph "1. ONBOARDING"
        A[Splash Screen] --> B{Premier lancement?}
        B -->|Oui| C[Onboarding Slides]
        B -->|Non| D[Login Auto]
        C --> E[Ecran Login/Register]
        E --> F{Compte existant?}
        F -->|Non| G[Inscription]
        F -->|Oui| H[Connexion]
        G --> I[Validation Email]
        I --> J[Dashboard]
        H --> J
        D --> J
    end

    subgraph "2. IMPORT CONTENU"
        J --> K[Dashboard]
        K --> L[Tab Scanner]
        L --> M{Mode import?}
        M -->|Fichier| N[File Picker]
        M -->|Camera| O[Scanner OCR]
        N --> P[Upload Progress]
        O --> Q[Capture + OCR]
        P --> R[Analyse IA]
        Q --> R
        R --> S[Preview Resume]
    end

    subgraph "3. CONFIGURATION"
        S --> T[Detail Projet]
        T --> U[Choisir Style]
        U --> V[Configurer Audio]
        V --> W{Options avancees?}
        W -->|Oui| X[Personnalisation]
        W -->|Non| Y[Valider Config]
        X --> Y
    end

    subgraph "4. GENERATION"
        Y --> Z[Lancer Generation]
        Z --> AA[Progress Screen]
        AA --> AB{Generation OK?}
        AB -->|Non| AC[Ecran Erreur]
        AC --> AD{Reessayer?}
        AD -->|Oui| Z
        AD -->|Non| K
        AB -->|Oui| AE[Generation Complete]
    end

    subgraph "5. VISUALISATION"
        AE --> AF[Player VisioBook]
        AF --> AG[Controles Lecture]
        AG --> AH{Satisfait?}
        AH -->|Non| AI[Regenerer]
        AI --> T
        AH -->|Oui| AJ[Continuer]
    end

    subgraph "6. EXPORT/PARTAGE"
        AJ --> AK{Action?}
        AK -->|Telecharger| AL[Download Video]
        AK -->|Partager| AM[Share Sheet]
        AK -->|Retour| K
        AL --> AN[Fichier sauvegarde]
        AM --> AO[Lien partage]
        AN --> K
        AO --> K
    end

    classDef onboard fill:#e3f2fd,stroke:#1976d2
    classDef import fill:#fff3e0,stroke:#f57c00
    classDef config fill:#e8f5e9,stroke:#388e3c
    classDef generate fill:#fce4ec,stroke:#c2185b
    classDef view fill:#f3e5f5,stroke:#7b1fa2
    classDef export fill:#e0f2f1,stroke:#00695c

    class A,B,C,D,E,F,G,H,I,J onboard
    class K,L,M,N,O,P,Q,R,S import
    class T,U,V,W,X,Y config
    class Z,AA,AB,AC,AD,AE generate
    class AF,AG,AH,AI,AJ view
    class AK,AL,AM,AN,AO export
```

---

## Metriques et KPIs par Journey

### Funnel de conversion

```mermaid
graph TD
    subgraph "Funnel Nouvel Utilisateur"
        F1[100% - Telechargement] --> F2[80% - Onboarding complete]
        F2 --> F3[60% - Inscription]
        F3 --> F4[40% - Premier projet cree]
        F4 --> F5[25% - Premiere generation]
        F5 --> F6[15% - Partage]
    end
```

### KPIs par phase

| Phase | KPI | Objectif MVP | Objectif v2 |
|-------|-----|--------------|-------------|
| Onboarding | Taux completion | > 80% | > 90% |
| Inscription | Taux conversion | > 50% | > 70% |
| Premier projet | Time-to-first-project | < 5 min | < 3 min |
| Generation | Taux de succes | > 95% | > 99% |
| Retention | J1 | > 30% | > 50% |
| Retention | J7 | > 15% | > 30% |
| NPS | Score | > 30 | > 50 |

---

## Recommandations UX par Journey

### Nouvel Utilisateur

1. **Onboarding gamifie** : Montrer un exemple de resultat des le debut
2. **Zero friction signup** : Social login + email magic link
3. **Premier projet guide** : Wizard step-by-step avec exemple
4. **Celebration du succes** : Animation de felicitation au premier VisioBook

### Utilisateur Regulier

1. **Quick actions** : Raccourcis vers les actions frequentes
2. **Smart defaults** : Memoriser les preferences de style
3. **Background processing** : Generation sans bloquer l'UI
4. **Notifications intelligentes** : Pas de spam, que l'essentiel

### Utilisateur Premium

1. **Onboarding premium** : Tour des fonctionnalites avancees
2. **Export pro** : Formats professionnels (ProRes, etc.)
3. **Analytics** : Stats sur ses creations
4. **Priority queue** : Generation prioritaire garantie
