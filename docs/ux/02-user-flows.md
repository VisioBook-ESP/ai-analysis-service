# User Flows - VisioBook Mobile

## Vue d'ensemble

Ce document presente les flux utilisateur (User Flows) detailles de l'application mobile VisioBook. Chaque flow correspond a une fonctionnalite cle et est illustre par des diagrammes Mermaid.

---

## Flow 1: Inscription et Connexion

### US Associees
- **US P5.1** - Creation de compte (MUST)
- **US 5.1** - Authentification

### Flow d'inscription

```mermaid
flowchart TD
    A[Ecran Login] --> B{Methode choisie}

    B -->|Email| C[Formulaire Email]
    B -->|Google| D[OAuth Google]
    B -->|Apple| E[Sign in with Apple]

    C --> F[Saisir email]
    F --> G[Saisir mot de passe]
    G --> H[Confirmer mot de passe]
    H --> I{Validation OK?}

    I -->|Non| J[Afficher erreurs]
    J --> F

    I -->|Oui| K[Envoyer verification]
    K --> L[Ecran verification email]
    L --> M{Email verifie?}

    M -->|Non| N[Renvoyer email]
    N --> L

    M -->|Oui| O[Compte active]

    D --> P{Auth Google OK?}
    P -->|Oui| O
    P -->|Non| Q[Erreur OAuth]
    Q --> B

    E --> R{Auth Apple OK?}
    R -->|Oui| O
    R -->|Non| S[Erreur Apple]
    S --> B

    O --> T[Redirection Dashboard]

    style A fill:#e3f2fd
    style T fill:#e8f5e9
    style J fill:#ffebee
    style Q fill:#ffebee
    style S fill:#ffebee
```

### Flow de connexion

```mermaid
flowchart TD
    A[Ecran Login] --> B{Connexion automatique?}

    B -->|Oui - Biometrie| C[Face ID / Touch ID]
    B -->|Non| D{Methode choisie}

    C --> E{Biometrie OK?}
    E -->|Oui| F[Token refresh]
    E -->|Non| D

    D -->|Email| G[Saisir email]
    D -->|Google| H[OAuth Google]
    D -->|Apple| I[Sign in with Apple]

    G --> J[Saisir mot de passe]
    J --> K{Credentials OK?}

    K -->|Non| L[Erreur: identifiants invalides]
    L --> M{Mot de passe oublie?}
    M -->|Oui| N[Flow reset password]
    M -->|Non| G

    K -->|Oui| F

    H --> O{OAuth OK?}
    O -->|Oui| F
    O -->|Non| P[Erreur OAuth]
    P --> D

    I --> Q{Apple OK?}
    Q -->|Oui| F
    Q -->|Non| R[Erreur Apple]
    R --> D

    F --> S{Token valide?}
    S -->|Oui| T[Dashboard]
    S -->|Non| U[Erreur session]
    U --> D

    N --> V[Saisir email]
    V --> W[Envoyer lien reset]
    W --> X[Email envoye]
    X --> Y[Nouveau mot de passe]
    Y --> G

    style A fill:#e3f2fd
    style T fill:#e8f5e9
    style L fill:#ffebee
    style P fill:#ffebee
    style R fill:#ffebee
```

### Validation des champs

| Champ | Regles | Message d'erreur |
|-------|--------|------------------|
| Email | Format email valide | "Adresse email invalide" |
| Password | Min 8 chars, 1 maj, 1 chiffre | "Le mot de passe doit contenir..." |
| Confirm | Egal au password | "Les mots de passe ne correspondent pas" |

---

## Flow 2: Import de Contenu (Fichier)

### US Associees
- **US 1.1** - Import de fichiers (MUST)
- **US 1.3** - Previsualisation et resume (SHOULD)

### Flow import fichier

```mermaid
flowchart TD
    A[Tab Scanner] --> B[Ecran Input]
    B --> C{Mode selection}

    C -->|Import Fichier| D[Ouvrir File Picker]
    C -->|Scanner| E[Flow Scanner OCR]

    D --> F{Fichier selectionne?}
    F -->|Non| G[Retour Input]
    F -->|Oui| H{Format supporte?}

    H -->|Non| I[Erreur: format non supporte]
    I --> J[Afficher formats acceptes]
    J --> D

    H -->|Oui| K{Taille OK?}
    K -->|Non| L[Erreur: fichier trop volumineux]
    L --> M[Suggestion compression]
    M --> D

    K -->|Oui| N[Debut upload]
    N --> O[Progress bar upload]
    O --> P{Upload OK?}

    P -->|Non| Q[Erreur upload]
    Q --> R{Reessayer?}
    R -->|Oui| N
    R -->|Non| G

    P -->|Oui| S[Extraction texte]
    S --> T[Progress analyse]
    T --> U{Extraction OK?}

    U -->|Non| V[Erreur extraction]
    V --> W[Proposer OCR manuel]
    W --> E

    U -->|Oui| X[Afficher preview]
    X --> Y[Resume automatique]
    Y --> Z[Ecran Detail Projet]

    style A fill:#e3f2fd
    style Z fill:#e8f5e9
    style I fill:#ffebee
    style L fill:#ffebee
    style Q fill:#ffebee
    style V fill:#ffebee
```

### Formats supportes

| Format | Extension | Taille max | Notes |
|--------|-----------|------------|-------|
| PDF | .pdf | 50 MB | Texte extractible |
| Texte | .txt | 10 MB | UTF-8 |
| Word | .docx | 25 MB | Via conversion |
| EPUB | .epub | 50 MB | Extraction chapitres |

---

## Flow 3: Import de Contenu (Scanner OCR)

### US Associees
- **US 1.2** - Scan de texte (SHOULD)

### Flow scanner OCR

```mermaid
flowchart TD
    A[Tab Scanner] --> B[Ecran Input]
    B --> C[Mode Scanner]
    C --> D{Permission camera?}

    D -->|Non| E[Demande permission]
    E --> F{Permission accordee?}
    F -->|Non| G[Message explicatif]
    G --> H[Bouton parametres]
    H --> I[Retour apres settings]
    I --> D

    F -->|Oui| J[Ouvrir camera]
    D -->|Oui| J

    J --> K[Cadrage page]
    K --> L[Detection automatique bords]
    L --> M{Page detectee?}

    M -->|Non| N[Guide cadrage manuel]
    N --> K

    M -->|Oui| O[Capture image]
    O --> P[Preview capture]
    P --> Q{Image OK?}

    Q -->|Non| R[Reprendre photo]
    R --> K

    Q -->|Oui| S{Autre page?}
    S -->|Oui| T[Ajouter page]
    T --> K

    S -->|Non| U[Lancer OCR]
    U --> V[Progress OCR]
    V --> W{OCR OK?}

    W -->|Non| X[Erreur OCR]
    X --> Y[Proposer saisie manuelle]
    Y --> Z[Editeur texte]

    W -->|Oui| AA[Afficher texte extrait]
    AA --> AB{Texte correct?}

    AB -->|Non| AC[Edition manuelle]
    AC --> AA

    AB -->|Oui| AD[Valider texte]
    AD --> AE[Ecran Detail Projet]

    Z --> AD

    style A fill:#e3f2fd
    style AE fill:#e8f5e9
    style G fill:#fff3e0
    style X fill:#ffebee
```

### Parametres OCR

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| Langues | FR, EN, ES, DE | Detection automatique |
| Resolution | 300 DPI | Minimum pour OCR fiable |
| Format sortie | Texte brut | Nettoyage automatique |
| Multi-page | Oui | Jusqu'a 50 pages |

---

## Flow 4: Configuration et Personnalisation

### US Associees
- **US 2.1** - Choix du style graphique (MUST)
- **US 2.2** - Choix de la langue audio (SHOULD)
- **US 2.3** - Definir la duree (COULD)
- **US 2.4** - Personnalisation personnages (COULD)
- **US 2.5** - Choix ambiance sonore (COULD)

### Flow configuration projet

```mermaid
flowchart TD
    A[Ecran Detail Projet] --> B[Section Configuration]

    B --> C[Style Graphique]
    C --> D{Style selectionne}
    D -->|Realiste| E[Preview realiste]
    D -->|Cartoon| F[Preview cartoon]
    D -->|Manga| G[Preview manga]
    D -->|Aquarelle| H[Preview aquarelle]
    D -->|Custom| I[Galerie styles]
    I --> J[Selection style custom]
    J --> K[Preview custom]

    E & F & G & H & K --> L[Valider style]

    L --> M[Langue Audio]
    M --> N{Langue selectionnee}
    N -->|Francais| O[Voix FR]
    N -->|Anglais| P[Voix EN]
    N -->|Espagnol| Q[Voix ES]
    N -->|Autre| R[Liste langues]
    R --> S[Selection langue]

    O & P & Q & S --> T[Preview voix]
    T --> U[Valider langue]

    U --> V[Duree Animation]
    V --> W{Duree choisie}
    W -->|Courte| X[2-5 min]
    W -->|Moyenne| Y[5-10 min]
    W -->|Longue| Z[10-20 min]
    W -->|Auto| AA[Calcul IA]

    X & Y & Z & AA --> AB[Valider duree]

    AB --> AC{Options avancees?}
    AC -->|Non| AD[Resume config]
    AC -->|Oui| AE[Personnalisation]

    AE --> AF[Personnages]
    AF --> AG[Modifier apparence]
    AG --> AH[Ambiance sonore]
    AH --> AI[Choisir musique]
    AI --> AD

    AD --> AJ{Confirmer?}
    AJ -->|Oui| AK[Pret a generer]
    AJ -->|Non| B

    style A fill:#e3f2fd
    style AK fill:#e8f5e9
```

### Options de style

| Style | Description | Premium |
|-------|-------------|---------|
| Realiste | Rendu photoréaliste | Non |
| Cartoon | Style dessin anime occidental | Non |
| Manga | Style manga japonais | Non |
| Aquarelle | Effet peinture aquarelle | Non |
| Pixel Art | Retro gaming | Oui |
| Art Nouveau | Style artistique classique | Oui |
| Custom | Import de style personnel | Oui |

---

## Flow 5: Generation de VisioBook

### US Associees
- **US 3.1** - Generation automatique (MUST)
- **US 3.3** - Suivi de la progression (SHOULD)
- **US 3.4** - Previsualisation scenes (SHOULD)

### Flow generation

```mermaid
flowchart TD
    A[Ecran Detail - Config OK] --> B[Bouton Generer]
    B --> C{Credits disponibles?}

    C -->|Non| D[Modal Premium]
    D --> E{Upgrade?}
    E -->|Oui| F[Flow paiement]
    E -->|Non| G[Retour detail]

    C -->|Oui| H[Confirmation generation]
    H --> I{Confirmer?}
    I -->|Non| G

    I -->|Oui| J[Demarrer generation]
    J --> K[Ecran Progress]

    K --> L[Etape 1: Analyse scenes]
    L --> M[Progress 0-20%]
    M --> N[Preview scenes extraites]

    N --> O[Etape 2: Generation images]
    O --> P[Progress 20-60%]
    P --> Q[Thumbnails intermediaires]

    Q --> R[Etape 3: Synthese audio]
    R --> S[Progress 60-80%]

    S --> T[Etape 4: Assemblage]
    T --> U[Progress 80-100%]

    U --> V{Generation OK?}

    V -->|Non| W[Ecran erreur]
    W --> X{Type erreur}
    X -->|Timeout| Y[Reessayer auto]
    Y --> J
    X -->|Contenu| Z[Ajuster contenu]
    Z --> G
    X -->|Serveur| AA[Notifier plus tard]
    AA --> AB[Retour dashboard]

    V -->|Oui| AC[Generation complete]
    AC --> AD[Animation celebration]
    AD --> AE[Bouton voir resultat]
    AE --> AF[Ecran Player]

    F --> C

    style A fill:#e3f2fd
    style AF fill:#e8f5e9
    style W fill:#ffebee
    style D fill:#fff3e0
```

### Etapes de generation

| Etape | Description | Duree estimee | Progress |
|-------|-------------|---------------|----------|
| 1. Analyse | Extraction scenes et personnages | 10-30s | 0-20% |
| 2. Images | Generation images IA | 1-5 min | 20-60% |
| 3. Audio | Synthese vocale + musique | 30s-2min | 60-80% |
| 4. Assemblage | Montage video final | 20-60s | 80-100% |

---

## Flow 6: Visualisation (Player)

### US Associees
- **US 3.2** - Visualisation dans l'application (MUST)
- **US 3.5** - Controles de lecture (MUST)

### Flow player

```mermaid
flowchart TD
    A[Ouvrir VisioBook] --> B[Ecran Player]
    B --> C[Chargement video]
    C --> D{Stream OK?}

    D -->|Non| E[Erreur chargement]
    E --> F{Connexion?}
    F -->|Non| G[Mode hors-ligne]
    G --> H{Cache disponible?}
    H -->|Oui| I[Lecture cache]
    H -->|Non| J[Message: telecharger d'abord]

    F -->|Oui| K[Reessayer stream]
    K --> C

    D -->|Oui| L[Afficher video]

    L --> M[Controles visibles]
    M --> N{Action utilisateur}

    N -->|Play/Pause| O[Toggle lecture]
    N -->|Seek| P[Deplacer timeline]
    N -->|Volume| Q[Ajuster son]
    N -->|Fullscreen| R[Mode plein ecran]
    N -->|Vitesse| S[Changer vitesse]
    N -->|Sous-titres| T[Toggle sous-titres]

    O & P & Q & R & S & T --> M

    N -->|Terminer| U[Fin video]
    U --> V[Ecran fin]
    V --> W{Action suivante}

    W -->|Rejouer| L
    W -->|Partager| X[Flow partage]
    W -->|Telecharger| Y[Flow download]
    W -->|Fermer| Z[Retour detail]

    I --> L
    J --> Z

    style B fill:#e3f2fd
    style L fill:#e8f5e9
    style E fill:#ffebee
```

### Controles du player

| Controle | Geste/Action | Description |
|----------|--------------|-------------|
| Play/Pause | Tap centre | Toggle lecture |
| Seek | Drag timeline | Navigation temporelle |
| Volume | Slider ou boutons physiques | Ajustement son |
| Fullscreen | Tap icone ou rotation | Plein ecran |
| Skip 10s | Double tap gauche/droite | Avance/recul rapide |
| Vitesse | Menu 0.5x, 1x, 1.5x, 2x | Vitesse lecture |
| Sous-titres | Toggle CC | Afficher/masquer |

---

## Flow 7: Export et Partage

### US Associees
- **US 4.1** - Telecharger la video (MUST)
- **US 4.2** - Partage reseaux sociaux (COULD)
- **US 4.3** - Choix format export (SHOULD)
- **US 4.4** - Generation lien partage (SHOULD)

### Flow export

```mermaid
flowchart TD
    A[Player ou Detail] --> B[Bouton Export]
    B --> C[Modal Export]

    C --> D{Type export}

    D -->|Telecharger| E[Choix qualite]
    E --> F{Qualite}
    F -->|SD 480p| G[Fichier leger]
    F -->|HD 720p| H[Fichier standard]
    F -->|FHD 1080p| I[Fichier HD]
    F -->|4K| J{Premium?}
    J -->|Non| K[Upgrade required]
    J -->|Oui| L[Fichier 4K]

    G & H & I & L --> M[Choix format]
    M --> N{Format}
    N -->|MP4| O[Export MP4]
    N -->|MOV| P[Export MOV]
    N -->|GIF| Q[Export GIF court]

    O & P & Q --> R[Telechargement]
    R --> S[Progress download]
    S --> T{Download OK?}
    T -->|Non| U[Erreur download]
    U --> V[Reessayer]
    V --> R
    T -->|Oui| W[Fichier sauvegarde]
    W --> X[Notification succes]

    D -->|Lien| Y[Generer lien]
    Y --> Z[Lien cree]
    Z --> AA[Copier presse-papier]
    AA --> AB[Options lien]
    AB --> AC{Protection?}
    AC -->|Oui| AD[Definir mot de passe]
    AD --> AE[Lien protege]
    AC -->|Non| AE

    D -->|Reseaux| AF[Share Sheet]
    AF --> AG{Plateforme}
    AG -->|Instagram| AH[Story/Post]
    AG -->|Facebook| AI[Publication FB]
    AG -->|Twitter| AJ[Tweet video]
    AG -->|TikTok| AK[Upload TikTok]
    AG -->|WhatsApp| AL[Message WA]
    AG -->|Autre| AM[App native]

    AH & AI & AJ & AK & AL & AM --> AN[Partage effectue]

    X --> AO[Fin export]
    AE --> AO
    AN --> AO

    K --> C

    style A fill:#e3f2fd
    style AO fill:#e8f5e9
    style U fill:#ffebee
    style K fill:#fff3e0
```

### Options d'export

| Option | Standard | Premium |
|--------|----------|---------|
| 480p SD | Oui | Oui |
| 720p HD | Oui | Oui |
| 1080p FHD | Non | Oui |
| 4K UHD | Non | Oui |
| Sans filigrane | Non | Oui |
| Lien protege | Non | Oui |

---

## Flow 8: Consultation Historique

### US Associees
- **US 5.1** - Acceder a l'historique (SHOULD)
- **US 5.2** - Modifier un projet (COULD)
- **US 5.3** - Dupliquer un projet (COULD)
- **US 5.5** - Supprimer un projet (SHOULD)

### Flow historique inputs

```mermaid
flowchart TD
    A[Tab Mes Textes] --> B[Liste inputs]
    B --> C{Liste vide?}

    C -->|Oui| D[Empty state]
    D --> E[CTA: Importer premier texte]
    E --> F[Flow Import]

    C -->|Non| G[Afficher liste]
    G --> H[Tri par date]
    H --> I{Recherche?}
    I -->|Oui| J[Barre recherche]
    J --> K[Filtrer resultats]
    K --> L[Afficher filtres]
    I -->|Non| L

    L --> M{Selection item}
    M --> N[Detail input]
    N --> O{Action}

    O -->|Voir| P[Afficher texte complet]
    O -->|Generer| Q[Flow Generation]
    O -->|Modifier| R[Editer texte]
    R --> S[Sauvegarder]
    S --> N

    O -->|Supprimer| T[Confirmation]
    T --> U{Confirmer?}
    U -->|Non| N
    U -->|Oui| V[Supprimer item]
    V --> G

    style A fill:#e3f2fd
    style D fill:#fff3e0
```

### Flow historique VisioBooks

```mermaid
flowchart TD
    A[Tab Mes VisioBooks] --> B[Liste VisioBooks]
    B --> C{Liste vide?}

    C -->|Oui| D[Empty state]
    D --> E[CTA: Creer premier VisioBook]
    E --> F[Flow Import]

    C -->|Non| G[Afficher grille]
    G --> H[Thumbnails videos]
    H --> I{Filtre?}
    I -->|Oui| J[Options filtre]
    J --> K[Par date/style/duree]
    K --> L[Appliquer filtre]
    I -->|Non| L

    L --> M{Selection item}
    M --> N[Preview thumbnail]
    N --> O{Action}

    O -->|Voir| P[Ouvrir Player]
    O -->|Details| Q[Ecran Detail]
    O -->|Partager| R[Flow Partage]
    O -->|Dupliquer| S[Creer copie]
    S --> T[Nouveau projet]
    T --> Q

    O -->|Supprimer| U[Confirmation]
    U --> V{Confirmer?}
    V -->|Non| N
    V -->|Oui| W[Supprimer VisioBook]
    W --> X[Liberer stockage]
    X --> G

    O -->|Regenerer| Y[Flow Configuration]

    style A fill:#e3f2fd
    style P fill:#e8f5e9
    style D fill:#fff3e0
```

### Tri et filtres disponibles

| Critere | Options |
|---------|---------|
| Date | Plus recent, Plus ancien |
| Style | Tous, Realiste, Cartoon, etc. |
| Duree | Court (<5min), Moyen, Long |
| Statut | Tous, Termine, En cours, Echec |

---

## Resume des flows par User Story

| US | Flow principal | Ecrans impliques |
|----|----------------|------------------|
| US P5.1 | Flow 1: Inscription | Login, Register, Validation |
| US 1.1 | Flow 2: Import fichier | Input, File Picker, Detail |
| US 1.2 | Flow 3: Scanner OCR | Input, Camera, OCR, Detail |
| US 1.3 | Flow 2 & 3: Preview | Detail (resume) |
| US 2.1-2.5 | Flow 4: Configuration | Detail (config) |
| US 3.1-3.4 | Flow 5: Generation | Detail, Progress, Player |
| US 3.5 | Flow 6: Player | Player |
| US 4.1-4.4 | Flow 7: Export | Player, Modal Export |
| US 5.1-5.5 | Flow 8: Historique | Historiques, Detail |
