# Screen Flows - VisioBook Mobile

## Vue d'ensemble

Ce document presente les enchainements d'ecrans (Screen Flows), la navigation et les etats des interfaces de l'application mobile VisioBook.

---

## Arborescence complete des ecrans (Sitemap)

```mermaid
graph TB
    subgraph "Pre-Auth"
        SPLASH[Splash Screen]
        ONBOARD[Onboarding<br/>3 slides]
        LOGIN[Login]
        REGISTER[Register]
        FORGOT[Forgot Password]
        VERIFY[Email Verification]
    end

    subgraph "Main App - Bottom Tabs"
        TAB_HOME[Tab: Accueil]
        TAB_SCAN[Tab: Scanner]
        TAB_INPUTS[Tab: Mes Textes]
        TAB_VISIO[Tab: Mes VisioBooks]
    end

    subgraph "Dashboard"
        DASH[Dashboard Home]
        DASH_EMPTY[Dashboard Empty State]
        DASH_RECENT[Recent Projects]
    end

    subgraph "Input/Scanner"
        INPUT[Input Home]
        FILE_PICK[File Picker]
        CAMERA[Camera Scanner]
        OCR_PROGRESS[OCR Progress]
        OCR_RESULT[OCR Result]
        TEXT_EDIT[Text Editor]
    end

    subgraph "Project Detail"
        DETAIL[Project Detail]
        DETAIL_CONFIG[Configuration]
        DETAIL_STYLE[Style Selection]
        DETAIL_AUDIO[Audio Settings]
        DETAIL_ADVANCED[Advanced Options]
        GENERATION[Generation Progress]
        GEN_ERROR[Generation Error]
    end

    subgraph "Player"
        PLAYER[Video Player]
        PLAYER_FULL[Fullscreen Player]
        PLAYER_END[End Screen]
    end

    subgraph "Export/Share"
        EXPORT[Export Modal]
        SHARE[Share Sheet]
        DOWNLOAD[Download Progress]
    end

    subgraph "History"
        HIST_INPUTS[Input History List]
        HIST_VISIO[VisioBook History Grid]
        HIST_SEARCH[Search/Filter]
    end

    subgraph "Settings/Profile"
        PROFILE[Profile]
        SETTINGS[Settings]
        PREMIUM[Premium Upgrade]
        NOTIF[Notifications Settings]
    end

    %% Pre-auth flow
    SPLASH --> ONBOARD
    ONBOARD --> LOGIN
    LOGIN --> REGISTER
    LOGIN --> FORGOT
    REGISTER --> VERIFY
    FORGOT --> LOGIN

    %% Auth to main
    LOGIN --> TAB_HOME
    VERIFY --> TAB_HOME

    %% Tab navigation
    TAB_HOME --> DASH
    TAB_SCAN --> INPUT
    TAB_INPUTS --> HIST_INPUTS
    TAB_VISIO --> HIST_VISIO

    %% Dashboard
    DASH --> DASH_EMPTY
    DASH --> DASH_RECENT
    DASH_RECENT --> DETAIL

    %% Input flows
    INPUT --> FILE_PICK
    INPUT --> CAMERA
    FILE_PICK --> DETAIL
    CAMERA --> OCR_PROGRESS
    OCR_PROGRESS --> OCR_RESULT
    OCR_RESULT --> TEXT_EDIT
    TEXT_EDIT --> DETAIL

    %% Project detail
    DETAIL --> DETAIL_CONFIG
    DETAIL_CONFIG --> DETAIL_STYLE
    DETAIL_CONFIG --> DETAIL_AUDIO
    DETAIL_CONFIG --> DETAIL_ADVANCED
    DETAIL --> GENERATION
    GENERATION --> GEN_ERROR
    GENERATION --> PLAYER

    %% Player
    PLAYER --> PLAYER_FULL
    PLAYER --> PLAYER_END
    PLAYER_END --> EXPORT

    %% Export
    EXPORT --> SHARE
    EXPORT --> DOWNLOAD

    %% History
    HIST_INPUTS --> DETAIL
    HIST_INPUTS --> HIST_SEARCH
    HIST_VISIO --> PLAYER
    HIST_VISIO --> DETAIL
    HIST_VISIO --> HIST_SEARCH

    %% Settings
    DASH --> PROFILE
    PROFILE --> SETTINGS
    SETTINGS --> PREMIUM
    SETTINGS --> NOTIF

    classDef preauth fill:#e3f2fd,stroke:#1976d2
    classDef tabs fill:#fff3e0,stroke:#f57c00
    classDef main fill:#e8f5e9,stroke:#388e3c
    classDef player fill:#f3e5f5,stroke:#7b1fa2
    classDef modal fill:#fce4ec,stroke:#c2185b

    class SPLASH,ONBOARD,LOGIN,REGISTER,FORGOT,VERIFY preauth
    class TAB_HOME,TAB_SCAN,TAB_INPUTS,TAB_VISIO tabs
    class DASH,DASH_EMPTY,DASH_RECENT,INPUT,FILE_PICK,CAMERA,OCR_PROGRESS,OCR_RESULT,TEXT_EDIT,DETAIL,DETAIL_CONFIG,DETAIL_STYLE,DETAIL_AUDIO,DETAIL_ADVANCED,GENERATION,GEN_ERROR,HIST_INPUTS,HIST_VISIO,HIST_SEARCH,PROFILE,SETTINGS,PREMIUM,NOTIF main
    class PLAYER,PLAYER_FULL,PLAYER_END player
    class EXPORT,SHARE,DOWNLOAD modal
```

---

## Navigation principale

### Structure de navigation

```mermaid
graph LR
    subgraph "Bottom Tab Bar"
        T1[Accueil<br/>HomeIcon]
        T2[Scanner<br/>CameraIcon]
        T3[Mes Textes<br/>FileTextIcon]
        T4[Mes VisioBooks<br/>PlayCircleIcon]
    end

    subgraph "Stack Navigator - Home"
        S1_1[Dashboard]
        S1_2[Profile]
        S1_3[Settings]
        S1_4[Project Detail]
    end

    subgraph "Stack Navigator - Scanner"
        S2_1[Input Home]
        S2_2[Camera]
        S2_3[File Picker]
        S2_4[OCR Result]
        S2_5[Project Detail]
    end

    subgraph "Stack Navigator - Inputs"
        S3_1[History List]
        S3_2[Input Detail]
        S3_3[Project Detail]
    end

    subgraph "Stack Navigator - VisioBooks"
        S4_1[History Grid]
        S4_2[Project Detail]
        S4_3[Player]
    end

    T1 --> S1_1
    T2 --> S2_1
    T3 --> S3_1
    T4 --> S4_1
```

### Configuration React Navigation

```typescript
// Navigation structure
const TabNavigator = {
  Home: {
    screen: HomeStack,
    icon: 'home',
    label: 'Accueil'
  },
  Scanner: {
    screen: ScannerStack,
    icon: 'camera',
    label: 'Scanner'
  },
  Inputs: {
    screen: InputsStack,
    icon: 'file-text',
    label: 'Mes Textes'
  },
  VisioBooks: {
    screen: VisioBooksStack,
    icon: 'play-circle',
    label: 'Mes VisioBooks'
  }
};
```

---

## Navigation secondaire

### Header Navigation

| Ecran | Left Action | Title | Right Actions |
|-------|-------------|-------|---------------|
| Dashboard | Menu/Profile | "VisioBook" | Notifications |
| Input | Back | "Nouveau" | - |
| Camera | Close | "Scanner" | Flash, Settings |
| History | - | "Mes Textes" | Search, Filter |
| Detail | Back | Titre projet | More (menu) |
| Player | Close | - | Share, Download |
| Settings | Back | "Parametres" | - |

### Modales et Sheets

```mermaid
graph TB
    subgraph "Bottom Sheets"
        BS1[Style Selection Sheet]
        BS2[Audio Settings Sheet]
        BS3[Filter Options Sheet]
        BS4[Sort Options Sheet]
    end

    subgraph "Full Modals"
        FM1[Export Modal]
        FM2[Premium Upgrade Modal]
        FM3[Error Details Modal]
        FM4[Confirmation Dialog]
    end

    subgraph "Alerts"
        AL1[Delete Confirmation]
        AL2[Exit Confirmation]
        AL3[Permission Request]
        AL4[Error Alert]
    end
```

---

## Etats des ecrans

### Dashboard - Etats

```mermaid
stateDiagram-v2
    [*] --> Loading: Open app
    Loading --> Empty: No projects
    Loading --> WithData: Has projects
    Loading --> Error: Load failed

    Empty --> WithData: Create first project
    WithData --> Empty: Delete all projects

    Error --> Loading: Retry
    Error --> Empty: Dismiss

    WithData --> Refreshing: Pull to refresh
    Refreshing --> WithData: Refresh complete
    Refreshing --> Error: Refresh failed

    state WithData {
        [*] --> RecentProjects
        RecentProjects --> AllProjects: View all
        AllProjects --> RecentProjects: Back
    }
```

### Composants d'etat Dashboard

```yaml
Loading State:
  Components:
    - Skeleton cards (3x)
    - Loading indicator subtle
  Duration: Max 3s puis timeout

Empty State:
  Components:
    - Illustration vide
    - Titre: "Aucun projet"
    - Description: "Creez votre premier VisioBook"
    - CTA: "Commencer" -> Tab Scanner

WithData State:
  Components:
    - Header avec stats (X projets, Y VisioBooks)
    - Section "Recents" (3 derniers)
    - Section "En cours" (si applicable)
    - CTA "Nouveau projet"

Error State:
  Components:
    - Icone erreur
    - Message explicatif
    - Bouton "Reessayer"
    - Lien "Aide"
```

### Input/Scanner - Etats

```mermaid
stateDiagram-v2
    [*] --> Idle: Open screen

    state Idle {
        [*] --> ModeSelection
        ModeSelection --> FileMode: Select file
        ModeSelection --> CameraMode: Select camera
    }

    FileMode --> FilePicking: Open picker
    FilePicking --> Idle: Cancel
    FilePicking --> Uploading: File selected

    CameraMode --> PermissionCheck: Init camera
    PermissionCheck --> CameraReady: Granted
    PermissionCheck --> PermissionDenied: Denied
    PermissionDenied --> Settings: Open settings
    Settings --> PermissionCheck: Return

    CameraReady --> Capturing: Take photo
    Capturing --> Preview: Photo taken
    Preview --> CameraReady: Retake
    Preview --> OCRProcessing: Confirm

    Uploading --> Processing: Upload complete
    Uploading --> UploadError: Upload failed
    UploadError --> FilePicking: Retry

    OCRProcessing --> OCRResult: OCR complete
    OCRProcessing --> OCRError: OCR failed
    OCRError --> TextEdit: Manual entry

    Processing --> AnalysisResult: Analysis complete
    OCRResult --> TextEdit: Edit text
    TextEdit --> AnalysisResult: Confirm text

    AnalysisResult --> ProjectDetail: Continue
```

### Generation - Etats

```mermaid
stateDiagram-v2
    [*] --> Ready: Config complete

    Ready --> Queued: Start generation
    Queued --> Processing: Job started

    state Processing {
        [*] --> SceneExtraction
        SceneExtraction --> ImageGeneration: Scenes ready
        ImageGeneration --> AudioSynthesis: Images ready
        AudioSynthesis --> Assembly: Audio ready
        Assembly --> [*]: Complete
    }

    Processing --> Complete: Success
    Processing --> Failed: Error

    Failed --> Ready: Adjust & retry
    Failed --> Queued: Auto-retry

    Complete --> PlayerReady: View result
```

### Composants d'etat Generation

```yaml
Ready State:
  Components:
    - Resume configuration
    - Estimation temps
    - Bouton "Generer"
    - Info credits restants

Queued State:
  Components:
    - Animation attente
    - Position dans la queue
    - Option annuler
    - Estimation attente

Processing State:
  Components:
    - Progress bar globale
    - Etape actuelle avec description
    - Pourcentage completion
    - Thumbnails intermediaires (si dispo)
    - Temps restant estime
    - Option "Notifier quand pret"

Complete State:
  Components:
    - Animation celebration
    - Preview thumbnail
    - Stats generation (duree, scenes, etc.)
    - CTA "Voir le resultat"
    - Options rapides (partager, telecharger)

Failed State:
  Components:
    - Icone erreur
    - Message d'erreur comprehensible
    - Cause probable
    - Actions: Reessayer, Ajuster, Support
```

### Player - Etats

```mermaid
stateDiagram-v2
    [*] --> Loading: Open player

    Loading --> Ready: Video loaded
    Loading --> BufferError: Load failed

    Ready --> Playing: Press play
    Ready --> Seeking: Drag timeline

    Playing --> Paused: Press pause
    Playing --> Seeking: Drag timeline
    Playing --> Buffering: Network slow
    Playing --> Ended: Video complete

    Paused --> Playing: Press play
    Paused --> Seeking: Drag timeline

    Seeking --> Playing: Seek complete + was playing
    Seeking --> Paused: Seek complete + was paused

    Buffering --> Playing: Buffer ready
    Buffering --> BufferError: Timeout

    BufferError --> Loading: Retry
    BufferError --> Offline: No connection

    Offline --> Loading: Connection restored

    Ended --> Ready: Replay
    Ended --> ExportFlow: Export
    Ended --> ShareFlow: Share
```

---

## Transitions et animations

### Types de transitions

| Transition | Type | Duree | Ecrans |
|------------|------|-------|--------|
| Tab switch | Fade | 150ms | Entre tabs |
| Push screen | Slide right | 300ms | Stack navigation |
| Pop screen | Slide left | 300ms | Retour arriere |
| Modal open | Slide up | 350ms | Modales full |
| Modal close | Slide down | 300ms | Fermeture modale |
| Sheet open | Slide up | 250ms | Bottom sheets |
| Sheet close | Slide down | 200ms | Fermeture sheet |

### Animations specifiques

```yaml
Splash Screen:
  - Logo fade in: 500ms
  - Logo scale up: 300ms
  - Transition vers onboarding: slide left 400ms

Onboarding Slides:
  - Swipe horizontal entre slides
  - Indicators dot animation
  - Image parallax effect

Generation Progress:
  - Progress bar smooth animation
  - Step icons pulse when active
  - Confetti animation on complete

Player Controls:
  - Fade in/out on tap: 200ms
  - Auto-hide after 3s
  - Timeline thumb scale on drag

Card Interactions:
  - Press scale down: 0.98 over 100ms
  - Release scale up: 1.0 over 100ms
  - Shadow elevation change

Delete Actions:
  - Swipe to reveal delete
  - Red background slide in
  - Item slide out on confirm
```

---

## Navigation par gestes

### Gestes globaux

| Geste | Action | Contexte |
|-------|--------|----------|
| Swipe right | Retour arriere | Stack screens |
| Swipe down | Fermer modal | Modales |
| Pull down | Refresh | Listes |
| Long press | Menu contextuel | Cards, items |
| Pinch | Zoom | Images, Player |

### Gestes specifiques

```yaml
Player:
  - Single tap: Toggle controls
  - Double tap left: -10 secondes
  - Double tap right: +10 secondes
  - Swipe up/down: Volume
  - Pinch: Zoom video

Camera Scanner:
  - Tap: Focus
  - Long press: Lock exposure
  - Pinch: Zoom

History Grid:
  - Long press card: Select mode
  - Drag: Reorder (collections)
```

---

## Deep linking

### Schema de liens

```
visiobook://

Routes:
  visiobook://login
  visiobook://register
  visiobook://dashboard
  visiobook://project/{projectId}
  visiobook://project/{projectId}/play
  visiobook://scanner
  visiobook://history/inputs
  visiobook://history/visiobooks
  visiobook://settings
  visiobook://premium
```

### Exemples de liens partageables

```
# Lien vers un VisioBook (public)
https://visiobook.app/share/abc123

# Lien vers un VisioBook (protege)
https://visiobook.app/share/abc123?protected=true

# Lien d'invitation
https://visiobook.app/invite/xyz789
```

---

## Responsive et orientations

### Breakpoints

| Device | Width | Layout |
|--------|-------|--------|
| Phone portrait | < 428px | Default |
| Phone landscape | < 926px | Adapted |
| Tablet portrait | < 834px | 2 columns |
| Tablet landscape | < 1194px | 3 columns |

### Adaptations par ecran

```yaml
Dashboard:
  Phone: Single column, cards stacked
  Tablet: 2-3 columns grid

History Grid:
  Phone Portrait: 2 columns
  Phone Landscape: 3 columns
  Tablet: 4-5 columns

Player:
  Phone Portrait: 16:9 top, controls bottom
  Phone Landscape: Fullscreen auto
  Tablet: Picture-in-picture option

Scanner:
  Phone: Portrait optimized
  Tablet: Camera preview larger
```

### Support orientation

| Ecran | Portrait | Landscape |
|-------|----------|-----------|
| Splash | Oui | Non |
| Onboarding | Oui | Non |
| Login | Oui | Oui |
| Dashboard | Oui | Oui |
| Scanner | Oui | Oui |
| Player | Oui | Oui (preferred) |
| Detail | Oui | Oui |
| Settings | Oui | Oui |

---

## Accessibilite

### Labels et hints

```yaml
Tous les ecrans:
  - Labels descriptifs pour VoiceOver/TalkBack
  - Hints pour les actions
  - Focus order logique
  - Zones touch minimum 44x44pt

Player:
  - "Lecture" / "Pause" toggle
  - "Position: X minutes Y secondes"
  - "Avancer de 10 secondes"
  - "Reculer de 10 secondes"

Scanner:
  - "Prendre une photo du texte"
  - "Flash: Active/Desactive"
  - "Cadrage detecte, appuyez pour capturer"

Progress:
  - "Generation en cours, X pourcent complete"
  - "Etape Y sur Z"
  - Announcements automatiques a chaque etape
```

### Themes et contraste

| Element | Light Mode | Dark Mode | Contrast Ratio |
|---------|------------|-----------|----------------|
| Text primary | #1A1A1A | #FFFFFF | > 7:1 |
| Text secondary | #666666 | #AAAAAA | > 4.5:1 |
| Background | #FFFFFF | #1A1A1A | - |
| Primary action | #2196F3 | #64B5F6 | > 4.5:1 |
| Error | #D32F2F | #EF5350 | > 4.5:1 |
| Success | #388E3C | #66BB6A | > 4.5:1 |
