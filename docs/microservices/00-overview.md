# Documentation Microservices - VisioBook

## Vue d'ensemble

Cette documentation detaille les responsabilites de chaque microservice de l'architecture VisioBook, leurs communications inter-services, et les strategies de test.

## Table des matieres

| Document | Service | Phase | Description |
|----------|---------|-------|-------------|
| [database-infrastructure.md](./database-infrastructure.md) | Database | 1 | Infrastructure PostgreSQL (2 clusters CNPG) |
| [core-api-gateway.md](./core-api-gateway.md) | Gateway | 1 | Point d'entree unique, routage, auth |
| [core-user-service.md](./core-user-service.md) | User | 2 | Authentification, profils, permissions |
| [support-storage-service.md](./support-storage-service.md) | Storage | 2 | Stockage fichiers, OCR, streaming |
| [core-project-service.md](./core-project-service.md) | Project | 3 | Gestion projets, workflows |
| [ai-analysis-service.md](./ai-analysis-service.md) | AI Analysis | 3 | Analyse semantique, extraction scenes |
| [mobile-app.md](./mobile-app.md) | Mobile | **4** | **App mobile (PRIORITAIRE)** |
| [web-user-portal.md](./web-user-portal.md) | Web | 5 | Application web utilisateur |
| [core-payment-service.md](./core-payment-service.md) | Payment | 6 | Paiements, abonnements |
| [core-notification-service.md](./core-notification-service.md) | Notification | 6 | Emails, push, in-app |
| [support-security-service.md](./support-security-service.md) | Security | 6 | Encryption, audit, policies |
| [content-ingestion-service.md](./content-ingestion-service.md) | Ingestion | 7 | Pre-processing contenu |
| [ai-media-generation-service.md](./ai-media-generation-service.md) | Media Gen | 7 | Generation images/audio |
| [ai-storyboard-assembly-service.md](./ai-storyboard-assembly-service.md) | Assembly | 7 | Assemblage video |

## Architecture Globale

```mermaid
graph TB
    subgraph "Clients"
        MOBILE[mobile-app<br/>Phase 4 - PRIORITAIRE]
        WEB[web-user-portal<br/>Phase 5]
    end

    subgraph "Phase 1 - Infrastructure"
        GW[core-api-gateway<br/>Port 8080]
        PGCRUD[(postgres-crud<br/>CNPG Cluster)]
        PGIO[(postgres-io<br/>CNPG Cluster)]
    end

    subgraph "Phase 2 - Core"
        USER[core-user-service<br/>Port 8081]
        STORAGE[support-storage-service<br/>Port 8089]
    end

    subgraph "Phase 3 - Metier"
        PROJECT[core-project-service<br/>Port 8086]
        AI[ai-analysis-service<br/>Port 8083]
    end

    subgraph "Phase 6 - Complementaires"
        PAY[core-payment-service<br/>Port 8087]
        NOTIF[core-notification-service<br/>Port 8088]
        SEC[support-security-service<br/>Port 8093]
    end

    subgraph "Phase 7 - Pipeline IA"
        INGEST[content-ingestion-service<br/>Port 8082]
        MEDIA[ai-media-generation-service<br/>Port 8084]
        ASSEMBLY[ai-storyboard-assembly-service<br/>Port 8085]
    end

    subgraph "External Storage"
        BLOB[Azure Blob]
    end

    MOBILE --> GW
    WEB --> GW

    GW --> USER
    GW --> PROJECT
    GW --> STORAGE
    GW --> AI
    GW --> PAY
    GW --> NOTIF

    USER --> PGCRUD
    PROJECT --> PGCRUD
    STORAGE --> PGIO
    AI --> PGIO
    INGEST --> PGIO
    PAY --> PGCRUD
    NOTIF --> PGCRUD
    SEC --> PGCRUD
    MEDIA --> PGIO
    ASSEMBLY --> PGIO

    STORAGE --> BLOB

    PROJECT --> AI
    PROJECT --> STORAGE
    AI --> INGEST
    AI --> MEDIA
    MEDIA --> ASSEMBLY
    ASSEMBLY --> STORAGE

    classDef priority fill:#ff6b6b,stroke:#333,stroke-width:3px
    classDef phase1 fill:#e3f2fd,stroke:#1976d2
    classDef phase2 fill:#e8f5e9,stroke:#388e3c
    classDef phase3 fill:#fff3e0,stroke:#f57c00
    classDef phase6 fill:#fce4ec,stroke:#c2185b
    classDef phase7 fill:#f3e5f5,stroke:#7b1fa2
    classDef db fill:#dcedc8,stroke:#689f38

    class MOBILE priority
    class GW phase1
    class PGCRUD,PGIO db
    class USER,STORAGE phase2
    class PROJECT,AI phase3
    class PAY,NOTIF,SEC phase6
    class INGEST,MEDIA,ASSEMBLY phase7
```

## Matrice Flow -> Microservice

| Flow | Owner Principal | Services Impliques |
|------|-----------------|-------------------|
| **Flow 1: Auth** | core-user-service | api-gateway, notification |
| **Flow 2: Import fichier** | core-project-service | storage, ingestion |
| **Flow 3: OCR Scanner** | support-storage-service | project, ingestion |
| **Flow 4: Configuration** | core-project-service | - |
| **Flow 5: Generation** | ai-analysis-service | media-gen, assembly, storage, project |
| **Flow 6: Player** | support-storage-service | project, assembly |
| **Flow 7: Export/Partage** | core-project-service | storage |
| **Flow 8: Historique** | core-project-service | - |

## Matrice des Dependances

```mermaid
graph LR
    subgraph "Niveau 0 - Infrastructure"
        PGCRUD[(postgres-crud)]
        PGIO[(postgres-io)]
        GW[core-api-gateway]
    end

    subgraph "Niveau 1 - Services de base"
        USER[core-user-service]
        STORAGE[support-storage-service]
        SEC[support-security-service]
    end

    subgraph "Niveau 2 - Services metier"
        PROJECT[core-project-service]
        AI[ai-analysis-service]
        PAY[core-payment-service]
        NOTIF[core-notification-service]
        INGEST[content-ingestion-service]
    end

    subgraph "Niveau 3 - Pipeline IA"
        MEDIA[ai-media-generation-service]
    end

    subgraph "Niveau 4 - Assemblage"
        ASSEMBLY[ai-storyboard-assembly-service]
    end

    subgraph "Niveau 5 - Frontends"
        MOBILE[mobile-app]
        WEB[web-user-portal]
    end

    PGCRUD -.-> USER
    PGCRUD -.-> PROJECT
    PGIO -.-> STORAGE
    PGIO -.-> AI
    PGIO -.-> INGEST

    USER --> PROJECT
    STORAGE --> PROJECT
    USER --> PAY
    USER --> NOTIF
    STORAGE --> AI
    STORAGE --> INGEST

    AI --> MEDIA
    PROJECT --> INGEST

    MEDIA --> ASSEMBLY

    GW --> MOBILE
    GW --> WEB
```

## Ports et Technologies

| Service | Port | Stack | Repository |
|---------|------|-------|------------|
| core-api-gateway | 8080 | Kong/Docker | core-api-gateway |
| core-user-service | 8081 | Python/FastAPI | core-user-service |
| content-ingestion-service | 8082 | Python/FastAPI | content-ingestion-service |
| ai-analysis-service | 8083 | Python/FastAPI/PyTorch | ai-analysis-service |
| ai-media-generation-service | 8084 | Python/FastAPI/GPU | ai-media-generation-service |
| ai-storyboard-assembly-service | 8085 | Node.js | ai-storyboard-assembly-service |
| core-project-service | 8086 | Node.js/NestJS | core-project-service |
| core-payment-service | 8087 | Node.js/NestJS | core-payment-service |
| core-notification-service | 8088 | Node.js/NestJS | core-notification-service |
| support-storage-service | 8089 | Python/FastAPI | support-storage-service |
| support-security-service | 8093 | Java/Spring | support-security-service |
| web-user-portal | 3000 | TypeScript/Vue.js | web-user-portal |
| mobile-app | - | Mobile natif | mobile-app |

## Services Exclus du Scope MVP

Les services suivants sont redondants avec les outils Kubernetes/Azure natifs:

| Service | Alternative |
|---------|-------------|
| ~~support-config-service~~ | Kubernetes ConfigMaps/Secrets |
| ~~support-monitoring-service~~ | Prometheus/Grafana via Helm |
| ~~support-logging-service~~ | Azure Monitor/Log Analytics |
| ~~support-analytics-service~~ | Services tiers (Mixpanel, etc.) |

## Liens vers Documentation UX

- [User Journey Mapping](../ux/01-user-journey-mapping.md)
- [User Flows](../ux/02-user-flows.md)
- [Screen Flows](../ux/03-screen-flows.md)
- [MVP Screens](../ux/04-mvp-screens.md)
- [Interaction Sequences](../ux/05-interaction-sequences.md)

## Liens vers Documentation API

- [Core User Service API](../api/core-user-service.md)
- [Core Project Service API](../api/core-project-service.md)
- [Support Storage Service API](../api/support-storage-service.md)
- [AI Analysis Service API](../api/ai-analysis-service.md)

## Liens vers Documentation Architecture

- [Architecture Prioritaire](../architecture/visiobook-priority-architecture.md)
