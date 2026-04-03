# Infrastructure Base de Donnees (CNPG)

## Informations generales

| Propriete | Valeur |
|-----------|--------|
| **Technologie** | PostgreSQL via CloudNativePG (CNPG) operator |
| **Clusters** | 2 (postgres-crud, postgres-io) |
| **Phase** | 1 - Infrastructure Critique |
| **Priorite** | CRITIQUE (prerequis pour tous les services) |

## Architecture

Les bases de donnees sont gerees par l'operateur [CloudNativePG](https://cloudnative-pg.io/) qui deploie et supervise des clusters PostgreSQL natifs dans Kubernetes. Chaque service se connecte **directement** a sa base via son propre ORM et une `DATABASE_URL` fournie par un secret Kubernetes.

Il n'y a pas de service intermediaire : chaque microservice est responsable de son propre schema et de ses propres migrations.

```mermaid
graph TB
    subgraph "Cluster postgres-crud"
        PGCRUD[(PostgreSQL<br/>postgres-crud)]
        DB_USER[(core_user)]
        DB_PROJECT[(core_project)]
        PGCRUD --- DB_USER
        PGCRUD --- DB_PROJECT
    end

    subgraph "Cluster postgres-io"
        PGIO[(PostgreSQL<br/>postgres-io)]
        DB_STORAGE[(storage)]
        DB_ANALYSIS[(analysis)]
        DB_INGESTION[(ingestion)]
        PGIO --- DB_STORAGE
        PGIO --- DB_ANALYSIS
        PGIO --- DB_INGESTION
    end

    USER[core-user-service] -->|ORM direct| DB_USER
    PROJECT[core-project-service] -->|ORM direct| DB_PROJECT
    STORAGE[support-storage-service] -->|ORM direct| DB_STORAGE
    AI[ai-analysis-service] -->|ORM direct| DB_ANALYSIS
    INGEST[content-ingestion-service] -->|ORM direct| DB_INGESTION

    classDef crud fill:#e3f2fd,stroke:#1976d2
    classDef io fill:#e8f5e9,stroke:#388e3c
    classDef service fill:#fff3e0,stroke:#f57c00

    class PGCRUD,DB_USER,DB_PROJECT crud
    class PGIO,DB_STORAGE,DB_ANALYSIS,DB_INGESTION io
    class USER,PROJECT,STORAGE,AI,INGEST service
```

## Mapping services / bases de donnees

| Service | Cluster | Database | User | Secret | Status |
|---------|---------|----------|------|--------|--------|
| core-user-service | postgres-crud | core_user | core_user_app | postgres-crud-user-secret | Deploye |
| core-project-service | postgres-crud | core_project | core_project_app | postgres-crud-project-secret | En dev |
| support-storage-service | postgres-io | storage | storage_user | postgres-io-storage-secret | Deploye |
| ai-analysis-service | postgres-io | analysis | analysis_user | postgres-io-analysis-secret | Deploye |
| content-ingestion-service | postgres-io | ingestion | ingestion_user | postgres-io-ingestion-secret | Deploye |
| core-notification-service | TBD | TBD | TBD | TBD | Futur |
| core-payment-service | TBD | TBD | TBD | TBD | Futur |
| support-security-service | TBD | TBD | TBD | TBD | Futur |
| ai-media-generation-service | TBD | TBD | TBD | TBD | Futur |
| ai-storyboard-assembly-service | TBD | TBD | TBD | TBD | Futur |

## Strategie de separation des clusters

| Cluster | Vocation | Taille stockage | Services |
|---------|----------|-----------------|----------|
| **postgres-crud** | Operations CRUD classiques (lecture/ecriture moderee) | 5Gi | core-user, core-project |
| **postgres-io** | Operations I/O intensives (fichiers, analyse, ingestion) | 20Gi | storage, analysis, ingestion |

## Connexion depuis un service

Chaque service recoit ses credentials via un **Secret Kubernetes** injecte en variables d'environnement. Les variables standard sont :

| Variable | Description | Exemple |
|----------|-------------|---------|
| `DATABASE_URL` | URL complete de connexion | `postgresql://user:pass@host:5432/db` |
| `DB_HOST` | Hostname du cluster | `postgres-crud-rw.visiobook-namespace.svc.cluster.local` |
| `DB_PORT` | Port PostgreSQL | `5432` |
| `DB_NAME` | Nom de la base | `core_user` |
| `DB_USERNAME` | Utilisateur | `core_user_app` |
| `DB_PASSWORD` | Mot de passe | (dans le secret) |

### Endpoints reseau

| Cluster | Endpoint lecture/ecriture | Usage |
|---------|---------------------------|-------|
| postgres-crud | `postgres-crud-rw.visiobook-namespace.svc.cluster.local:5432` | Toutes les operations |
| postgres-io | `postgres-io-rw.visiobook-namespace.svc.cluster.local:5432` | Toutes les operations |

### Exemples de connexion

**Python / SQLAlchemy :**
```python
from sqlalchemy import create_engine
import os

engine = create_engine(os.environ["DATABASE_URL"])
```

**Node.js / Prisma :**
```prisma
// schema.prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}
```

**Node.js / TypeORM :**
```typescript
const dataSource = new DataSource({
  type: "postgres",
  url: process.env.DATABASE_URL,
});
```

## Migrations

Chaque service gere ses propres migrations de schema via son ORM :

| ORM | Outil de migration | Commande |
|-----|-------------------|----------|
| SQLAlchemy (Python) | Alembic | `alembic upgrade head` |
| Prisma (Node.js) | Prisma Migrate | `npx prisma migrate deploy` |
| TypeORM (Node.js) | TypeORM migrations | `typeorm migration:run` |

Les migrations sont executees au demarrage du service ou dans un init container Kubernetes.

## Infrastructure Kubernetes

### Ordre de deploiement (sync-waves ArgoCD)

| Sync-wave | Ressource | Description |
|-----------|-----------|-------------|
| -2 | Secrets | Credentials des bases de donnees |
| -1 | Clusters CNPG | Clusters PostgreSQL (postgres-crud, postgres-io) |
| 0 | Services | Microservices applicatifs |

### Fichiers de configuration

| Fichier | Description |
|---------|-------------|
| `environnement/dev/charts/cloud-native-postgres/templates/cluster.yaml` | Template Helm du cluster CNPG |
| `environnement/dev/charts/cloud-native-postgres/values-crud.yaml` | Configuration cluster postgres-crud |
| `environnement/dev/charts/cloud-native-postgres/values-io.yaml` | Configuration cluster postgres-io |
| `environnement/dev/app/configs/secrets/postgres-*-secret.yaml` | Secrets des bases de donnees |

### Ajouter une nouvelle base de donnees

Pour ajouter un service qui a besoin d'une base de donnees :

1. **Choisir le cluster** : `postgres-crud` (CRUD) ou `postgres-io` (I/O intensif)

2. **Ajouter l'utilisateur et la base** dans le `postInitSQL` du fichier `values-{cluster}.yaml` :
   ```yaml
   postInitSQL:
     - CREATE USER new_user WITH PASSWORD 'NewPass123!'
     - CREATE DATABASE new_db OWNER new_user
     - GRANT ALL PRIVILEGES ON DATABASE new_db TO new_user
   ```

3. **Creer le secret** dans `environnement/dev/app/configs/secrets/` :
   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: postgres-{cluster}-{service}-secret
     namespace: visiobook-namespace
     annotations:
       argocd.argoproj.io/sync-wave: "-2"
     labels:
       app: postgres-{cluster}
   type: Opaque
   stringData:
     DATABASE_URL: "postgresql://new_user:NewPass123!@postgres-{cluster}-rw.visiobook-namespace.svc.cluster.local:5432/new_db"
     DB_HOST: "postgres-{cluster}-rw.visiobook-namespace.svc.cluster.local"
     DB_PORT: "5432"
     DB_NAME: "new_db"
     DB_USERNAME: "new_user"
     DB_PASSWORD: "NewPass123!"
   ```

4. **Ajouter la source ArgoCD** dans `application-visiobook.yml` avec :
   ```yaml
   helm:
     values: |
       existingSecret: "postgres-{cluster}-{service}-secret"
   ```

## Metriques de succes

| Metrique | Objectif | Description |
|----------|----------|-------------|
| Cluster health | Healthy | Phase CNPG = "Cluster in healthy state" |
| Connection time | < 5ms | Temps de connexion depuis un service |
| Availability | > 99.9% | Disponibilite des clusters |
| Storage usage | < 80% | Utilisation du stockage alloue |
