# API Documentation - Support Storage Service

## Vue d'ensemble du service

### Rôle et responsabilités
Le **Support Storage Service** est responsable de la gestion centralisée de tous les fichiers et médias de l'écosystème Visiobook. Il gère l'upload, le stockage, la transformation, la distribution et l'optimisation des coûts de stockage.

### Justification de l'atomisation
- **Performance** : Optimisation spécifique pour les opérations de stockage
- **Sécurité** : Gestion centralisée des permissions et chiffrement des fichiers
- **Coût** : Optimisation des coûts de stockage (tiers, compression, lifecycle)
- **Scalabilité** : Gestion indépendante des volumes de données massifs

### Informations techniques
- **Port** : 8089
- **Technology Stack** : Python 3.11 + FastAPI + Pillow + FFmpeg
- **Storage** : Azure Blob Storage + S3
- **Processing** : Image/Video transformation en temps réel
- **Version API** : v1

## Architecture du service

```mermaid
graph TB
    subgraph "Support Storage Service"
        API[API Layer<br/>FastAPI + Pydantic]
        UPLOAD[Upload Manager<br/>Multipart + Resumable]
        TRANSFORM[Media Processor<br/>Pillow + FFmpeg]
        METADATA[Metadata Manager<br/>EXIF + File Analysis]
        SECURITY[Security Module<br/>Permissions + Encryption]
    end

    subgraph "Storage Backends"
        AZURE[Azure Blob Storage<br/>Primary Storage]
        S3[AWS S3<br/>Backup Storage]
        LOCAL[Local Storage<br/>Temp Processing]
    end

    subgraph "Processing & Cache"
        REDIS[(Redis<br/>Metadata Cache)]
    end

    API --> UPLOAD
    API --> TRANSFORM
    API --> METADATA
    API --> SECURITY

    UPLOAD --> AZURE
    UPLOAD --> S3
    UPLOAD --> LOCAL

    TRANSFORM --> LOCAL
    METADATA --> REDIS

    classDef service fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef storage fill:#fff8e1,stroke:#f9a825,stroke-width:2px
    classDef cache fill:#e8f5e8,stroke:#388e3c,stroke-width:2px

    class API,UPLOAD,TRANSFORM,METADATA,SECURITY service
    class AZURE,S3,LOCAL storage
    class REDIS cache
```

### Schémas de base de données

#### Note sur l'architecture de données

> **🏗️ Responsabilité de ce service (Phase actuelle)**
>
> Le Support Storage Service est **propriétaire** de toutes les données de fichiers, métadonnées et transformations. Il mocke localement les références aux utilisateurs (user_id) et projets (project_id) nécessaires à son fonctionnement.
>
> **🎯 Migration future**
>
> Lors de la centralisation via le Core Database Service, ce service fournira les migrations de référence pour toutes les tables liées au stockage et à la gestion des fichiers.

#### PostgreSQL - Tables propriétaires

```sql
-- Files table (PROPRIÉTAIRE - Support Storage Service)
-- Cette table est la source de vérité pour toutes les données de fichiers
CREATE TABLE files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL, -- RÉFÉRENCE MOCKÉE vers Core User Service
    project_id UUID, -- RÉFÉRENCE MOCKÉE vers Core Project Service
    original_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_type VARCHAR(100) NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    size_bytes BIGINT NOT NULL,
    checksum_md5 VARCHAR(32) NOT NULL,
    checksum_sha256 VARCHAR(64) NOT NULL,
    storage_provider VARCHAR(50) DEFAULT 'azure_blob',
    storage_tier VARCHAR(50) DEFAULT 'hot',
    encryption_key_id VARCHAR(255),
    is_public BOOLEAN DEFAULT FALSE,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP
);

CREATE INDEX idx_files_user_id ON files(user_id);
CREATE INDEX idx_files_project_id ON files(project_id);
CREATE INDEX idx_files_type ON files(file_type);
CREATE INDEX idx_files_created_at ON files(created_at);
CREATE INDEX idx_files_checksum_md5 ON files(checksum_md5);

-- File metadata table
CREATE TABLE file_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID REFERENCES files(id) ON DELETE CASCADE,
    metadata_type VARCHAR(50) NOT NULL,
    metadata JSONB NOT NULL,
    extracted_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_file_metadata_file_id ON file_metadata(file_id);
CREATE INDEX idx_file_metadata_type ON file_metadata(metadata_type);

-- File transformations table
CREATE TABLE file_transformations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file_id UUID REFERENCES files(id) ON DELETE CASCADE,
    target_file_id UUID REFERENCES files(id) ON DELETE CASCADE,
    transformation_type VARCHAR(100) NOT NULL,
    transformation_params JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    progress_percent INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_file_transformations_source ON file_transformations(source_file_id);
CREATE INDEX idx_file_transformations_target ON file_transformations(target_file_id);
CREATE INDEX idx_file_transformations_status ON file_transformations(status);

```

### Variables d'environnement

```bash
# Storage Providers
AZURE_STORAGE_ACCOUNT=visiobookstorage
AZURE_STORAGE_KEY=your-azure-storage-key
AZURE_CONTAINER_NAME=visiobook-files
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_S3_BUCKET=visiobook-backup
AWS_REGION=eu-west-1

# Processing
TEMP_STORAGE_PATH=/tmp/visiobook
MAX_FILE_SIZE=100MB
MAX_CONCURRENT_UPLOADS=10
IMAGE_QUALITY=85
VIDEO_BITRATE=2000k

# Security
ENCRYPTION_KEY=your-file-encryption-key
SIGNED_URL_EXPIRES=3600
ALLOWED_MIME_TYPES=image/jpeg,image/png,image/gif,video/mp4,application/pdf,text/plain

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/visiobook
REDIS_URL=redis://localhost:6379

# Monitoring
PROMETHEUS_PORT=9090
LOG_LEVEL=info
```

## Authentification et sécurité

> **📋 Référence** : Voir [Règles Communes](./regles_communes.md) pour les standards d'authentification, permissions et sécurité.

### Système JWT
```json
{
  "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "Content-Type": "multipart/form-data",
  "X-Upload-Session": "upload_session_id"
}
```

### Structure du JWT Token
```json
{
  "sub": "user_uuid",
  "email": "user@example.com",
  "role": "user|premium|admin",
  "subscription_type": "free|premium",
  "iat": 1642234567,
  "exp": 1642320967,
  "jti": "token_unique_id"
}
```

> **📋 Référence** : Voir [Règles Communes](./regles_communes.md) pour la gestion des permissions basées sur les rôles simples.

### Niveaux de permissions
- **admin** : Accès complet à tous les fichiers et gestion du stockage
- **user** : Accès aux fichiers personnels et upload selon quotas
- **premium** : Quotas étendus et fonctionnalités avancées de transformation

### Headers de sécurité requis
```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
X-Request-ID: <unique_request_id>
X-Client-Version: <client_version>
```

## Endpoints API

### Health & Monitoring

#### GET /health
**Description** : Vérification de l'état du service de stockage

**Permissions** : Aucune

**Réponse** :
```json
{
  "status": "UP",
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "support-storage-service",
  "version": "1.0.0",
  "checks": {
    "azure_blob": {
      "status": "UP",
      "responseTime": "45ms",
      "available_space": "95%"
    },
    "aws_s3": {
      "status": "UP",
      "responseTime": "67ms",
      "available_space": "98%"
    },
    "processing_queue": {
      "status": "UP",
      "pending_jobs": 15,
      "active_workers": 8
    }
  }
}
```

### File Upload

#### POST /api/v1/storage/upload
**Description** : Upload de fichier avec support multipart et resumable

**Permissions** : user, premium, admin

**Requête** (multipart/form-data) :
```
Content-Type: multipart/form-data
Content-Length: 1048576

--boundary123
Content-Disposition: form-data; name="file"; filename="document.pdf"
Content-Type: application/pdf

[binary data]
--boundary123
Content-Disposition: form-data; name="metadata"

{
  "project_id": "proj_123456789",
  "description": "Document principal du projet",
  "tags": ["document", "source"],
  "is_public": false,
  "auto_transform": true
}
--boundary123--
```

**Réponse** :
```json
{
  "file": {
    "id": "file_123456789",
    "original_name": "document.pdf",
    "file_path": "users/user_123/projects/proj_123/document_abc123.pdf",
    "file_type": "document",
    "mime_type": "application/pdf",
    "size_bytes": 1048576,
    "checksum_md5": "d41d8cd98f00b204e9800998ecf8427e",
    "checksum_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "storage_provider": "azure_blob",
    "is_public": false,
    "created_at": "2024-01-15T10:30:00Z"
  },
  "upload": {
    "status": "completed",
    "upload_time_ms": 2500,
    "transfer_rate_mbps": 3.2
  },
  "urls": {
    "download": "https://api.visiobook.com/api/v1/storage/download/file_123456789",
    "thumbnail": "https://api.visiobook.com/api/v1/storage/download/file_123456789_thumb"
  }
}
```

#### POST /api/v1/storage/upload/resumable/init
**Description** : Initialisation d'un upload resumable pour gros fichiers

**Permissions** : user, premium, admin

**Requête** :
```json
{
  "filename": "large_video.mp4",
  "file_size": 104857600,
  "mime_type": "video/mp4",
  "chunk_size": 1048576,
  "project_id": "proj_123456789",
  "metadata": {
    "description": "Vidéo source pour animation",
    "tags": ["video", "source"]
  }
}
```

**Réponse** :
```json
{
  "upload_session": {
    "session_id": "upload_session_abc123",
    "upload_url": "https://api.visiobook.com/api/v1/storage/upload/resumable/chunk",
    "chunk_size": 1048576,
    "total_chunks": 100,
    "expires_at": "2024-01-15T12:30:00Z"
  },
  "file": {
    "id": "file_pending_123",
    "status": "uploading",
    "progress": 0
  }
}
```

#### PUT /api/v1/storage/upload/resumable/chunk
**Description** : Upload d'un chunk pour upload resumable

**Permissions** : user, premium, admin

**Headers** :
```http
X-Upload-Session: upload_session_abc123
X-Chunk-Index: 5
X-Chunk-Checksum: d41d8cd98f00b204e9800998ecf8427e
Content-Type: application/octet-stream
Content-Length: 1048576
```

**Réponse** :
```json
{
  "chunk": {
    "index": 5,
    "status": "uploaded",
    "checksum_verified": true
  },
  "upload": {
    "progress_percent": 5,
    "chunks_uploaded": 5,
    "chunks_remaining": 95,
    "estimated_completion": "2024-01-15T10:45:00Z"
  }
}
```

### File Management

#### GET /api/v1/storage/files
**Description** : Liste des fichiers de l'utilisateur avec filtres

**Permissions** : user, premium, admin

**Paramètres de requête** :
```
?project_id=proj_123456789
&file_type=image
&page=1
&limit=20
&sort=created_at
&order=desc
&search=document
```

**Réponse** :
```json
{
  "files": [
    {
      "id": "file_123456789",
      "original_name": "image.jpg",
      "file_type": "image",
      "mime_type": "image/jpeg",
      "size_bytes": 524288,
      "is_public": false,
      "created_at": "2024-01-15T10:30:00Z",
      "urls": {
        "download": "https://api.visiobook.com/api/v1/storage/download/file_123456789",
        "thumbnail": "https://api.visiobook.com/api/v1/storage/download/file_123456789_thumb"
      },
      "metadata": {
        "dimensions": {
          "width": 1920,
          "height": 1080
        },
        "exif": {
          "camera": "Canon EOS R5",
          "iso": 400,
          "aperture": "f/2.8"
        }
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 156,
    "pages": 8
  },
  "storage_usage": {
    "used_bytes": 1073741824,
    "quota_bytes": 5368709120,
    "usage_percent": 20
  }
}
```

#### GET /api/v1/storage/files/{file_id}
**Description** : Détails d'un fichier spécifique

**Permissions** : user, premium, admin (propriétaire de la ressource ou admin)

**Réponse** :
```json
{
  "file": {
    "id": "file_123456789",
    "user_id": "user_123456789",
    "project_id": "proj_123456789",
    "original_name": "document.pdf",
    "file_path": "users/user_123/projects/proj_123/document_abc123.pdf",
    "file_type": "document",
    "mime_type": "application/pdf",
    "size_bytes": 1048576,
    "checksum_md5": "d41d8cd98f00b204e9800998ecf8427e",
    "storage_provider": "azure_blob",
    "storage_tier": "hot",
    "is_public": false,
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  },
  "metadata": {
    "document": {
      "pages": 25,
      "language": "fr",
      "author": "John Doe",
      "title": "Mon Document"
    },
    "processing": {
      "text_extracted": true,
      "thumbnails_generated": true,
      "searchable": true
    }
  },
  "urls": {
    "download": "https://api.visiobook.com/api/v1/storage/download/file_123456789",
    "preview": "https://api.visiobook.com/api/v1/storage/download/file_123456789_preview",
    "thumbnail": "https://api.visiobook.com/api/v1/storage/download/file_123456789_thumb"
  },
  "transformations": [
    {
      "id": "transform_123",
      "type": "thumbnail_generation",
      "status": "completed",
      "output_file_id": "file_thumb_123"
    },
    {
      "id": "transform_124",
      "type": "text_extraction",
      "status": "completed",
      "extracted_text_length": 15420
    }
  ]
}
```

#### PUT /api/v1/storage/files/{file_id}
**Description** : Mise à jour des métadonnées d'un fichier

**Permissions** : user, premium, admin (propriétaire de la ressource ou admin)

**Requête** :
```json
{
  "original_name": "nouveau_nom.pdf",
  "is_public": true,
  "metadata": {
    "description": "Document mis à jour",
    "tags": ["document", "final", "public"],
    "custom_fields": {
      "department": "Marketing",
      "version": "2.1"
    }
  }
}
```

**Réponse** :
```json
{
  "file": {
    "id": "file_123456789",
    "original_name": "nouveau_nom.pdf",
    "is_public": true,
    "updated_at": "2024-01-15T11:30:00Z"
  },
  "message": "File metadata updated successfully"
}
```

#### DELETE /api/v1/storage/files/{file_id}
**Description** : Suppression d'un fichier

**Permissions** : user, premium, admin (propriétaire de la ressource ou admin)

**Paramètres de requête** :
```
?permanent=false
```

**Réponse** :
```json
{
  "file_id": "file_123456789",
  "deleted": true,
  "permanent": false,
  "recovery_deadline": "2024-02-14T10:30:00Z",
  "storage_freed_bytes": 1048576,
  "message": "File moved to trash. Can be recovered within 30 days."
}
```

### File Download

#### GET /api/v1/storage/download/{file_id}
**Description** : Téléchargement direct d'un fichier

**Permissions** : user, premium, admin (propriétaire de la ressource ou admin)

**Paramètres de requête** :
```
?disposition=attachment
&filename=custom_name.pdf
```

**Réponse** : Redirection 302 vers URL signée ou stream direct
```http
HTTP/1.1 302 Found
Location: https://visiobookstorage.blob.core.windows.net/files/signed-url-with-sas-token
Cache-Control: public, max-age=3600
Content-Disposition: attachment; filename="document.pdf"
```

#### GET /api/v1/storage/stream/{file_id}
**Description** : Streaming d'un fichier média avec support Range

**Permissions** : user, premium, admin (propriétaire de la ressource ou admin)

**Headers de requête** :
```http
Range: bytes=0-1023
Accept-Ranges: bytes
```

**Réponse** :
```http
HTTP/1.1 206 Partial Content
Content-Range: bytes 0-1023/1048576
Content-Length: 1024
Content-Type: video/mp4
Accept-Ranges: bytes

[binary data]
```

### File Transformation

#### POST /api/v1/storage/transform
**Description** : Demande de transformation d'un fichier

**Permissions** : user, premium, admin

**Requête** :
```json
{
  "source_file_id": "file_123456789",
  "transformation_type": "image_resize",
  "parameters": {
    "width": 800,
    "height": 600,
    "quality": 85,
    "format": "jpeg",
    "maintain_aspect_ratio": true
  },
  "output_name": "image_resized.jpg"
}
```

**Réponse** :
```json
{
  "transformation": {
    "id": "transform_123456789",
    "source_file_id": "file_123456789",
    "transformation_type": "image_resize",
    "status": "queued",
    "progress_percent": 0,
    "estimated_completion": "2024-01-15T10:35:00Z",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "queue_position": 3
}
```

#### GET /api/v1/storage/transform/{transform_id}
**Description** : Statut d'une transformation

**Permissions** : user, premium, admin

**Réponse** :
```json
{
  "transformation": {
    "id": "transform_123456789",
    "source_file_id": "file_123456789",
    "target_file_id": "file_987654321",
    "transformation_type": "image_resize",
    "status": "completed",
    "progress_percent": 100,
    "started_at": "2024-01-15T10:32:00Z",
    "completed_at": "2024-01-15T10:33:30Z",
    "processing_time_ms": 90000
  },
  "output_file": {
    "id": "file_987654321",
    "original_name": "image_resized.jpg",
    "size_bytes": 245760,
    "urls": {
      "download": "https://api.visiobook.com/api/v1/storage/download/file_987654321"
    }
  }
}
```

## Flux de transactions CRUD

### Diagramme de séquence - Upload de fichier

```mermaid
sequenceDiagram
    participant Client as Web Client
    participant API as Storage API
    participant Validator as File Validator
    participant Azure as Azure Blob
    participant S3 as AWS S3 Backup
    participant DB as Database

    Client->>API: POST /api/v1/storage/upload
    API->>Validator: Validate file type & size
    Validator-->>API: Validation passed
    API->>Azure: Upload to primary storage
    Azure-->>API: Upload successful
    API->>S3: Backup to secondary storage
    S3-->>API: Backup successful
    API->>DB: Store file metadata
    DB-->>API: Metadata stored
    API-->>Client: Return file info + URLs
```

### Diagramme de séquence - Transformation de fichier

```mermaid
sequenceDiagram
    participant Client as Web Client
    participant API as Storage API
    participant Queue as Processing Queue
    participant Worker as Transform Worker
    participant Storage as File Storage
    participant DB as Database

    Client->>API: POST /api/v1/storage/transform
    API->>DB: Create transformation record
    DB-->>API: Record created
    API->>Queue: Add to processing queue
    Queue-->>API: Job queued
    API-->>Client: Return transformation ID

    Queue->>Worker: Assign transformation job
    Worker->>Storage: Download source file
    Storage-->>Worker: File downloaded
    Worker->>Worker: Process transformation
    Worker->>Storage: Upload transformed file
    Storage-->>Worker: Upload successful
    Worker->>DB: Update transformation status
    Worker->>Queue: Mark job complete
```

### Diagramme de flux - Gestion du stockage

```mermaid
flowchart TD
    Start([File Upload Request]) --> ValidateAuth{Valid Authentication?}
    ValidateAuth -->|No| AuthError[Return 401 Unauthorized]
    ValidateAuth -->|Yes| ValidateFile{Valid File?}

    ValidateFile -->|No| FileError[Return 400 Bad Request]
    ValidateFile -->|Yes| CheckQuota{Within Quota?}

    CheckQuota -->|No| QuotaError[Return 413 Quota Exceeded]
    CheckQuota -->|Yes| ProcessUpload[Process Upload]

    ProcessUpload --> UploadPrimary[Upload to Primary Storage]
    UploadPrimary --> CheckBackup{Backup Required?}

    CheckBackup -->|Yes| UploadBackup[Upload to Backup Storage]
    CheckBackup -->|No| StoreMetadata[Store Metadata]
    UploadBackup --> StoreMetadata

    StoreMetadata --> AutoTransform{Auto Transform?}

    AutoTransform -->|Yes| QueueTransform[Queue Transformations]
    AutoTransform -->|No| Success[Return Success]
    QueueTransform --> Success

    AuthError --> End([End])
    FileError --> End
    QuotaError --> End
    Success --> End
```

## Parcours utilisateurs détaillés

### Milestone 1: Importer un contenu

**US 1.1 - Import de fichiers**
```
1. POST /api/v1/storage/upload
   - Upload du fichier source (PDF, TXT)
   - Validation du type et de la taille
   - Stockage sécurisé avec backup

2. POST /api/v1/storage/transform (automatique)
   - Extraction de texte pour PDF
   - Génération de thumbnails
   - Optimisation pour traitement IA
```

**US 1.2 - Scan de texte (OCR)**
```
1. POST /api/v1/storage/upload
   - Upload de l'image scannée
   - Validation du format image

2. POST /api/v1/storage/transform
   - Transformation OCR pour extraction de texte
   - Génération du fichier texte résultant
```

### Milestone 2: Personnaliser le style de l'animation

**US 2.1 - Choix du style graphique**
```
1. GET /api/v1/storage/files
   - Récupération des assets de style disponibles
   - Filtrage par type "style_template"

2. GET /api/v1/storage/download/{file_id}
   - Téléchargement des previews de styles
   - Affichage direct via le service de stockage
```

### Milestone 3: Générer et visualiser une animation

**US 3.1 - Génération automatique**
```
1. POST /api/v1/storage/transform (multiple)
   - Génération des images de scènes
   - Création des animations intermédiaires
   - Assemblage de la vidéo finale

2. GET /api/v1/storage/transform/{transform_id}
   - Suivi de progression de génération
   - Notification de fin de traitement
```

**US 3.2 - Visualisation dans l'application**
```
1. GET /api/v1/storage/stream/{file_id}
   - Streaming de la vidéo générée
   - Support du Range pour lecture progressive

2. GET /api/v1/storage/files/{file_id}
   - Récupération des métadonnées de visualisation
   - Statistiques d'accès aux fichiers
```

### Milestone 4: Exporter et partager l'animation

**US 4.1 - Télécharger la vidéo**
```
1. GET /api/v1/storage/download/{file_id}
   - Téléchargement de la vidéo finale
   - Choix du format et de la qualité

2. POST /api/v1/storage/transform (si format différent)
   - Conversion vers le format demandé
   - Optimisation pour la plateforme cible
```

**US 4.3 - Choix du format d'export**
```
1. POST /api/v1/storage/transform
   - Conversion MP4, GIF, MOV selon demande
   - Optimisation des paramètres par format

2. GET /api/v1/storage/transform/{transform_id}
   - Suivi de la conversion
   - Notification de disponibilité
```

## Codes d'erreur

| Code | Message | Description |
|------|---------|-------------|
| 200 | Success | Opération réussie |
| 201 | Created | Fichier créé avec succès |
| 202 | Accepted | Transformation en cours de traitement |
| 206 | Partial Content | Contenu partiel (streaming) |
| 400 | Bad Request | Requête malformée ou fichier invalide |
| 401 | Unauthorized | Token JWT manquant ou invalide |
| 403 | Forbidden | Permissions insuffisantes sur le fichier |
| 404 | Not Found | Fichier non trouvé |
| 409 | Conflict | Fichier déjà existant avec même checksum |
| 413 | Payload Too Large | Fichier trop volumineux ou quota dépassé |
| 415 | Unsupported Media Type | Type de fichier non supporté |
| 422 | Unprocessable Entity | Fichier corrompu ou métadonnées invalides |
| 429 | Too Many Requests | Limite de taux d'upload dépassée |
| 500 | Internal Server Error | Erreur interne du serveur |
| 502 | Bad Gateway | Erreur de connexion au stockage |
| 503 | Service Unavailable | Service de stockage temporairement indisponible |
| 507 | Insufficient Storage | Espace de stockage insuffisant |

### Format d'erreur standardisé
```json
{
  "error": {
    "code": "VISIOBOOK_FILE_TOO_LARGE",
    "message": "File size exceeds maximum allowed limit",
    "details": {
      "file_size_bytes": 104857600,
      "max_size_bytes": 52428800,
      "user_quota_remaining": 1073741824
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789",
    "service": "support-storage-service"
  }
}
```

### Codes d'erreur spécifiques
```json
{
  "VISIOBOOK_FILE_TYPE_INVALID": "File type not supported",
  "VISIOBOOK_FILE_TOO_LARGE": "File size exceeds maximum allowed limit",
  "VISIOBOOK_QUOTA_EXCEEDED": "User storage quota exceeded",
  "VISIOBOOK_UPLOAD_FAILED": "File upload operation failed",
  "VISIOBOOK_TRANSFORMATION_FAILED": "File transformation failed",
  "VISIOBOOK_RESOURCE_NOT_FOUND": "File not found",
  "VISIOBOOK_INSUFFICIENT_PERMISSIONS": "User lacks required permissions",
  "VISIOBOOK_SERVICE_UNAVAILABLE": "Backend storage service unavailable"
}
```

## Versioning et migration

### Convention v1
- **URL Base** : `/api/v1/`
- **Headers** : `Accept: application/vnd.visiobook.v1+json`
- **Rétrocompatibilité** : Maintenue pendant 12 mois minimum

### Stratégie de migration
1. **Fichiers existants** : Migration transparente des métadonnées
2. **URLs de téléchargement** : Redirection automatique vers nouvelles URLs
3. **Transformations** : Retraitement automatique si nécessaire
4. **Cache** : Mise à jour progressive des caches applicatifs

### Exemple de migration v1 → v2
```json
{
  "migration": {
    "from_version": "v1",
    "to_version": "v2",
    "changes": [
      {
        "type": "endpoint_enhanced",
        "endpoint": "/api/v2/storage/upload",
        "new_features": ["progressive_upload", "auto_optimization"]
      },
      {
        "type": "field_added",
        "endpoint": "/api/v2/storage/files",
        "field": "ai_analysis_metadata"
      }
    ],
    "breaking_changes": false,
    "migration_deadline": "2024-12-31"
  }
}
```

### Gestion des formats de fichiers
```json
{
  "supported_formats": {
    "images": ["jpeg", "png", "gif", "webp", "svg"],
    "videos": ["mp4", "mov", "avi", "webm"],
    "documents": ["pdf", "txt", "docx", "rtf"],
    "audio": ["mp3", "wav", "aac", "ogg"]
  },
  "transformation_matrix": {
    "image_to_image": ["resize", "format_conversion", "compression"],
    "video_to_video": ["resize", "format_conversion", "bitrate_adjustment"],
    "document_to_text": ["ocr", "text_extraction"],
    "any_to_thumbnail": ["thumbnail_generation"]
  }
}
```
