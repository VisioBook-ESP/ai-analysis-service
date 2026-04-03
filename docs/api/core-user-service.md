# API Documentation - Core User Service

## Vue d'ensemble du service

### Rôle et responsabilités
Le **Core User Service** est responsable de la gestion complète des utilisateurs, incluant l'inscription, l'authentification, la gestion des profils, des sessions et des permissions. Il constitue le pilier de sécurité de l'écosystème Visiobook.

### Justification de l'atomisation
- **Sécurité centralisée** : Isolation des données sensibles d'authentification
- **Réutilisabilité** : Service utilisé par tous les autres microservices
- **Conformité RGPD** : Gestion centralisée des données personnelles
- **Scalabilité** : Gestion indépendante des pics de connexions

### Informations techniques
- **Port** : 8081
- **Technology Stack** : Python 3.11 + FastAPI + SQLAlchemy + Pydantic
- **Authentication** : JWT + Password Authentication
- **Database** : PostgreSQL + Redis (sessions)
- **Version API** : v1

## Architecture du service

```mermaid
graph TB
    subgraph "Core User Service"
        API[API Layer<br/>FastAPI + Pydantic]
        AUTH[Authentication Module<br/>JWT + Password]
        PROFILE[Profile Management<br/>SQLAlchemy Models]
        SESSION[Session Manager<br/>Redis Sessions]
        SECURITY[Security Module<br/>Password Policy]
        NOTIFICATION[Notification Client<br/>Email]
    end

    subgraph "External Dependencies"
        DB[(PostgreSQL<br/>User Data)]
        REDIS[(Redis<br/>Sessions + Cache)]
        EMAIL[Email Service<br/>SendGrid/SMTP]
    end

    API --> AUTH
    API --> PROFILE
    API --> SESSION
    API --> SECURITY
    AUTH --> DB
    AUTH --> REDIS
    SESSION --> REDIS
    PROFILE --> DB
    SECURITY --> NOTIFICATION
    NOTIFICATION --> EMAIL

    classDef service fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef external fill:#fff8e1,stroke:#f9a825,stroke-width:2px

    class API,AUTH,PROFILE,SESSION,SECURITY,NOTIFICATION service
    class DB,REDIS,EMAIL external
```

### Schémas de base de données

#### Note sur l'architecture de données

> **🏗️ Responsabilité de ce service (Phase actuelle)**
>
> Le Core User Service est **propriétaire** de toutes les données utilisateur. Les autres microservices (Project, Storage, AI) mockent localement les données utilisateur nécessaires à leur fonctionnement.
>
> **🎯 Migration future**
>
> Lors de la centralisation via le Core Database Service, ce service fournira les migrations de référence pour toutes les tables liées aux utilisateurs.

#### PostgreSQL - Tables propriétaires

```sql
-- Users table (PROPRIÉTAIRE - Core User Service)
-- Cette table est la source de vérité pour toutes les données utilisateur
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'user',
    subscription_type VARCHAR(50) DEFAULT 'free',
    email_verified BOOLEAN DEFAULT FALSE,
    phone_number VARCHAR(20),
    phone_verified BOOLEAN DEFAULT FALSE,
    -- MFA fields - NON UTILISÉES DANS MVP
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    last_login_at TIMESTAMP,
    login_count INTEGER DEFAULT 0,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_subscription ON users(subscription_type);
CREATE INDEX idx_users_created_at ON users(created_at);

-- User profiles table
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    avatar_url VARCHAR(500),
    bio TEXT,
    language VARCHAR(10) DEFAULT 'fr',
    timezone VARCHAR(50) DEFAULT 'Europe/Paris',
    preferences JSONB DEFAULT '{}',
    privacy_settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);

-- User sessions table
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires ON user_sessions(expires_at);

-- OAuth connections table - NON UTILISÉE DANS MVP
-- CREATE TABLE oauth_connections (
--     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
--     user_id UUID REFERENCES users(id) ON DELETE CASCADE,
--     provider VARCHAR(50) NOT NULL,
--     provider_user_id VARCHAR(255) NOT NULL,
--     access_token TEXT,
--     refresh_token TEXT,
--     expires_at TIMESTAMP,
--     created_at TIMESTAMP DEFAULT NOW(),
--     updated_at TIMESTAMP DEFAULT NOW(),
--     UNIQUE(provider, provider_user_id)
-- );

-- CREATE INDEX idx_oauth_connections_user_id ON oauth_connections(user_id);
-- CREATE INDEX idx_oauth_connections_provider ON oauth_connections(provider);
```

### Variables d'environnement

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/visiobook
REDIS_URL=redis://localhost:6379

# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key
JWT_EXPIRES_IN=900
JWT_REFRESH_EXPIRES_IN=604800

# OAuth Providers - NON UTILISÉES DANS MVP
# GOOGLE_CLIENT_ID=your-google-client-id
# GOOGLE_CLIENT_SECRET=your-google-client-secret
# GITHUB_CLIENT_ID=your-github-client-id
# GITHUB_CLIENT_SECRET=your-github-client-secret

# Email Service
SENDGRID_API_KEY=your-sendgrid-api-key
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password

# SMS Service - NON UTILISÉ DANS MVP
# TWILIO_ACCOUNT_SID=your-twilio-sid
# TWILIO_AUTH_TOKEN=your-twilio-token
# TWILIO_PHONE_NUMBER=+1234567890

# Security
BCRYPT_ROUNDS=12
# MFA_ISSUER=Visiobook - NON UTILISÉ DANS MVP
PASSWORD_MIN_LENGTH=8
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION=900

# Rate Limiting
RATE_LIMIT_WINDOW=900000
RATE_LIMIT_MAX=100
```

## Authentification et sécurité

> **📋 Référence** : Voir [Règles Communes](./regles_communes.md) pour les standards d'authentification, permissions et sécurité.

### Système JWT
```json
{
  "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "Content-Type": "application/json",
  "X-Refresh-Token": "refresh_token_here"
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
- **admin** : Accès complet à tous les utilisateurs et données système
- **user** : Accès aux fonctionnalités de base et ses propres données
- **premium** : Accès étendu avec fonctionnalités avancées et quotas augmentés

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
**Description** : Vérification de l'état du service utilisateur

**Permissions** : Aucune

**Réponse** :
```json
{
  "status": "UP",
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "core-user-service",
  "version": "1.0.0",
  "checks": {
    "database": {
      "status": "UP",
      "responseTime": "12ms"
    },
    "redis": {
      "status": "UP",
      "responseTime": "3ms"
    },
    "email_service": {
      "status": "UP",
      "responseTime": "45ms"
    }
  }
}
```

### Authentication

#### POST /api/v1/auth/register
**Description** : Inscription d'un nouvel utilisateur

**Permissions** : Aucune

**Requête** :
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "username": "johndoe",
  "first_name": "John",
  "last_name": "Doe",
  "language": "fr",
  "accept_terms": true,
  "marketing_consent": false
}
```

**Réponse** :
```json
{
  "user": {
    "id": "user_123456789",
    "email": "user@example.com",
    "username": "johndoe",
    "first_name": "John",
    "last_name": "Doe",
    "role": "user",
    "subscription_type": "free",
    "email_verified": false,
    "created_at": "2024-01-15T10:30:00Z"
  },
  "tokens": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "refresh_token_here",
    "expires_in": 900,
    "token_type": "Bearer"
  },
  "verification": {
    "email_sent": true,
    "verification_required": true
  }
}
```

#### POST /api/v1/auth/login
**Description** : Connexion utilisateur

**Permissions** : Aucune

**Requête** :
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "remember_me": true,
  "device_info": {
    "device_type": "web",
    "browser": "Chrome",
    "os": "Windows 10"
  }
}
```

**Réponse** :
```json
{
  "user": {
    "id": "user_123456789",
    "email": "user@example.com",
    "username": "johndoe",
    "first_name": "John",
    "last_name": "Doe",
    "role": "user",
    "subscription_type": "premium",
    "last_login_at": "2024-01-15T10:30:00Z"
  },
  "tokens": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "refresh_token_here",
    "expires_in": 900,
    "token_type": "Bearer"
  },
  "session": {
    "session_id": "session_987654321",
    "expires_at": "2024-01-22T10:30:00Z"
  }
}
```

#### POST /api/v1/auth/logout
**Description** : Déconnexion utilisateur

**Permissions** : user, premium, admin

**Requête** :
```json
{
  "logout_all_devices": false
}
```

**Réponse** :
```json
{
  "message": "Successfully logged out",
  "sessions_terminated": 1,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /api/v1/auth/refresh
**Description** : Renouvellement du token d'accès

**Permissions** : Aucune (refresh token requis)

**Requête** :
```json
{
  "refresh_token": "refresh_token_here"
}
```

**Réponse** :
```json
{
  "tokens": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "new_refresh_token_here",
    "expires_in": 900,
    "token_type": "Bearer"
  }
}
```

### User Management

#### GET /api/v1/users/profile
**Description** : Récupération du profil utilisateur

**Permissions** : user, premium, admin

**Réponse** :
```json
{
  "user": {
    "id": "user_123456789",
    "email": "user@example.com",
    "username": "johndoe",
    "first_name": "John",
    "last_name": "Doe",
    "role": "user",
    "subscription_type": "premium",
    "email_verified": true,
    "phone_verified": false,
    "mfa_enabled": true,
    "created_at": "2024-01-01T00:00:00Z",
    "last_login_at": "2024-01-15T10:30:00Z"
  },
  "profile": {
    "avatar_url": "https://cdn.visiobook.com/avatars/user_123.jpg",
    "bio": "Créateur de contenu passionné",
    "language": "fr",
    "timezone": "Europe/Paris",
    "preferences": {
      "theme": "dark",
      "notifications": {
        "email": true,
        "push": true,
        "sms": false
      },
      "privacy": {
        "profile_public": false,
        "show_activity": true
      }
    }
  },
  "statistics": {
    "projects_count": 15,
    "total_generations": 45,
    "storage_used_mb": 1250
  }
}
```

#### PUT /api/v1/users/profile
**Description** : Mise à jour du profil utilisateur

**Permissions** : user, premium, admin

**Requête** :
```json
{
  "first_name": "John",
  "last_name": "Smith",
  "username": "johnsmith",
  "bio": "Créateur de contenu et storyteller",
  "language": "en",
  "timezone": "America/New_York",
  "preferences": {
    "theme": "light",
    "notifications": {
      "email": true,
      "push": false,
      "sms": true
    }
  }
}
```

**Réponse** :
```json
{
  "user": {
    "id": "user_123456789",
    "email": "user@example.com",
    "username": "johnsmith",
    "first_name": "John",
    "last_name": "Smith",
    "updated_at": "2024-01-15T10:30:00Z"
  },
  "message": "Profile updated successfully"
}
```

#### POST /api/v1/users/change-password
**Description** : Changement de mot de passe

**Permissions** : user, premium, admin

**Requête** :
```json
{
  "current_password": "OldPassword123!",
  "new_password": "NewSecurePassword456!",
  "confirm_password": "NewSecurePassword456!"
}
```

**Réponse** :
```json
{
  "message": "Password changed successfully",
  "security_alert_sent": true,
  "sessions_invalidated": 3
}
```

### Session Management

#### GET /api/v1/users/sessions
**Description** : Liste des sessions actives

**Permissions** : user, premium, admin

**Réponse** :
```json
{
  "sessions": [
    {
      "id": "session_123456789",
      "device_info": {
        "device_type": "web",
        "browser": "Chrome",
        "os": "Windows 10"
      },
      "ip_address": "192.168.1.100",
      "location": "Paris, France",
      "is_current": true,
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity": "2024-01-15T11:45:00Z",
      "expires_at": "2024-01-22T10:30:00Z"
    },
    {
      "id": "session_987654321",
      "device_info": {
        "device_type": "mobile",
        "browser": "Safari",
        "os": "iOS 17"
      },
      "ip_address": "192.168.1.101",
      "location": "Paris, France",
      "is_current": false,
      "created_at": "2024-01-14T15:20:00Z",
      "last_activity": "2024-01-15T09:15:00Z",
      "expires_at": "2024-01-21T15:20:00Z"
    }
  ],
  "total_sessions": 2
}
```

#### DELETE /api/v1/users/sessions/{session_id}
**Description** : Suppression d'une session spécifique

**Permissions** : user, premium, admin

**Réponse** :
```json
{
  "message": "Session terminated successfully",
  "session_id": "session_987654321"
}
```

### Account Management

#### POST /api/v1/users/verify-email
**Description** : Vérification de l'adresse email

**Permissions** : Aucune

**Requête** :
```json
{
  "verification_token": "email_verification_token_here"
}
```

**Réponse** :
```json
{
  "email_verified": true,
  "message": "Email successfully verified",
  "bonus_credits": 100
}
```

#### POST /api/v1/users/resend-verification
**Description** : Renvoi de l'email de vérification

**Permissions** : user, premium, admin

**Réponse** :
```json
{
  "email_sent": true,
  "message": "Verification email sent successfully",
  "expires_at": "2024-01-15T11:30:00Z"
}
```

#### POST /api/v1/users/forgot-password
**Description** : Demande de réinitialisation de mot de passe

**Permissions** : Aucune

**Requête** :
```json
{
  "email": "user@example.com"
}
```

**Réponse** :
```json
{
  "message": "Password reset email sent if account exists",
  "email_sent": true
}
```

#### POST /api/v1/users/reset-password
**Description** : Réinitialisation du mot de passe

**Permissions** : Aucune

**Requête** :
```json
{
  "reset_token": "password_reset_token_here",
  "new_password": "NewSecurePassword123!",
  "confirm_password": "NewSecurePassword123!"
}
```

**Réponse** :
```json
{
  "password_reset": true,
  "message": "Password successfully reset",
  "auto_login": true,
  "tokens": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "refresh_token_here",
    "expires_in": 900
  }
}
```

#### DELETE /api/v1/users/account
**Description** : Suppression du compte utilisateur

**Permissions** : user, premium, admin

**Requête** :
```json
{
  "password": "CurrentPassword123!",
  "confirmation": "DELETE_MY_ACCOUNT",
  "reason": "No longer needed"
}
```

**Réponse** :
```json
{
  "account_deleted": true,
  "data_retention_days": 30,
  "message": "Account successfully deleted. Data will be permanently removed in 30 days."
}
```

## Flux de transactions CRUD

### Diagramme de séquence - Inscription utilisateur

```mermaid
sequenceDiagram
    participant Client as Web Client
    participant API as User API
    participant DB as Database
    participant Redis as Redis Cache
    participant Email as Email Service

    Client->>API: POST /api/v1/auth/register
    API->>API: Validate input data
    API->>DB: Check email uniqueness
    DB-->>API: Email available
    API->>API: Hash password (bcrypt)
    API->>DB: Insert user record
    DB-->>API: User created
    API->>API: Generate JWT tokens
    API->>Redis: Store session
    API->>API: Generate email verification token
    API->>Email: Send verification email
    Email-->>API: Email sent
    API-->>Client: Return user + tokens
```

### Diagramme de séquence - Authentification simple

```mermaid
sequenceDiagram
    participant Client as Web Client
    participant API as User API
    participant DB as Database
    participant Redis as Redis Cache

    Client->>API: POST /api/v1/auth/login
    API->>DB: Verify credentials
    DB-->>API: Credentials valid
    API->>API: Generate JWT tokens
    API->>Redis: Store session
    API->>DB: Update last login
    API-->>Client: Return tokens + user data
```

### Diagramme de flux - Gestion des sessions

```mermaid
flowchart TD
    Start([User Login]) --> ValidateCreds{Validate Credentials}
    ValidateCreds -->|Invalid| LoginFail[Login Failed]
    ValidateCreds -->|Valid| CreateSession[Create Session]

    CreateSession --> GenerateTokens[Generate JWT Tokens]
    GenerateTokens --> StoreSession[Store in Redis]
    StoreSession --> UpdateLastLogin[Update Last Login]
    UpdateLastLogin --> Success[Login Success]

    LoginFail --> IncrementAttempts[Increment Failed Attempts]
    IncrementAttempts --> CheckLockout{Max Attempts?}
    CheckLockout -->|Yes| LockAccount[Lock Account]
    CheckLockout -->|No| End([End])

    LockAccount --> End
    Success --> End
```

## Parcours utilisateurs détaillés

### Milestone 1: Importer un contenu

**US 1.1 - Import de fichiers (Authentification requise)**
```
1. GET /api/v1/users/profile
   - Vérifier les permissions utilisateur
   - Récupérer les quotas et limites

2. Validation des droits d'upload
   - Vérifier le type d'abonnement
   - Contrôler les limites de stockage
```

### Milestone 2: Personnaliser le style de l'animation

**US 2.1 - Choix du style graphique (Préférences utilisateur)**
```
1. GET /api/v1/users/profile
   - Récupérer les préférences de style
   - Obtenir l'historique des choix

2. PUT /api/v1/users/profile
   - Sauvegarder les nouvelles préférences
   - Mettre à jour le profil utilisateur
```

### Milestone 5: Historique et réutilisation

**US 5.1 - Accéder à l'historique (Gestion des permissions)**
```
1. GET /api/v1/users/profile
   - Vérifier l'identité utilisateur
   - Récupérer les métadonnées de profil

2. Validation des droits d'accès
   - Contrôler les permissions sur les projets
   - Appliquer les filtres de confidentialité
```

**US P5.1 - Création de compte**
```
1. POST /api/v1/auth/register
   - Créer le compte utilisateur
   - Générer les tokens d'authentification
   - Envoyer l'email de vérification

2. POST /api/v1/users/verify-email
   - Vérifier l'adresse email
   - Activer le compte
   - Débloquer les fonctionnalités
```

**US P5.2 - Abonnement premium**
```
1. GET /api/v1/users/profile
   - Vérifier le statut d'abonnement actuel
   - Récupérer les informations de facturation

2. PUT /api/v1/users/profile (après paiement)
   - Mettre à jour le type d'abonnement
   - Activer les fonctionnalités premium
   - Mettre à jour les quotas
```

## Codes d'erreur

| Code | Message | Description |
|------|---------|-------------|
| 200 | Success | Opération réussie |
| 201 | Created | Utilisateur créé avec succès |
| 400 | Bad Request | Données d'entrée invalides |
| 401 | Unauthorized | Token JWT manquant ou invalide |
| 403 | Forbidden | Permissions insuffisantes |
| 404 | Not Found | Utilisateur non trouvé |
| 409 | Conflict | Email ou username déjà utilisé |
| 422 | Unprocessable Entity | Validation des données échouée |
| 423 | Locked | Compte verrouillé (trop de tentatives) |
| 429 | Too Many Requests | Limite de taux dépassée |
| 500 | Internal Server Error | Erreur interne du serveur |
| 503 | Service Unavailable | Service temporairement indisponible |

### Format d'erreur standardisé
```json
{
  "error": {
    "code": "VISIOBOOK_INVALID_CREDENTIALS",
    "message": "Invalid email or password",
    "details": {
      "field": "password",
      "attempts_remaining": 3,
      "lockout_duration": 900
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789",
    "service": "core-user-service"
  }
}
```

### Codes d'erreur spécifiques
```json
{
  "VISIOBOOK_INVALID_CREDENTIALS": "Invalid email or password",
  "VISIOBOOK_TOKEN_EXPIRED": "JWT token has expired",
  "VISIOBOOK_TOKEN_INVALID": "JWT token is invalid or malformed",
  "VISIOBOOK_INSUFFICIENT_PERMISSIONS": "User lacks required permissions",
  "VISIOBOOK_VALIDATION_FAILED": "Input validation failed",
  "VISIOBOOK_RESOURCE_NOT_FOUND": "User not found",
  "VISIOBOOK_RESOURCE_CONFLICT": "Email or username already exists",
  "VISIOBOOK_RATE_LIMIT_EXCEEDED": "Rate limit exceeded for this endpoint"
}
```

## Versioning et migration

### Convention v1
- **URL Base** : `/api/v1/`
- **Headers** : `Accept: application/vnd.visiobook.v1+json`
- **Rétrocompatibilité** : Maintenue pendant 12 mois minimum

### Stratégie de migration
1. **Dépréciation** : Annonce 3 mois avant suppression
2. **Coexistence** : v1 et v2 fonctionnent en parallèle pendant 6 mois
3. **Migration automatique** : Scripts de migration des données
4. **Documentation** : Guide de migration détaillé
5. **Support** : Assistance technique pendant la transition

### Processus de migration spécifique
1. **Authentification** : Migration transparente des tokens
2. **Profils utilisateur** : Migration automatique des données
3. **Sessions** : Renouvellement automatique lors de la connexion
4. **OAuth** : Reconnexion requise pour les nouveaux scopes

###
