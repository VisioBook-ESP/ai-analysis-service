# core-payment-service

## Informations generales

| Propriete | Valeur |
|-----------|--------|
| **Repository** | core-payment-service |
| **Port** | 8087 |
| **Stack** | Node.js / NestJS |
| **Phase** | 6 - Services Complementaires |
| **Priorite** | Post-MVP (monetisation) |

## Flows/Journeys concernes

| Flow | Role | Responsabilite |
|------|------|----------------|
| Flow Premium | Owner | Gestion abonnements |
| Flow Quota | Owner | Verification/consommation credits |

## Architecture interne

```mermaid
graph TB
    subgraph "core-payment-service"
        API[NestJS App]

        subgraph "Controllers"
            SUB[SubscriptionController<br/>/subscriptions/*]
            PAY[PaymentController<br/>/payments/*]
            QUOTA[QuotaController<br/>/quotas/*]
            WH[WebhookController<br/>/webhooks/*]
        end

        subgraph "Services"
            SUB_SVC[SubscriptionService]
            PAY_SVC[PaymentService]
            QUOTA_SVC[QuotaService]
        end

        subgraph "Adapters"
            STRIPE[StripeAdapter]
        end
    end

    subgraph "External"
        STRIPE_API[Stripe API]
        PG[(PostgreSQL<br/>TBD)]
        USER[core-user-service]
        NOTIF[core-notification-service]
    end

    API --> SUB
    API --> PAY
    API --> QUOTA
    API --> WH

    SUB --> SUB_SVC
    PAY --> PAY_SVC
    QUOTA --> QUOTA_SVC

    SUB_SVC --> STRIPE
    PAY_SVC --> STRIPE
    STRIPE --> STRIPE_API

    SUB_SVC --> PG
    QUOTA_SVC --> PG
    SUB_SVC --> USER
    PAY_SVC --> NOTIF
```

## Controllers et Endpoints

### SubscriptionController (`/api/v1/subscriptions`)

| Methode | Endpoint | Description | Auth |
|---------|----------|-------------|------|
| GET | `/plans` | Liste des plans disponibles | Non |
| GET | `/current` | Abonnement actuel | Oui |
| POST | `/checkout` | Creer session checkout | Oui |
| POST | `/cancel` | Annuler abonnement | Oui |
| POST | `/upgrade` | Upgrade plan | Oui |
| POST | `/downgrade` | Downgrade plan | Oui |

```typescript
// GET /api/v1/subscriptions/plans
interface Plan {
  id: string;
  name: 'free' | 'premium' | 'enterprise';
  price: number;
  currency: string;
  interval: 'month' | 'year';
  features: PlanFeature[];
  limits: PlanLimits;
}

interface PlanLimits {
  generationsPerMonth: number;
  storageGB: number;
  maxProjectSize: number;
  exportQuality: '720p' | '1080p' | '4k';
  watermark: boolean;
}

// POST /api/v1/subscriptions/checkout
interface CheckoutRequest {
  planId: string;
  successUrl: string;
  cancelUrl: string;
}

interface CheckoutResponse {
  sessionId: string;
  checkoutUrl: string;
}
```

### QuotaController (`/api/v1/quotas`)

| Methode | Endpoint | Description | Auth |
|---------|----------|-------------|------|
| GET | `/` | Quotas utilisateur | Oui |
| POST | `/consume` | Consommer quota | Service-only |
| POST | `/reset` | Reset quotas (admin) | Admin |

```typescript
// GET /api/v1/quotas
interface UserQuota {
  userId: string;
  plan: string;
  generations: {
    used: number;
    limit: number;
    resetDate: string;
  };
  storage: {
    used: number;  // bytes
    limit: number;
  };
}

// POST /api/v1/quotas/consume (internal)
interface ConsumeQuotaRequest {
  userId: string;
  type: 'generation' | 'storage';
  amount: number;
}

interface ConsumeQuotaResponse {
  success: boolean;
  remaining: number;
  error?: string;
}
```

### WebhookController (`/api/v1/webhooks`)

| Methode | Endpoint | Description | Auth |
|---------|----------|-------------|------|
| POST | `/stripe` | Webhook Stripe | Stripe signature |

```typescript
// Stripe webhook events handled
type StripeWebhookEvent =
  | 'checkout.session.completed'
  | 'customer.subscription.created'
  | 'customer.subscription.updated'
  | 'customer.subscription.deleted'
  | 'invoice.paid'
  | 'invoice.payment_failed';
```

## Communications Inter-services

### Appels sortants

```mermaid
graph LR
    PAY[core-payment-service] --> STRIPE[Stripe API]
    PAY --> PG[(PostgreSQL<br/>TBD)]
    PAY --> USER[core-user-service]
    PAY --> NOTIF[core-notification-service]
```

| Service cible | Objectif |
|---------------|----------|
| Stripe API | Traitement paiements |
| core-user-service | Mise a jour tier utilisateur |
| core-notification-service | Emails confirmation |
| PostgreSQL (TBD) | Stockage transactions (connexion directe ORM, voir database-infrastructure.md) |

### Appels entrants

```mermaid
graph RL
    MOBILE[mobile-app] -->|"Checkout"| PAY[core-payment-service]
    WEB[web-user-portal] -->|"Checkout"| PAY
    PROJECT[core-project-service] -->|"Check quota"| PAY
```

## Diagramme de sequence: Checkout Stripe

```mermaid
sequenceDiagram
    participant C as Client
    participant PS as core-payment-service
    participant STRIPE as Stripe API
    participant PG as PostgreSQL
    participant US as core-user-service
    participant NS as Notification Service

    C->>PS: POST /subscriptions/checkout<br/>{ planId: "premium" }

    PS->>STRIPE: Create checkout session
    STRIPE-->>PS: { sessionId, url }

    PS-->>C: { checkoutUrl }
    C->>STRIPE: Redirect to Stripe checkout

    Note over STRIPE: User completes payment

    STRIPE->>PS: POST /webhooks/stripe<br/>checkout.session.completed

    PS->>PG: Create subscription record
    PS->>US: PATCH /users/:id/tier<br/>{ tier: "premium" }
    PS->>NS: Send confirmation email

    PS-->>STRIPE: 200 OK

    C->>PS: Redirect to successUrl
    PS-->>C: Show success page
```

## Mocks pour tests

### Mock Stripe

```typescript
// tests/mocks/stripe.mock.ts
export const mockStripe = {
  checkout: {
    sessions: {
      create: jest.fn().mockResolvedValue({
        id: 'cs_test_123',
        url: 'https://checkout.stripe.com/test',
      }),
    },
  },
  subscriptions: {
    retrieve: jest.fn().mockResolvedValue({
      id: 'sub_123',
      status: 'active',
      current_period_end: Date.now() / 1000 + 30 * 24 * 3600,
    }),
    cancel: jest.fn().mockResolvedValue({ status: 'canceled' }),
  },
  webhooks: {
    constructEvent: jest.fn().mockReturnValue({
      type: 'checkout.session.completed',
      data: { object: { subscription: 'sub_123' } },
    }),
  },
};

jest.mock('stripe', () => jest.fn(() => mockStripe));
```

## Metriques de succes

| Metrique | Objectif | Description |
|----------|----------|-------------|
| Conversion rate | > 5% | Free -> Premium |
| Payment success | > 99% | Taux succes paiements |
| Churn rate | < 5%/mois | Annulations |
| MRR growth | > 10%/mois | Croissance revenus |
