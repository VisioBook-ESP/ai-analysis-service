# Auth Flow Architecture — VisioBook Microservices

## Current State (Istio Gateway)

- **Istio ingress gateway** validates JWT and injects `X-User-Id` header (replacing Kong)
- core-user-service (FastAPI) issues RS256 JWTs (UUID `sub`) and exposes JWKS at `/api/v1/auth/.well-known/jwks.json`
- `RequestAuthentication` validates JWT signature against JWKS, extracts `sub` → `x-user-id` header
- VirtualServices strip client-supplied `x-user-id` to prevent header spoofing
- `AuthorizationPolicy` enforces `requestPrincipals: ["*"]` on protected routes
- Istio mTLS (`ISTIO_MUTUAL`) secures all pod-to-pod traffic

---

## Actual Flow — Istio Gateway (Current Implementation)

Istio validates JWT at the ingress, injects `x-user-id` from the `sub` claim, and forwards both the header and the original Bearer token to services.

```mermaid
sequenceDiagram
    participant Client
    participant Istio as Istio Ingress Gateway
    participant VS as VirtualService
    participant SvcA as core-project-service
    participant SvcB as ai-analysis-service
    participant UserSvc as core-user-service

    Note over UserSvc: Issues RS256 JWTs (sub=UUID)<br/>Exposes JWKS at /.well-known/jwks.json

    rect rgb(240, 248, 255)
        Note over Istio,UserSvc: Istio startup (RequestAuthentication)
        Istio->>UserSvc: GET /api/v1/auth/.well-known/jwks.json
        UserSvc-->>Istio: { keys: [{ kid: "visiobook-key-1", alg: RS256, ... }] }
        Note over Istio: Caches JWKS for JWT verification
    end

    Note over Client,UserSvc: Login (once per session)
    Client->>Istio: POST /api/v1/auth/login { email, password }
    Note over Istio: AuthorizationPolicy: public route<br/>No JWT required
    Istio->>UserSvc: Forward (no auth check)
    UserSvc-->>Client: { access_token: "eyJ..." }

    rect rgb(245, 255, 245)
        Note over Client,SvcB: Authenticated request
        Client->>Istio: GET /api/v1/projects<br/>Authorization: Bearer eyJ...

        Istio->>Istio: RequestAuthentication:<br/>1. Verify RS256 signature (cached JWKS)<br/>2. Check exp, iss="core-user-service"<br/>3. outputClaimToHeaders: sub → x-user-id

        Istio->>VS: Forward with x-user-id header
        VS->>VS: Remove client-supplied x-user-id<br/>(anti-spoofing)

        VS->>SvcA: Forward request<br/>+ X-User-Id: "550e8400-..." (from JWT sub)<br/>+ Authorization: Bearer eyJ... (forwardOriginalToken)<br/>+ Istio mTLS transport

        Note over SvcA: GatewayAuthGuard:<br/>Read x-user-id header (UUID)<br/>@CurrentUser() → userId

        SvcA->>SvcA: Ownership check:<br/>findFirst({ where: { id, userId, deletedAt: null } })<br/>Return 404 if not owned

        SvcA->>SvcB: POST /api/v1/analyze<br/>X-User-Id: "550e8400-..."<br/>Authorization: Bearer eyJ...<br/>X-Request-Id: correlationId
        Note over SvcA,SvcB: Istio mTLS (pod-to-pod)

        SvcB->>SvcB: Read x-user-id header<br/>Associate job with userId

        SvcB-->>SvcA: { jobId: "..." }
        SvcA-->>Client: 200 { project: {...} }
    end

    rect rgb(255, 240, 240)
        Note over Client,Istio: Invalid/expired token
        Client->>Istio: GET /api/v1/projects<br/>Authorization: Bearer expired...
        Istio-->>Client: 401 Unauthorized<br/>(AuthorizationPolicy: requestPrincipals required)
    end
```

### Istio Resources Required

| Resource | Purpose |
|----------|---------|
| **RequestAuthentication** | Validate RS256 JWT using JWKS from core-user-service. `outputClaimToHeaders: sub → x-user-id`. `forwardOriginalToken: true` |
| **AuthorizationPolicy (deny-all)** | Default deny on ingress gateway |
| **AuthorizationPolicy (public)** | Allow `/api/v1/auth/login`, `/api/v1/auth/register`, `/.well-known/*`, `/health/*` without JWT |
| **AuthorizationPolicy (per-service)** | Require `requestPrincipals: ["*"]` for protected routes |
| **VirtualService (per-service)** | Route to backend + strip client-supplied `x-user-id` header |
| **DestinationRule** | `ISTIO_MUTUAL` mTLS + connection pooling + outlier detection |
| **PeerAuthentication** | Enforce mTLS between pods |

### Security Layers

1. **Gateway-level** — Istio `RequestAuthentication` validates JWT signature, expiry, issuer
2. **Ingress-level** — `AuthorizationPolicy` enforces path-based access + requires valid principal
3. **Anti-spoofing** — VirtualService removes client-supplied `x-user-id`; only Istio-injected header reaches services
4. **Transport** — `ISTIO_MUTUAL` mTLS between all pods (only pods with valid certs can communicate)
5. **Service-level** — Application ownership enforcement (`userId == resource.userId`, return 404 not 403)

### What Each Service Needs

- Read `x-user-id` header for user identity (no JWT library needed)
- Auth guard/middleware rejecting requests without `x-user-id` on protected routes
- Ownership enforcement: verify userId owns the resource before access
- Forward both `X-User-Id` AND `Authorization: Bearer` on outbound S2S HTTP calls
- Self-generate `X-Request-Id` (UUID) for request correlation

---

## Reference: Kong Gateway Flow (Not Deployed)

Kong validates JWT centrally, injects identity headers, and forwards to services. Services never touch JWT validation.

```mermaid
sequenceDiagram
    participant Client
    participant Kong as Kong API Gateway
    participant SvcA as core-project-service
    participant SvcB as ai-analysis-service
    participant UserSvc as core-user-service

    Note over UserSvc: Issues RS256 JWTs<br/>Exposes JWKS at /.well-known/jwks.json

    rect rgb(240, 248, 255)
        Note over Kong,UserSvc: Kong startup (once, cached)
        Kong->>UserSvc: GET /api/v1/auth/.well-known/jwks.json
        UserSvc-->>Kong: { keys: [{ kid, n, e, alg: RS256 }] }
        Note over Kong: Caches public key for JWT verification
    end

    Note over Client,UserSvc: Login (once per session)
    Client->>Kong: POST /api/v1/auth/login { email, password }
    Kong->>UserSvc: Forward (public route, no JWT check)
    UserSvc-->>Kong: { access_token: "eyJ..." }
    Kong-->>Client: { access_token: "eyJ..." }

    rect rgb(245, 255, 245)
        Note over Client,SvcB: Normal request — Kong validates JWT
        Client->>Kong: GET /api/v1/projects<br/>Authorization: Bearer eyJ...

        Kong->>Kong: 1. Verify JWT signature (cached JWKS)<br/>2. Check exp, iss<br/>3. Extract sub="6", role="user"<br/>4. Generate X-Request-Id UUID

        Kong->>SvcA: Forward request<br/>+ X-User-Id: 6<br/>+ X-Request-Id: abc-123<br/>(Authorization header stripped)

        Note over SvcA: Trusts X-User-Id from Kong<br/>No JWT library needed

        SvcA->>SvcB: POST /api/v1/analysis/start<br/>X-User-Id: 6<br/>X-Request-Id: abc-123
        Note over SvcA,SvcB: Istio mTLS (pod-to-pod)

        SvcB->>SvcB: Trusts X-User-Id<br/>(internal network only)

        SvcB-->>SvcA: { analysisId: "..." }
        SvcA-->>Kong: 200 { project: {...} }
        Kong-->>Client: 200 OK<br/>X-Request-Id: abc-123
    end

    rect rgb(255, 240, 240)
        Note over Client,Kong: Invalid/expired token
        Client->>Kong: GET /api/v1/projects<br/>Authorization: Bearer expired...
        Kong-->>Client: 401 Unauthorized<br/>(never reaches microservice)
    end
```

### Kong Plugins Required

| Plugin | Purpose |
|--------|---------|
| **jwt** | Validate RS256 signature using JWKS from core-user-service |
| **request-transformer** | Add `X-User-Id` (from JWT `sub`), generate `X-Request-Id` (UUID) |
| **cors** | Handle CORS headers, expose `X-Request-Id` |
| **rate-limiting** | Per-user rate limits (keyed by `sub` claim) |

### Pros
- Services are simple — no JWT libraries, just read headers
- Auth logic centralized in one place
- Invalid tokens rejected before reaching any service

### Cons
- Full trust of internal network required (anyone on the network can forge `X-User-Id`)
- Istio mTLS mitigates this (only pods with valid certs can communicate)

---

## Current Reality Flow — No Kong, Services Validate JWT

Without Kong, each service must validate JWT itself using the JWKS endpoint. The Bearer token is forwarded on inter-service calls.

```mermaid
sequenceDiagram
    participant Client
    participant LB as Load Balancer / Ingress
    participant SvcA as core-project-service
    participant SvcB as ai-analysis-service
    participant UserSvc as core-user-service

    Note over UserSvc: Issues RS256 JWTs<br/>Exposes JWKS at /.well-known/jwks.json

    rect rgb(240, 248, 255)
        Note over SvcA,UserSvc: Service startup (once, then cached + auto-refreshed)
        SvcA->>UserSvc: GET /api/v1/auth/.well-known/jwks.json
        UserSvc-->>SvcA: { keys: [{ kid, n, e, alg: RS256 }] }
        SvcB->>UserSvc: GET /api/v1/auth/.well-known/jwks.json
        UserSvc-->>SvcB: { keys: [{ kid, n, e, alg: RS256 }] }
        Note over SvcA,SvcB: Public keys cached in memory<br/>Auto-refresh on unknown kid (key rotation)
    end

    Note over Client,UserSvc: Login (once per session)
    Client->>LB: POST /api/v1/auth/login { email, password }
    LB->>UserSvc: Forward
    UserSvc-->>Client: { access_token: "eyJ..." }

    rect rgb(245, 255, 245)
        Note over Client,SvcB: Normal request — each service validates JWT
        Client->>LB: GET /api/v1/projects<br/>Authorization: Bearer eyJ...
        LB->>SvcA: Forward (Bearer token passthrough)

        SvcA->>SvcA: Verify JWT signature<br/>using cached JWKS public key<br/>Extract sub="6", role="user"<br/>Generate correlationId (UUID)

        SvcA->>SvcB: POST /api/v1/analysis/start<br/>Authorization: Bearer eyJ...<br/>(forward original token)
        Note over SvcA,SvcB: Istio mTLS (pod-to-pod)

        SvcB->>SvcB: Verify JWT signature<br/>using cached JWKS public key<br/>Extract sub="6"

        SvcB-->>SvcA: { analysisId: "..." }
        SvcA-->>Client: 200 { project: {...} }
    end

    rect rgb(255, 245, 238)
        Note over SvcA,UserSvc: Key rotation (rare, automatic)
        SvcA->>SvcA: JWT verify fails:<br/>unknown kid
        SvcA->>UserSvc: Re-fetch JWKS
        UserSvc-->>SvcA: Updated key set
        SvcA->>SvcA: Retry verify with new key
    end
```

### What Each Service Needs (without Kong)
- `jose` library (or equivalent) for JWT verification
- JWKS URL config pointing to core-user-service
- Auth guard that reads `Authorization: Bearer` header
- Forward the original Bearer token on inter-service HTTP calls
- Self-generate correlationId (no gateway to do it)

### Pros
- Defense in depth — each service independently verifies identity
- No single point of auth failure
- Works without Kong

### Cons
- JWT library needed in every service
- Token verification overhead on every request (minimal — RSA verify is ~0.1ms with cached key)
- Every service must know the JWKS URL

---

## Comparison

| Aspect | Istio (current) | Kong (not deployed) | No gateway |
|--------|-----------------|---------------------|------------|
| JWT validation | Istio ingress (centralized) | Kong (centralized) | Each service (distributed) |
| User identity | `X-User-Id` from Istio `outputClaimToHeaders` | `X-User-Id` from Kong `request-transformer` | Decoded from JWT `sub` claim |
| Anti-spoofing | VirtualService strips client `x-user-id` | Kong strips/overwrites | N/A (each service validates JWT) |
| Request tracing | Self-generated UUID per service | `X-Request-Id` from Kong | Self-generated UUID per service |
| Inter-service auth | Forward `X-User-Id` + Bearer token | Forward `X-User-Id` header | Forward `Authorization: Bearer` token |
| Transport security | `ISTIO_MUTUAL` mTLS (built-in) | Requires Istio mTLS alongside | Requires Istio mTLS alongside |
| Service complexity | Simple (read headers) | Simple (read headers) | Moderate (JWT + JWKS logic) |
| Failure mode | Istio ingress down = everything down | Kong down = everything down | User-service JWKS down = new services can't start |
