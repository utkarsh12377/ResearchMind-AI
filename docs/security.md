# Security

What this system defends against, where each control lives, and what is
deliberately out of scope.

## Threat model

The system ingests documents from users, retrieves passages from them, and feeds
those passages to a language model that can call tools. That shape produces four
categories of risk, and they are treated separately because the mitigations do
not overlap.

| Risk | Where it enters | Control |
| --- | --- | --- |
| Cross-tenant data access | Any read path | Authorization applied before scoring |
| Malicious uploads | `POST /papers` | Magic-byte check, filename sanitization, size caps |
| Prompt injection | Paper text, web results | Sanitization, delimiting, detection and dropping |
| Generated-query abuse | `POST /graph/query` | Whitelist validator over generated Cypher |

## Authentication and authorization

Two credential types, one identity model. JWTs are for interactive sessions;
API keys are for programmatic clients that cannot complete a login flow. Keys
are stored as SHA-256 hashes with a short display prefix, so the database never
holds a usable credential and a leaked dump does not grant access.

**Authorization runs before scoring, not after.** The retrieval pipeline resolves
the caller's accessible paper ids first and restricts the candidate set to them.
Filtering after ranking would be simpler and would leak: result counts, score
distributions, and reranker behaviour all change measurably in the presence of
documents the caller cannot read. Scoping first means another workspace's
content is never a candidate at all.

The same rule applies to the insight endpoints. Passing another user's paper id
narrows the analysis to nothing rather than returning an error, because a
distinguishable error is itself a disclosure that the paper exists.

## Upload handling

A declared `Content-Type` is a claim the client makes about its own file. Uploads
are checked three ways:

1. **Magic bytes.** The first five bytes must be `%PDF-`. This closes the gap
   between what the client says and what it sent.
2. **Filename sanitization.** Filenames are attacker-controlled text that reaches
   a storage path, log lines, and eventually a UI. `sanitize_filename` NFKC-normalizes
   first (so a fullwidth solidus cannot smuggle a separator past the character
   filter), strips separators and `..`, replaces control characters, and defuses
   Windows reserved device names.
3. **Size, twice.** `BodySizeLimitMiddleware` refuses on the declared
   `Content-Length` before the body is buffered; the upload service checks the
   actual byte count after. The first stops a hostile client from making the
   process hold a gigabyte on the way to a 413; the second stops a lying header.

Blob storage is content-addressed, and `LocalStorageBackend._resolve` performs a
containment check against the storage root, so a crafted key cannot escape it.

## Prompt injection

Retrieved text is data that a model reads as if it were instructions. There is no
single fix, so the defences are layered:

- **Sanitization.** Zero-width and bidirectional control characters are stripped
  from external content. These are invisible in a UI and meaningful to a
  tokenizer, which is exactly the asymmetry an injection exploits.
- **Explicit delimiting.** External content is wrapped in labelled UNTRUSTED
  markers, so the model is told which region is evidence rather than instruction.
- **Detection and dropping.** Content matching known injection patterns is
  discarded rather than passed through. Forwarding it and relying on the model to
  resist would make the system's safety a property of prompt adherence.
- **Typed tools.** Tools take structured arguments, not model-authored strings.
  There is no path from generated text to a shell, a filesystem, or an arbitrary
  URL.

The same principle governs extraction. Entities and relations proposed by a model
are validated against a closed schema before they are written, so a hallucinated
triple cannot become durable state.

## Generated Cypher

`POST /graph/query` translates a question into Cypher and runs it. This is only
acceptable because something other than the model decides what runs.

`validate_cypher` is a **whitelist**: allowed clauses, allowed labels, allowed
relationship types. A blacklist of dangerous keywords loses to the first spelling
you did not anticipate. It also:

- blanks string literals before scanning for keywords, so a paper titled
  "Deleting Noisy Labels" is not read as a `DELETE`, and equally a keyword hidden
  inside quotes cannot execute;
- rejects anything with more than one statement;
- requires the query to start with a read clause and to `RETURN`;
- appends or clamps a `LIMIT` regardless of what the model wrote.

The rejected query is returned to the caller. A graph answer nobody can inspect
is not one worth trusting.

## Transport and headers

Every response carries `X-Content-Type-Options`, `X-Frame-Options`,
`Referrer-Policy`, `Cross-Origin-Opener-Policy`, `Permissions-Policy`, and a
restrictive CSP. The API serves JSON to a separate origin and has no first-party
HTML, so `default-src 'none'` costs nothing and covers the case of a browser
pointed at an endpoint directly.

CORS is an allow-list of origins from configuration, never `*` with credentials.

## Secrets

- `.env` is git-ignored; `.env.example` documents every variable with placeholder
  values only.
- `JWT_SECRET` ships with a development default that names itself as one. Rotating
  it invalidates every outstanding token, which is the intended blast radius.
- Provider keys are read from the environment and never logged. Structured log
  events carry identifiers and counts, not payloads.
- In Kubernetes, secrets are mounted from a `Secret` rather than baked into the
  image or the manifest. `infra/k8s/secret.example.yaml` is a template with no
  real values.

## Rate limiting

Applied per-IP on authentication routes, which are the endpoints where an
unauthenticated attacker gets unlimited attempts. Limits live in configuration
because the right number depends on deployment shape.

## Dependency scanning

`.github/workflows/security.yml` runs `pip-audit` and `npm audit` on every push
and on a weekly schedule. The schedule matters more than the push trigger: most
vulnerabilities are disclosed after the code that depends on them was last
touched, so a push-only scan goes quiet exactly when a project stabilizes.

## Known gaps

Stated rather than hidden, because a security document that lists only what was
done is not useful:

- **Tokens live in `localStorage`**, readable by any script on the origin. The
  stronger option is an httpOnly cookie, which is not available while the API is
  a separate origin serving non-browser clients too. Revisit if both are ever
  served from one origin.
- **No audit log.** Access is authorized but not recorded per-object.
- **Injection detection is pattern-based** and will miss novel phrasings. It
  reduces exposure; it does not eliminate it.
- **Neo4j runs with a single credential**, so query-level authorization depends
  entirely on the validator rather than on database roles.
