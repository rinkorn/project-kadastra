# kadastra

Cadastral valuation pilot on H3 hexagonal grid (pilot region: Tatarstan / Kazan agglomeration).

## Local development

```sh
uv sync
uv run python scripts/serve.py
# → http://127.0.0.1:15777
```

Or via Docker (rebuilds the image, mounts `./data` from host):

```sh
docker compose up --build
# → http://127.0.0.1:15777
```

## Branches

- `dev` — active development. CI runs lint + tests on every push; no auto-deploy.
- `stage` — auto-deploys to VM 224 (`kadastra.ohnice.synology.me`) on every push to this branch.
- `main` — reserved for future prod. CI runs lint + tests; no auto-deploy yet.

Workflow: feature → `dev` → PR/merge → `stage` (auto-deploy) → eventually `stage` → `main` (when prod env is set up).

## Deployment

CI/CD: [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml). Push to `stage` triggers `lint → test → deploy-stage`.

### Topology

```text
Internet ──▶ Keenetic (port-forward 2336 → 22) ──▶ Proxmox VM 224 (10.0.0.152)
                                                          │
                                                          ▼
                                                  Docker Compose (kadastra-stage project)
                                                          │
                                                          ▼
                                                  kadastra:15777 (container)
                                                          │
                                                          ▼
                                                  host:15778 (mapped)

Internet ──▶ Synology reverse-proxy ──▶ http://10.0.0.152:15778
              (TLS termination, kadastra.ohnice.synology.me)
```

The VM listens only on the LAN; public access is exclusively through the Synology reverse-proxy. SSH (port 2336 → 22) is reserved for the GitHub Actions deploy step.

### One-time setup on VM 224

```sh
# As rinkorn:
sudo mkdir -p /opt/kadastra-stage
sudo chown rinkorn:rinkorn /opt/kadastra-stage

# Add the public half of STAGE_SSH_KEY (see GitHub Secrets) to:
~/.ssh/authorized_keys
```

On Synology, add a reverse-proxy entry:

- Source: `kadastra.ohnice.synology.me` (HTTPS)
- Destination: `http://10.0.0.152:15778`

### GitHub Settings → Secrets and variables → Actions

Variables:

- `DEPLOY_USER` = `rinkorn`
- `STAGE_HOST` = public hostname or IP (whatever resolves Keenetic's WAN IP)
- `STAGE_PORT` = `2336`
- `STAGE_DEPLOY_PATH` = `/opt/kadastra-stage`
- `STAGE_INTERNAL_PORT` = `15778`
- `STAGE_PULL_DATA_ON_START` = `true` — keep on. The mirror is idempotent (size-match skip), so subsequent deploys only transfer files that actually changed in S3 since the last pull. Cold-start = ~10 GB / 10–15 min; code-only redeploy = ~30 s of HEAD requests.
- `STAGE_S3_ENDPOINT_URL`, `STAGE_S3_BUCKET`, `STAGE_S3_REGION`, `STAGE_S3_ADDRESSING_STYLE`

Secrets:

- `STAGE_SSH_KEY` — private key whose public counterpart sits in `rinkorn@VM224:~/.ssh/authorized_keys`
- `STAGE_S3_ACCESS_KEY`, `STAGE_S3_SECRET_KEY`

### Manual deploy (without GitHub Actions)

```sh
# On dev machine:
rsync -az --delete --exclude='.git' --exclude='.venv' --exclude='data' \
    --exclude='.env' --exclude='__pycache__' \
    -e "ssh -p 2336" ./ rinkorn@<host>:/opt/kadastra-stage/

# On VM:
ssh kadastra
cd /opt/kadastra-stage
# Edit .env (see entrypoint env vars: PULL_DATA_ON_START, S3_*, KADASTRA_HOST_PORT, ...)
docker compose -p kadastra-stage -f docker-compose.stage.yml up -d --build
curl http://localhost:15778/health
```

### Healthcheck

The container exposes `GET /health` → `{"status": "ok"}`. The Compose file uses it for the Docker healthcheck; the GitHub Actions deploy step polls it for up to 150 s before declaring the deploy successful.
