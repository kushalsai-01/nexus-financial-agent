#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${CYAN}[NEXUS]${NC} $*"; }
success() { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }

usage() {
    cat <<EOF
NEXUS Deployment Script

Usage: $(basename "$0") <command> [options]

Commands:
    build       Build Docker image
    up          Start all services (docker-compose)
    down        Stop all services
    restart     Restart services
    logs        View logs
    status      Show service status
    db-migrate  Run database migrations
    deploy-k8s  Deploy to Kubernetes cluster
    clean       Remove all containers and volumes

Options:
    -e, --env ENV    Environment (dev|staging|prod) [default: dev]
    -t, --tag TAG    Docker image tag [default: latest]
    -h, --help       Show this help
EOF
}

ENV="dev"
TAG="latest"
IMAGE_NAME="nexus-financial-agent"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env) ENV="$2"; shift 2 ;;
        -t|--tag) TAG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) COMMAND="$1"; shift ;;
    esac
done

COMMAND="${COMMAND:-help}"

cmd_build() {
    log "Building Docker image: ${IMAGE_NAME}:${TAG}"
    docker build \
        -f "$PROJECT_ROOT/docker/Dockerfile" \
        -t "${IMAGE_NAME}:${TAG}" \
        --build-arg "BUILD_ENV=${ENV}" \
        "$PROJECT_ROOT"
    success "Image built: ${IMAGE_NAME}:${TAG}"
}

cmd_up() {
    log "Starting NEXUS services (${ENV})..."
    cd "$PROJECT_ROOT/docker"
    docker compose up -d
    success "Services started"
    echo ""
    docker compose ps
}

cmd_down() {
    log "Stopping NEXUS services..."
    cd "$PROJECT_ROOT/docker"
    docker compose down
    success "Services stopped"
}

cmd_restart() {
    cmd_down
    cmd_up
}

cmd_logs() {
    cd "$PROJECT_ROOT/docker"
    docker compose logs -f --tail=100
}

cmd_status() {
    log "NEXUS Service Status"
    echo ""
    cd "$PROJECT_ROOT/docker"
    docker compose ps
    echo ""
    log "Resource Usage"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
        $(docker compose ps -q 2>/dev/null) 2>/dev/null || warn "No running containers"
}

cmd_db_migrate() {
    log "Running database migrations..."
    cd "$PROJECT_ROOT/docker"
    docker compose exec nexus python -m nexus.data.migrations
    success "Migrations complete"
}

cmd_deploy_k8s() {
    log "Deploying to Kubernetes (${ENV})..."

    if ! command -v kubectl &>/dev/null; then
        error "kubectl not found. Install it first."
    fi

    DEPLOY_DIR="$PROJECT_ROOT/deploy/k8s"

    kubectl apply -f "$DEPLOY_DIR/namespace.yaml"
    kubectl apply -f "$DEPLOY_DIR/configmap.yaml"
    kubectl apply -f "$DEPLOY_DIR/services.yaml"
    kubectl apply -f "$DEPLOY_DIR/deployment.yaml"

    log "Waiting for deployment..."
    kubectl -n nexus rollout status deployment/nexus-agent --timeout=300s
    kubectl -n nexus rollout status deployment/nexus-dashboard --timeout=300s

    success "Kubernetes deployment complete"
    echo ""
    kubectl -n nexus get pods
}

cmd_clean() {
    warn "This will remove all NEXUS containers, images, and volumes."
    read -p "Are you sure? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_ROOT/docker"
        docker compose down -v --rmi all 2>/dev/null || true
        success "Cleanup complete"
    fi
}

case "$COMMAND" in
    build)      cmd_build ;;
    up)         cmd_up ;;
    down)       cmd_down ;;
    restart)    cmd_restart ;;
    logs)       cmd_logs ;;
    status)     cmd_status ;;
    db-migrate) cmd_db_migrate ;;
    deploy-k8s) cmd_deploy_k8s ;;
    clean)      cmd_clean ;;
    help)       usage ;;
    *)          error "Unknown command: $COMMAND. Use --help for usage." ;;
esac
