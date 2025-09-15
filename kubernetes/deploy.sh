#!/bin/bash
# HPC QuantSim Kubernetes Deployment Script
# 
# This script deploys HPC QuantSim to a Kubernetes cluster with various configuration options

set -e

# Default values
NAMESPACE="hpc-quantsim"
DEPLOY_TYPE="dashboard"  # dashboard, gpu, mpi, all
DRY_RUN=false
SKIP_BUILD=false
IMAGE_TAG="latest"
REGISTRY=""
KUBECONFIG=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
HPC QuantSim Kubernetes Deployment Script

Usage: $0 [OPTIONS]

Options:
  -h, --help              Show this help message
  -n, --namespace NAME    Kubernetes namespace (default: hpc-quantsim)
  -t, --type TYPE         Deployment type: dashboard|gpu|mpi|all (default: dashboard)
  -d, --dry-run           Perform a dry-run without applying changes
  -s, --skip-build        Skip Docker image build
  -i, --image-tag TAG     Docker image tag (default: latest)
  -r, --registry URL      Container registry URL
  -k, --kubeconfig PATH   Path to kubeconfig file
  
Deployment Types:
  dashboard    Deploy web dashboard only
  gpu          Deploy GPU-accelerated simulation workers
  mpi          Deploy MPI cluster for distributed computing
  all          Deploy complete HPC QuantSim stack

Examples:
  $0 --type dashboard                    # Deploy dashboard only
  $0 --type all --namespace prod         # Full deployment to 'prod' namespace
  $0 --type gpu --dry-run               # Test GPU deployment without applying
  $0 --registry myregistry.com/quantsim # Use custom registry

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOY_TYPE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -s|--skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -i|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -k|--kubeconfig)
            KUBECONFIG="$2"
            export KUBECONFIG
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate deployment type
case $DEPLOY_TYPE in
    dashboard|gpu|mpi|all)
        ;;
    *)
        log_error "Invalid deployment type: $DEPLOY_TYPE"
        usage
        exit 1
        ;;
esac

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker is available (unless skipping build)
    if [ "$SKIP_BUILD" = false ] && ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Test kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker images
build_images() {
    if [ "$SKIP_BUILD" = true ]; then
        log_info "Skipping Docker image build"
        return
    fi
    
    log_info "Building Docker images..."
    
    # Build base image
    docker build -t hpc-quantsim:${IMAGE_TAG} .
    
    # Build GPU-optimized image
    docker build -t hpc-quantsim:gpu-optimized --target gpu-optimized .
    
    # Build MPI cluster image
    docker build -t hpc-quantsim:mpi-cluster --target mpi-cluster .
    
    # Tag and push to registry if specified
    if [ -n "$REGISTRY" ]; then
        log_info "Pushing images to registry: $REGISTRY"
        
        for variant in "" ":gpu-optimized" ":mpi-cluster"; do
            local_tag="hpc-quantsim${variant}"
            remote_tag="${REGISTRY}/hpc-quantsim${variant}"
            
            docker tag "$local_tag" "$remote_tag"
            docker push "$remote_tag"
        done
    fi
    
    log_info "Docker images built successfully"
}

# Apply Kubernetes manifests
apply_manifests() {
    local manifests=("$@")
    
    for manifest in "${manifests[@]}"; do
        if [ ! -f "$manifest" ]; then
            log_warn "Manifest file not found: $manifest"
            continue
        fi
        
        log_info "Applying manifest: $manifest"
        
        if [ "$DRY_RUN" = true ]; then
            kubectl apply -f "$manifest" --dry-run=client -o yaml
        else
            kubectl apply -f "$manifest"
        fi
    done
}

# Deploy namespace and common resources
deploy_common() {
    log_info "Deploying common resources..."
    
    # Update namespace in manifests if not default
    if [ "$NAMESPACE" != "hpc-quantsim" ]; then
        log_info "Updating namespace to: $NAMESPACE"
        # This would typically involve sed commands to update manifests
        # For now, we assume the default namespace
    fi
    
    apply_manifests \
        "namespace.yaml" \
        "rbac.yaml" \
        "storage.yaml" \
        "configmap.yaml"
}

# Deploy dashboard
deploy_dashboard() {
    log_info "Deploying HPC QuantSim dashboard..."
    
    apply_manifests "dashboard-deployment.yaml"
    
    if [ "$DRY_RUN" = false ]; then
        log_info "Waiting for dashboard deployment to be ready..."
        kubectl wait --for=condition=available deployment/hpc-quantsim-dashboard \
                     --namespace=$NAMESPACE --timeout=300s
        
        # Get service information
        kubectl get services --namespace=$NAMESPACE | grep dashboard
        
        log_info "Dashboard deployed successfully!"
        log_info "Access the dashboard at: http://<cluster-ip>:8000"
    fi
}

# Deploy GPU workers
deploy_gpu() {
    log_info "Deploying GPU simulation workers..."
    
    # Check if GPU nodes are available
    gpu_nodes=$(kubectl get nodes -l accelerator=nvidia-tesla-gpu --no-headers 2>/dev/null | wc -l)
    if [ "$gpu_nodes" -eq 0 ]; then
        log_warn "No GPU nodes found in cluster. GPU jobs may not schedule."
    else
        log_info "Found $gpu_nodes GPU nodes"
    fi
    
    apply_manifests "gpu-job.yaml"
    
    if [ "$DRY_RUN" = false ]; then
        log_info "GPU workers deployed. Monitor with: kubectl get jobs --namespace=$NAMESPACE"
    fi
}

# Deploy MPI cluster
deploy_mpi() {
    log_info "Deploying MPI cluster..."
    
    apply_manifests "mpi-cluster.yaml"
    
    if [ "$DRY_RUN" = false ]; then
        log_info "Waiting for MPI workers to be ready..."
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=mpi-worker \
                     --namespace=$NAMESPACE --timeout=300s
        
        log_info "MPI cluster deployed successfully!"
    fi
}

# Main deployment function
deploy() {
    log_info "Starting HPC QuantSim deployment..."
    log_info "Deployment type: $DEPLOY_TYPE"
    log_info "Namespace: $NAMESPACE"
    log_info "Image tag: $IMAGE_TAG"
    log_info "Dry run: $DRY_RUN"
    
    # Always deploy common resources first
    deploy_common
    
    case $DEPLOY_TYPE in
        dashboard)
            deploy_dashboard
            ;;
        gpu)
            deploy_dashboard  # Dashboard is needed for monitoring
            deploy_gpu
            ;;
        mpi)
            deploy_dashboard  # Dashboard is needed for monitoring
            deploy_mpi
            ;;
        all)
            deploy_dashboard
            deploy_gpu
            deploy_mpi
            ;;
    esac
    
    if [ "$DRY_RUN" = false ]; then
        log_info "Deployment completed successfully!"
        log_info "Check status with: kubectl get all --namespace=$NAMESPACE"
    else
        log_info "Dry-run completed. Use without --dry-run to apply changes."
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up HPC QuantSim deployment..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Dry-run: Would delete namespace $NAMESPACE and all resources"
        return
    fi
    
    # Delete namespace (this will delete all resources)
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    
    log_info "Cleanup completed"
}

# Handle cleanup on script exit
trap 'log_info "Deployment script interrupted"' INT TERM

# Main execution
check_prerequisites

# Check if this is a cleanup operation
if [ "$1" = "cleanup" ]; then
    cleanup
    exit 0
fi

build_images
deploy

log_info "HPC QuantSim deployment script completed successfully!"
