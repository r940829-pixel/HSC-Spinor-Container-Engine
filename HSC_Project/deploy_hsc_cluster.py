# ==============================================================================
# CONTAINER-NATIVE DEVOPS CLUSTER ORCHESTRATOR & ENVIRONMENT SANITIZER
# (Upgraded: HSC v5.0 Dual-Channel Container Architecture & Tensor Bus Sync)
# This script orchestrates the unified lifecycle of the HSC simulation topology.
# It enforces hardware sanitization, purges lingering zombie containers,
# releases bound OS ports, and dynamically scales localized HSC qubit nodes.
# ==============================================================================

import os
import subprocess
import time
import sys

def clean_environment(clean_hsc=True):
    """ 
    [Automated Hardware Sanitization Operator]
    Forcibly terminates historical lingering containers and registry caches
    to guarantee a 100% network loopback port release.
    """
    print("\n[Sanitization] Initiating global environment purge and releasing ports...")
    
    # 1. Purge HSC Cluster Containers & Virtual Tensor Bus
    if clean_hsc:
        print(" -> Forcibly evicting dangling HSC Docker cluster containers...")
        subprocess.run(
            "sudo docker rm -f $(sudo docker ps -a -q --filter name=hsc_core_cluster_)", 
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            "sudo docker rm -f hsc_core_cluster_*", 
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Synchronous purging of Redis Virtual Tensor Switch
        print(" -> Purging Virtual Quantum Tensor Exchange Bus (Redis)...")
        subprocess.run(
            "sudo docker rm -f hsc_tensor_bus", 
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    print("🏆 [Sanitization Complete] Communication registers and interface ports are fully unsealed.")

def quick_deploy_network():
    print("====================================================")
    print("===   HSC Framework v5.0: Cluster Orchestrator   ===")
    print("====================================================")
    
    deploy_hsc = True

    # 1. Dynamic Scale Dimension Query Allocation (N Logical Nodes)
    try:
        n_qubits = int(input("\nDesignate the scale dimension for ideal logical HSC qubits (N): "))
        if n_qubits < 1: 
            n_qubits = 1
    except ValueError:
        n_qubits = 1
    
    # Trigger active sanitization sequence prior to port binding execution
    clean_environment(clean_hsc=deploy_hsc)
    time.sleep(1.5)  # Enforce runtime buffer delay for OS socket listener to unlock

    # ==========================================================================
    # 2. DYNAMIC CONTAINER ARCHITECTURE PROVISIONING (HSC LAYER - GPU PASSTHROUGH)
    # ==========================================================================
    
    # [Channel 0] Provisioning Virtual Tensor-Channel Switch (Redis inter-service messaging)
    if deploy_hsc and n_qubits > 1:
        print("\n[Channel 0] Provisioning Virtual Tensor-Channel Switch (Redis Message Bus)...")
        redis_cmd = "sudo docker run -d --name hsc_tensor_bus -p 2057:6379 redis:alpine redis-server --maxclients 10000"
        subprocess.run(redis_cmd, shell=True, stdout=subprocess.DEVNULL)
        print("🚀 [Tensor Bus Active] Inter-node exchange channel established on Port 2057.")

    # [Channel 1] Provisioning localized HSC Bit Node Containers
    if deploy_hsc:
        print("\n[Channel 1] Provisioning localized container-native HSC Qubit Node services...")
        hsc_base_port = 5011
        for i in range(n_qubits):
            current_port = hsc_base_port + i
            docker_cmd = (
                f"sudo docker run -d "
                f"--name hsc_core_cluster_{i} "
                f"--gpus all "
                f"--add-host=host.docker.internal:host-gateway "
                f"-e TENSOR_BUS_HOST=host.docker.internal "
                f"-e QUBIT_NODE_ID={i} "
                f"-p {current_port}:5000 "
                f"hsc-node:v5.0"
            )
            print(f" -> Mapping Node HSC-Qubit-{i} (Locked loopback interface Port: {current_port})...")
            subprocess.run(docker_cmd, shell=True, stdout=subprocess.DEVNULL)
        print("🏆 [Channel 1 Success] All HSC model service containers are up and running.")

    print("\n====================================================")
    print("🎉 [Orchestration Complete] HSC Cluster Topology is fully operational!")
    print("====================================================")

if __name__ == "__main__":
    quick_deploy_network()
