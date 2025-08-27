#!/usr/bin/env python3
"""
RunPod Deployment Script pro AI Style Transfer aplikaci

Tento skript automatizuje nasazení aplikace na RunPod GPU servery.
"""

import os
import json
import requests
import time
from typing import Dict, Optional

class RunPodDeployer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runpod.ai/graphql"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_template(self, template_config: Dict) -> Optional[str]:
        """Vytvoří template pro deployment"""
        mutation = """
        mutation createTemplate($input: PodTemplateInput!) {
            createTemplate(input: $input) {
                id
                name
            }
        }
        """
        
        variables = {"input": template_config}
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json={"query": mutation, "variables": variables}
        )
        
        if response.status_code == 200:
            data = response.json()
            if "errors" not in data:
                template_id = data["data"]["createTemplate"]["id"]
                print(f"✅ Template vytvořen: {template_id}")
                return template_id
            else:
                print(f"❌ Chyba při vytváření template: {data['errors']}")
        else:
            print(f"❌ HTTP chyba: {response.status_code}")
        
        return None
    
    def deploy_pod(self, pod_config: Dict) -> Optional[str]:
        """Nasadí pod na RunPod"""
        mutation = """
        mutation createPod($input: PodRentInterruptableInput!) {
            createPod(input: $input) {
                id
                desiredStatus
                imageName
                env
                machineId
                machine {
                    podHostId
                }
            }
        }
        """
        
        variables = {"input": pod_config}
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json={"query": mutation, "variables": variables}
        )
        
        if response.status_code == 200:
            data = response.json()
            if "errors" not in data:
                pod_id = data["data"]["createPod"]["id"]
                print(f"✅ Pod nasazen: {pod_id}")
                return pod_id
            else:
                print(f"❌ Chyba při nasazování pod: {data['errors']}")
        else:
            print(f"❌ HTTP chyba: {response.status_code}")
        
        return None
    
    def get_pod_status(self, pod_id: str) -> Dict:
        """Získá status podu"""
        query = """
        query getPod($input: String!) {
            pod(input: $input) {
                id
                desiredStatus
                lastStatusChange
                imageName
                env
                machineId
                machine {
                    podHostId
                }
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
            }
        }
        """
        
        variables = {"input": pod_id}
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json={"query": query, "variables": variables}
        )
        
        if response.status_code == 200:
            data = response.json()
            if "errors" not in data:
                return data["data"]["pod"]
        
        return {}

def main():
    # Konfigurace
    API_KEY = os.getenv("RUNPOD_API_KEY")
    if not API_KEY:
        print("❌ Nastavte RUNPOD_API_KEY environment variable")
        return
    
    deployer = RunPodDeployer(API_KEY)
    
    # Template konfigurace
    template_config = {
        "name": "ai-style-transfer-template",
        "imageName": "your-dockerhub-username/ai-style-transfer:latest",
        "dockerArgs": "",
        "containerDiskInGb": 50,
        "volumeInGb": 100,
        "volumeMountPath": "/app/models",
        "ports": "8501/http",
        "env": [
            {"key": "FORCE_CPU", "value": "false"},
            {"key": "MAX_MEMORY_GB", "value": "24"},
            {"key": "ENABLE_ATTENTION_SLICING", "value": "true"},
            {"key": "ENABLE_CPU_OFFLOAD", "value": "auto"},
            {"key": "BASE_MODEL", "value": "stabilityai/stable-diffusion-xl-base-1.0"}
        ]
    }
    
    # Pod konfigurace
    pod_config = {
        "cloudType": "ALL",
        "gpuCount": 1,
        "volumeInGb": 100,
        "containerDiskInGb": 50,
        "minVcpuCount": 4,
        "minMemoryInGb": 16,
        "gpuTypeId": "NVIDIA RTX A5000",
        "name": "ai-style-transfer-pod",
        "imageName": "your-dockerhub-username/ai-style-transfer:latest",
        "dockerArgs": "",
        "ports": "8501/http",
        "volumeMountPath": "/app/models",
        "env": [
            {"key": "FORCE_CPU", "value": "false"},
            {"key": "MAX_MEMORY_GB", "value": "24"},
            {"key": "ENABLE_ATTENTION_SLICING", "value": "true"},
            {"key": "ENABLE_CPU_OFFLOAD", "value": "auto"},
            {"key": "BASE_MODEL", "value": "stabilityai/stable-diffusion-xl-base-1.0"}
        ]
    }
    
    print("🚀 Spouštím deployment na RunPod...")
    
    # Vytvoření template (volitelné)
    template_id = deployer.create_template(template_config)
    
    # Nasazení podu
    pod_id = deployer.deploy_pod(pod_config)
    
    if pod_id:
        print(f"⏳ Čekám na spuštění podu {pod_id}...")
        
        # Čekání na spuštění
        for i in range(30):  # Max 5 minut
            time.sleep(10)
            status = deployer.get_pod_status(pod_id)
            
            if status.get("runtime"):
                ports = status["runtime"].get("ports", [])
                if ports:
                    public_url = f"https://{ports[0]['ip']}:{ports[0]['publicPort']}"
                    print(f"✅ Aplikace je dostupná na: {public_url}")
                    break
            
            print(f"⏳ Stále čekám... ({i+1}/30)")
        else:
            print("❌ Timeout při čekání na spuštění")

if __name__ == "__main__":
    main()