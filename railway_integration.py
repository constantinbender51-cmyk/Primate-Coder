import os
import requests
import json
import threading
import time

RAILWAY_API_URL = 'https://backboard.railway.app/graphql/v2'

def get_deployment_status(debug_logs):
    """Get latest Railway deployment status"""
    api_key = os.environ.get("RAILWAY_API_KEY")
    project_id = os.environ.get("RAILWAY_PROJECT_ID")
    
    if not api_key or not project_id:
        debug_logs.put({
            "type": "Railway Status",
            "data": "Railway API credentials not configured"
        })
        return None
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    query = f"""
    {{
        deployments(input: {{projectId: "{project_id}"}}, first: 1) {{
            edges {{
                node {{
                    id
                    status
                    createdAt
                    staticUrl
                    environment {{
                        name
                    }}
                }}
            }}
        }}
    }}
    """
    
    try:
        response = requests.post(
            RAILWAY_API_URL,
            headers=headers,
            json={'query': query},
            timeout=10
        )
        
        data = response.json()
        
        if 'errors' in data:
            debug_logs.put({
                "type": "Railway API Error",
                "data": data['errors'][0]['message']
            })
            return None
        
        deployments = data.get('data', {}).get('deployments', {}).get('edges', [])
        
        if not deployments:
            debug_logs.put({
                "type": "Railway Status",
                "data": "No deployments found"
            })
            return None
        
        deployment = deployments[0]['node']
        
        status_info = {
            'id': deployment['id'],
            'status': deployment['status'],
            'createdAt': deployment['createdAt'],
            'url': deployment.get('staticUrl'),
            'environment': deployment.get('environment', {}).get('name', 'unknown')
        }
        
        debug_logs.put({
            "type": "Railway Deployment",
            "data": f"Status: {status_info['status']} | Env: {status_info['environment']}",
            "fullData": json.dumps(status_info, indent=2)
        })
        
        return status_info
        
    except Exception as e:
        debug_logs.put({
            "type": "Railway Error",
            "data": f"Failed to check deployment: {str(e)}"
        })
        return None


def get_deployment_logs(deployment_id, debug_logs):
    """Get logs for a specific deployment"""
    api_key = os.environ.get("RAILWAY_API_KEY")
    
    if not api_key:
        return None
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    query = f"""
    {{
        deploymentLogs(deploymentId: "{deployment_id}", limit: 100) {{
            message
            severity
            timestamp
        }}
    }}
    """
    
    try:
        response = requests.post(
            RAILWAY_API_URL,
            headers=headers,
            json={'query': query},
            timeout=10
        )
        
        data = response.json()
        
        if 'errors' in data:
            return None
        
        logs = data.get('data', {}).get('deploymentLogs', [])
        
        if not logs:
            return "No logs available"
        
        formatted_logs = '\n'.join([
            f"{log['timestamp']} [{log['severity']}] {log['message']}"
            for log in sorted(logs, key=lambda x: x['timestamp'])
        ])
        
        return formatted_logs
        
    except Exception as e:
def monitor_deployment_status(debug_logs, interval=60):
    """Continuously monitor Railway deployment status"""
    while True:
        get_deployment_status(debug_logs)
        time.sleep(interval)


def start_railway_monitor(debug_logs):
    """Start Railway monitoring in background thread"""
    monitor_thread = threading.Thread(
        target=monitor_deployment_status,
        args=(debug_logs,),
        daemon=True
    )
    monitor_thread.start()
    debug_logs.put({
        "type": "Railway Monitor",
        "data": "Started deployment status monitoring"
    })
