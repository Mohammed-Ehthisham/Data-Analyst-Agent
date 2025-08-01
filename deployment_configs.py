"""
Cloud deployment configurations for various platforms
"""

# Railway deployment configuration
railway_config = {
    "build": {
        "builder": "DOCKERFILE"
    },
    "deploy": {
        "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
        "healthcheckPath": "/health",
        "healthcheckTimeout": 300,
        "restartPolicyType": "ON_FAILURE",
        "restartPolicyMaxRetries": 10
    }
}

# Render deployment configuration  
render_config = {
    "services": [
        {
            "type": "web",
            "name": "data-analyst-agent",
            "env": "python",
            "buildCommand": "pip install -r requirements.txt",
            "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
            "healthCheckPath": "/health",
            "envVars": [
                {
                    "key": "OPENAI_API_KEY",
                    "sync": False
                },
                {
                    "key": "OPENAI_MODEL", 
                    "value": "gpt-4"
                }
            ]
        }
    ]
}

# Google Cloud Run configuration
cloudrun_config = {
    "apiVersion": "serving.knative.dev/v1",
    "kind": "Service",
    "metadata": {
        "name": "data-analyst-agent"
    },
    "spec": {
        "template": {
            "metadata": {
                "annotations": {
                    "autoscaling.knative.dev/maxScale": "10",
                    "run.googleapis.com/cpu-throttling": "false",
                    "run.googleapis.com/memory": "2Gi",
                    "run.googleapis.com/timeout": "300"
                }
            },
            "spec": {
                "containerConcurrency": 10,
                "containers": [
                    {
                        "image": "gcr.io/PROJECT_ID/data-analyst-agent",
                        "ports": [{"containerPort": 8000}],
                        "env": [
                            {"name": "OPENAI_API_KEY", "valueFrom": {"secretKeyRef": {"name": "openai-key", "key": "key"}}},
                            {"name": "OPENAI_MODEL", "value": "gpt-4"}
                        ],
                        "resources": {
                            "limits": {
                                "memory": "2Gi",
                                "cpu": "1000m"
                            }
                        }
                    }
                ]
            }
        }
    }
}

# AWS Lambda serverless configuration
serverless_config = {
    "service": "data-analyst-agent",
    "provider": {
        "name": "aws",
        "runtime": "python3.11",
        "region": "us-east-1",
        "timeout": 300,
        "memorySize": 3008,
        "environment": {
            "OPENAI_API_KEY": "${env:OPENAI_API_KEY}",
            "OPENAI_MODEL": "gpt-4"
        }
    },
    "functions": {
        "api": {
            "handler": "lambda_handler.handler",
            "events": [
                {
                    "http": {
                        "path": "/{proxy+}",
                        "method": "ANY",
                        "cors": True
                    }
                }
            ]
        }
    },
    "plugins": [
        "serverless-python-requirements"
    ],
    "custom": {
        "pythonRequirements": {
            "dockerizePip": True,
            "zip": True
        }
    }
}

# Docker Compose for production
docker_compose_prod = {
    "version": "3.8",
    "services": {
        "data-analyst-agent": {
            "build": ".",
            "ports": ["80:8000"],
            "environment": [
                "OPENAI_API_KEY=${OPENAI_API_KEY}",
                "OPENAI_MODEL=${OPENAI_MODEL:-gpt-4}",
                "DEBUG=False",
                "LOG_LEVEL=INFO"
            ],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "40s"
            }
        },
        "nginx": {
            "image": "nginx:alpine",
            "ports": ["443:443"],
            "volumes": [
                "./nginx.conf:/etc/nginx/nginx.conf",
                "./ssl:/etc/ssl/certs"
            ],
            "depends_on": ["data-analyst-agent"],
            "restart": "unless-stopped"
        }
    }
}
