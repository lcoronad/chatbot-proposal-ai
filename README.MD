# chatbot-proposal-ai

Chatbot para la aplicación de propuestas de Red Hat, para su correcta ejecución se debe tener un llama-stack-server corriendo.

## Setup local

Inicializar el ambiente de Python

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

Instalar las dependencias

```bash
pip install -r requirements.txt
```

Ejecutar localmente

```bash
python chatbot_ui.py
```

## Generar imagen de contenedor

### Variables
```bash
export USER=lcoronad
export ORGANIZATION_QUAY=lcoronad
export VERSION=1.0.0
```

### Build imagen
```bash
podman build -t chatbot-proposal-ai .
```

### Run local
```bash
podman run -p 7861:7861 -e LLAMA_STACK_BASE_URL=http://192.168.68.122:8321 --name chatbot-proposal-ai chatbot-proposal-ai:latest
```

### Login quay
```bash
podman login -u ${USER} quay.io
```

### tag the local container image and push to quay
```bash
podman tag localhost/chatbot-proposal-ai quay.io/${ORGANIZATION_QUAY}/chatbot-proposal-ai:${VERSION}
```

### Push image to quay
```bash
podman push quay.io/${ORGANIZATION_QUAY}/chatbot-proposal-ai:${VERSION}
```