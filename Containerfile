# Use a base Python image
FROM registry.redhat.io/ubi9/python-312:latest

# 1. Cambiar a root temporalmente para poder modificar permisos
USER root

# 2. Establecer el directorio
WORKDIR /app

# 3. Copiar tus archivos
COPY . /app

# 4. Ahora sí, el comando funcionará porque eres root
RUN chgrp -R 0 /app && \
    chmod -R g=u /app

# 5. Instalar dependencias (si no lo has hecho ya)
RUN pip install -r requirements.txt

# 6. Volver a un usuario sin privilegios (REQUISITO PARA OPENSHIFT)
USER 1001

# Define the command to run your application
CMD ["chainlit", "run", "chatbot_proposal.py", "--headless", "--host", "0.0.0.0", "--port", "7861"]