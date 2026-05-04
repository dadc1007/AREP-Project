# Docker para FastAPI RAG Project

Este directorio contiene la configuración de Docker para el backend de FastAPI.

## Archivos creados
1. `Dockerfile`: Define la imagen de Docker para la aplicación.
2. `.dockerignore`: Evita que archivos innecesarios (como entornos virtuales o caché) entren en la imagen.
3. `docker-compose.yml`: Orquestador para levantar el servicio de manera sencilla.

## Cómo usar

### Requisitos
- Tener Docker y Docker Compose instalados.
- Tener un archivo `.env` configurado con las credenciales necesarias (OpenAI, Pinecone, AWS).

### Comandos básicos

#### Construir y levantar el contenedor
```bash
docker-compose up --build
```

#### Correr en segundo plano
```bash
docker-compose up -d
```

#### Ver logs
```bash
docker-compose logs -f
```

#### Detener el contenedor
```bash
docker-compose down
```

La API estará disponible en `http://localhost:8000`.
