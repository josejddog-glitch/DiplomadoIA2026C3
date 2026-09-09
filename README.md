# Diplomado Inteligencia Artificial Generativa

Material práctico del **Diplomado en Inteligencia Artificial Generativa**.
El proyecto reúne ejemplos de modelos de lenguaje, embeddings, búsqueda
semántica, bases vectoriales, agentes y automatización con correo electrónico.

Los contenidos fueron realizados por el **PhD Oscar Rodriguez Melendez**.
Todos los derechos de autor están reservados. Queda prohibida la reproducción,
distribución o modificación de este material sin autorización del autor.

## Entorno de ejecución

El proyecto está diseñado para ejecutarse localmente en el entorno Conda
`DiplomadoIA`; no requiere Docker.

```bash
conda env update -f environment.yml
conda activate DiplomadoIA
cp secrets/.env.example secrets/.env
```

Completa `secrets/.env` con las credenciales de OpenRouter y Google. El archivo
`secrets/.env` y el token OAuth de Gmail no deben publicarse.

## Contenido del proyecto

### Chats y modelos de lenguaje

- `Chat.py`: chat conversacional básico utilizando `ChatOpenAI` conectado a
  OpenRouter. Lee `OPENAI_API_KEY` desde `secrets/.env`.
- `ChanThinking.py`: chat planificador que analiza una solicitud, construye
  pasos de razonamiento y presenta una respuesta. Usa el mismo proveedor y
  credencial que `Chat.py`.
- `ChatConsultaBase.py`: chat con recuperación aumentada por generación (RAG).
  Busca información relevante en la base Deep Lake, la incorpora al prompt y
  consulta el modelo de lenguaje. También puede conservar memoria resumida.
- `gmail_autoresponder.py`: automatiza la lectura y respuesta de mensajes de
  Gmail. Usa Google OAuth para acceder al buzón y OpenRouter para redactar las
  respuestas en formato JSON.

### Embeddings y búsqueda semántica

- `Embeddings.py`: recibe una lista libre de palabras, frases o párrafos,
  genera sus embeddings con `text-embedding-3-small` y muestra un mapa semántico
  en dos dimensiones usando PCA o t-SNE.
- `SimilitudCoseno.py`: recibe tres textos y calcula sus embeddings y la
  similitud del coseno entre cada par, mostrando una matriz de resultados.
- `TensorialBase.py`: carga `Anillos.pdf`, divide el contenido en fragmentos,
  genera embeddings y crea o sobrescribe la base vectorial Deep Lake.

### Datos y configuración

- `Anillos.pdf`: documento de ejemplo sobre el que se construye la base de
  conocimiento.
- `config.json`: configuración común del PDF, Deep Lake, modelo de embeddings,
  división de texto, recuperación, modelo de lenguaje y memoria.
- `deeplake_db/anillos/`: base vectorial generada a partir de `Anillos.pdf`.
  `TensorialBase.py` la crea y `ChatConsultaBase.py` la utiliza para consultas.
- `environment.yml`: versiones de Python y dependencias utilizadas en el
  entorno `DiplomadoIA`, incluyendo LangChain, Deep Lake, Transformers,
  sentence-transformers, PyTorch y las APIs de Google.
- `secrets/.env.example`: plantilla de variables de entorno necesarias.
- `secrets/.env`: credenciales locales reales; está excluido del repositorio.
- `secrets/token.json`: token OAuth local generado por Gmail; está excluido del
  repositorio.
- `.gitignore`: evita publicar secretos, tokens, cachés y archivos generados.

### Agente 1

`Agente1/` contiene un agente conversacional que clasifica la intención del
usuario y ejecuta una decisión correspondiente.

- `Agente1/src/main.py`: punto de entrada; carga configuración, credenciales,
  prompts y modelo de lenguaje.
- `Agente1/src/brain.py`: coordina el ciclo principal de razonamiento y
  ejecución del agente.
- `Agente1/src/decision.py`: interpreta la decisión elegida y conecta con la
  función correspondiente.
- `Agente1/src/memory.py`: administra la memoria de la conversación.
- `Agente1/config/config.json`: modelo, parámetros y rutas internas del agente.
- `Agente1/prompts/`: prompt de clasificación y personalidad del agente.
- `Agente1/decisions/options.json`: catálogo de opciones que el agente puede
  seleccionar.
- `Agente1/services/call_llm.py`: conecta el agente con el modelo de lenguaje.
- `Agente1/services/extract_json.py`: extrae respuestas JSON del modelo.
- `Agente1/services/load_config.py`, `load_json.py` y `load_text_file.py`:
  cargan configuración, datos JSON y archivos de texto.
- `Agente1/services/decision_functions/`: contiene las funciones deterministas
  `suma.py`, `suma_hasta.py`, `multiplica.py`, `multiplica_hasta.py`,
  `division.py` y `exponencial_hasta.py`.
- `Agente1/README.md`: documentación específica del agente.
- `Agente1/__init__.py`, `Agente1/src/__init__.py` y
  `Agente1/services/__init__.py`: identifican los paquetes Python.

## Relación entre los archivos

El flujo principal de conocimiento es:

```text
Anillos.pdf
    ↓
TensorialBase.py + config.json
    ↓
deeplake_db/anillos/
    ↓
ChatConsultaBase.py → contexto relevante → modelo OpenRouter
```

`Embeddings.py` permite explorar visualmente cualquier conjunto de textos y
`SimilitudCoseno.py` muestra cómo comparar sus representaciones vectoriales.
Ambos utilizan el mismo concepto de embeddings y funcionan como herramientas
independientes de la base Deep Lake.

`Chat.py` y `ChanThinking.py` son chats generales. `ChatConsultaBase.py` añade
la base vectorial y la memoria resumida. `Agente1` utiliza sus propios prompts,
opciones y funciones, pero comparte el entorno Conda, las credenciales y el
modelo de lenguaje configurado para OpenRouter.

`gmail_autoresponder.py` es un flujo independiente: recibe mensajes desde
Gmail, utiliza el modelo para generar una respuesta y la envía nuevamente al
correo mediante la API de Google.

## Ejecución

Desde la raíz del proyecto:

```bash
conda activate DiplomadoIA
python TensorialBase.py       # Crear o actualizar la base vectorial
python Embeddings.py          # Explorar embeddings de una lista libre
python SimilitudCoseno.py     # Comparar tres textos
python Chat.py                # Chat básico
python ChanThinking.py        # Chat planificador
python ChatConsultaBase.py    # Chat con RAG y memoria
python Agente1/src/main.py    # Ejecutar Agente 1
python gmail_autoresponder.py # Autorrespondedor de Gmail
```

Los scripts resuelven sus rutas a partir de la ubicación del proyecto, por lo
que pueden ejecutarse desde la raíz sin depender de rutas absolutas del equipo
original.
