# Búsqueda semántica local de facturas

## Alcance actual

Hay dos modos: **texto** extrae texto del PDF con MarkItDown; **directo** envía el PDF o la imagen a Jina multimodal. En directo se conservan embeddings visuales incluso en PDFs escaneados; el texto extraíble sigue alimentando BM25. El worker almacena originales en Azure Blob Storage (Azurite en desarrollo).

El worker procesa el PDF localmente y envía el Markdown resultante al endpoint configurado de embeddings. Si el endpoint apunta a otro equipo, ese contenido sale del Mac; úsese contenido sintético/anonimizado hasta verificar la ubicación, autorización y controles del servidor. El texto extraído se conserva en Elasticsearch local para búsqueda léxica; no se escribe en logs.

## Prueba del proveedor local

La instancia local actual anuncia `jinaai/jina-embeddings-v5-omni-nano`. Verifiqué consultas de texto, un PDF y una imagen sintéticos; devuelve vectores de **768 dimensiones** y acepta `input_type=query|document`. La inferencia multimodal del PDF de prueba tardó ~175 s en CPU; el timeout predeterminado es 300 s.

Jina local envía PDFs como base64 junto con un `document_id`, e imágenes PNG/JPEG/WEBP como base64. El modo directo está disponible solo para el proveedor Jina local. El proveedor OpenAI-compatible conserva el flujo de texto.

## Ejecución

1. Copiar `.env.example` a `.env` y `.env.runtime.example` a `.env.runtime`. Reemplazar los secretos de API y PostgreSQL por valores aleatorios; la contraseña incluida en `DATABASE_URL` debe coincidir con `POSTGRES_PASSWORD`. Configurar `EMBEDDING_API_KEY` sin versionar los archivos.
2. Para Jina local en el host, configura `EMBEDDING_PROVIDER=jina-local`, `EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1`, `EMBEDDING_MODEL=jinaai/jina-embeddings-v5-omni-nano`, `EMBEDDING_DOCUMENT_MODE=direct` y `EMBEDDING_DIMENSIONS=768`. En Ajustes elige Jina y modo multimodal directo. Otros proveedores requieren modo texto.
3. Cambiar de modelo/dimensiones requiere un índice Elasticsearch nuevo y reindexar documentos. El ejemplo usa `invoices-poc-jina-omni-v5-768`; la aplicación rechaza un índice cuyo vector no coincide con `EMBEDDING_DIMENSIONS`.
4. Ejecuta `docker compose up --build` y abre `http://127.0.0.1:8080`. Elasticsearch y Azurite se ejecutan localmente para desarrollo; PostgreSQL guarda la cola y Azurite los PDFs originales.
5. En modo directo se admiten PDFs y PNG/JPEG/WEBP; en modo texto, PDFs con texto extraíble. Buscar conceptos y abrir los originales desde los resultados.

La subida acepta una firma `%PDF-` dentro de los primeros 1024 bytes, para archivos con preámbulo MIME corto; guarda solo desde la firma. La API devuelve un resultado por fichero (`queued`, `duplicate` o `rejected`) para que un archivo inválido no cancele el resto del bloque. `MAX_UPLOAD_MB` se aplica a cada archivo y `MAX_BATCH_FILES` limita el número por petición. La interfaz muestra el envío, los errores HTTP y páginas de 50 documentos.

## API y MCP

Configura `API_TOKEN` en el entorno del servicio con un secreto aleatorio. Todos los endpoints `/api/...` salvo `/api/health`, y el endpoint MCP `/mcp`, requieren `Authorization: Bearer <token>`. La interfaz permite introducirlo y lo conserva solo en `sessionStorage`.

Swagger UI está disponible en `/api/docs`, la especificación OpenAPI en `/api/openapi.json` y ReDoc en `/api/redoc`. En Swagger, pulsa **Authorize** e introduce el token Bearer. La especificación describe la API REST; las herramientas MCP se consultan mediante MCP `tools/list`.

La API REST acepta PDFs e imágenes compatibles como `multipart/form-data` en `POST /api/documents`, usando el campo repetido `files`. Las imágenes se validan como PNG/JPEG/WEBP y requieren Jina en modo directo. Devuelve identificadores para consultar individualmente `GET /api/documents/{id}`. El listado admite `limit` (1-200), `offset` y filtro `status`. La búsqueda es `POST /api/search`.

El servidor MCP usa Streamable HTTP en `/mcp` y publica las herramientas `index_invoice` (un PDF por llamada, codificado en base64), `get_invoice_status` y `semantic_search`. Para archivos grandes, usa el endpoint REST multipart: el transporte MCP tiene un límite de tamaño de mensaje del SDK, por lo que no debe usarse para PDFs grandes.

Para desplegar detrás de un proxy HTTPS, conserva `APP_BIND_ADDRESS=127.0.0.1` si el proxy corre en el host. Configúralo como `0.0.0.0` solo si el puerto está protegido por firewall y terminado TLS por el proxy. Elasticsearch se configura mediante `ELASTICSEARCH_URL` y `ELASTICSEARCH_INDEX`; para un clúster protegido se puede usar `ELASTICSEARCH_API_KEY` o `ELASTICSEARCH_USERNAME` junto con `ELASTICSEARCH_PASSWORD`. El servicio de embeddings se apunta mediante `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` y, si aplica, `EMBEDDING_API_KEY`; `EMBEDDING_DIMENSIONS` debe coincidir con la mapping. En despliegue, PDFs e imágenes originales se almacenan en Azure Blob Storage usando `AZURE_STORAGE_CONNECTION_STRING` y `AZURE_STORAGE_CONTAINER`.

### Portainer

Despliega `docker-compose.portainer.yml` desde el repositorio Git para que Portainer pueda construir las imágenes desde el `Dockerfile`. Define en las variables del Stack: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `DATABASE_URL`, `API_TOKEN`, `ELASTICSEARCH_URL`, `ELASTICSEARCH_INDEX`, `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_CONTAINER`, `EMBEDDING_PROVIDER`, `EMBEDDING_DOCUMENT_MODE`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, y las claves opcionales de Elasticsearch/embeddings. Para Jina Omni v5 usa un índice nuevo con `EMBEDDING_DIMENSIONS=768`; la dimensión debe coincidir con la mapping del índice. Usa secretos de Portainer para contraseñas, tokens y connection string. `DATABASE_URL` debe apuntar al servicio `postgres` del stack y su contraseña debe coincidir con `POSTGRES_PASSWORD`. Elasticsearch, Azure Blob y el servicio de embeddings son dependencias externas accesibles desde la red Docker. Configura `APP_BIND_ADDRESS=0.0.0.0` si el proxy no comparte el host; protege entonces el puerto con firewall y TLS en el proxy.

La cola transaccional está en PostgreSQL y los originales en Azure Blob Storage; `FOR UPDATE SKIP LOCKED` permite varios workers sin reclamar dos veces un trabajo. La configuración del proveedor se persiste en el volumen compartido `embedding-settings-data` para que app y workers usen el mismo modo.

El worker indexa un vector y, si está disponible, texto extraído en Elasticsearch. La búsqueda combina BM25 y kNN cosine con RRF calculado en la aplicación (la licencia Basic de este nodo no permite RRF server-side); así los códigos/frases exactas se combinan con la similitud semántica. La puntuación RRF es un orden relativo, no certeza. No hay filtros CIF/cliente/fecha ni extracción de metadatos del nombre todavía.

## Diagnóstico y datos

`docker compose` publica la aplicación y Elasticsearch solo en loopback por defecto. Para acceso desde otro host, pon la API detrás de un proxy HTTPS/firewall y configura `APP_BIND_ADDRESS` según dónde se ejecute el proxy. Elasticsearch usa 1 GiB de heap en desarrollo; modelo e inferencia se ejecutan en servicios externos. Los trabajos viven en PostgreSQL; en desarrollo los PDFs persisten en Azurite y en producción en Azure Blob Storage. `docker compose down -v` elimina los datos locales de PostgreSQL, Azurite y Elasticsearch.

Los workers se pueden distribuir en el mismo servidor con `docker compose up --build --scale worker=4`; PostgreSQL reclama cada trabajo de forma atómica para evitar que dos workers lo procesen a la vez. Cada tarea en `processing` tiene una lease temporal y se vuelve a encolar si el worker no la completa en una hora. Azurite es solo para desarrollo local; producción usa la cuenta externa de Azure Blob Storage.

El diagnóstico `/api/health` comprueba Elasticsearch, PostgreSQL, almacenamiento de objetos y la lista de modelos del proveedor, sin generar vectores. Se procesaron las seis facturas indicadas por el usuario: las seis quedaron `indexed`. Una regresión con consulta de frase y código de cliente, cuyo resultado relevante se confirmó con el usuario, pasó a posición 1 (Recall@5 y Recall@10: 1.0). También se validó el flujo extremo a extremo con un PDF sintético. La métrica cubre un único caso etiquetado; no es una evaluación representativa del corpus.

## Pruebas de búsqueda

Pruebas deterministas (sin enviar datos al proveedor):

```sh
docker compose run --rm --no-deps app python -m unittest discover -s tests -v
```

Prueba real de humo contra el servicio local y oMLX (envía solo las consultas de ejemplo): `python3 scripts/smoke_search.py`.

Para medir calidad sobre las facturas ya indexadas, crea un JSON con consultas y nombres de archivo que el usuario haya etiquetado como relevantes:

```json
[
  {"query": "mantenimiento de impresoras", "relevant": ["reemplazar-con-filename-etiquetado.pdf"]},
  {"query": "compra de material informático", "relevant": ["reemplazar-con-otro-filename-etiquetado.pdf"]}
]
```

Los filenames anteriores son marcadores de posición, no resultados esperados.

Después, con la aplicación activa, ejecuta `python3 scripts/evaluate_search.py casos.json`. Informa Recall@5/10, latencia, top 10 y puede exigir `expected_top1`. Hay un caso de regresión etiquetado; añade más juicios humanos para medir generalización, no interpretes un único caso como evaluación representativa.

## Modos y límites

El modo texto sigue limitado por MarkItDown y no procesa PDFs escaneados sin OCR. El modo Jina directo ya envía PDFs e imágenes al modelo visual; las consultas de texto y documentos visuales deben usar el mismo modelo e índice vectorial. En CPU, la inferencia puede tardar minutos, por lo que el timeout se configura a 300 segundos por defecto.

Fuentes: [MarkItDown](https://github.com/microsoft/markitdown), [Jina Embeddings v5 Omni Nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano).
