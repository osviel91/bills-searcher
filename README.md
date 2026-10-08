# Búsqueda semántica local de facturas

## Alcance actual

Esta iteración avanza con el modelo que ya responde en oMLX: `embedding` (Qwen3-Embedding, 1024 dimensiones). El flujo es **PDF con capa de texto → MarkItDown → embedding de texto → Elasticsearch**. No es búsqueda multimodal del PDF: se deja de representar la apariencia visual del documento. MarkItDown usa extracción textual/tabular PDF (pdfplumber/pdfminer); OCR está desactivado y un PDF escaneado sin capa de texto terminará en `failed`.

El worker procesa el PDF localmente y envía el Markdown resultante al endpoint configurado de embeddings. Si el endpoint apunta a otro equipo, ese contenido sale del Mac; úsese contenido sintético/anonimizado hasta verificar la ubicación, autorización y controles del servidor. El texto extraído se conserva en Elasticsearch local para búsqueda léxica; no se escribe en logs.

## Comprobación de oMLX

La instancia remota anuncia el alias `embedding`. Una llamada real de consulta con texto sintético devolvió un vector numérico de **1024 dimensiones**. El alias Jina Omni también aparece en `/v1/models`, pero oMLX responde HTTP 409 al cargarlo (`Model type jina_omni_small_mlx not supported`). No se usó Jina en esta ruta.

El prompt de consulta sigue el formato recomendado por Qwen3 Embedding (`Instruct: ...\nQuery:`); el documento se envía sin prefijo de consulta. El cliente desactiva truncamiento y fija 32768 tokens, el máximo publicado del modelo, para fallar antes que indexar silenciosamente texto recortado.

## Ejecución

1. Copiar `.env.example` a `.env` y `.env.runtime.example` a `.env.runtime`. Reemplazar los secretos de API y PostgreSQL por valores aleatorios; la contraseña incluida en `DATABASE_URL` debe coincidir con `POSTGRES_PASSWORD`. Configurar el endpoint de embeddings y `EMBEDDING_API_KEY`. No versionar ninguno de los dos archivos.
2. Para oMLX en el Mac, usar `EMBEDDING_BASE_URL=http://host.docker.internal:8800/v1`. Si se usa el servidor remoto autorizado, poner su URL HTTPS; el Markdown extraído se enviará allí.
3. Confirmar `EMBEDDING_MODEL=embedding` o cambiarlo por el ID exacto que acepte `/v1/embeddings`.
4. Ejecuta `docker compose up --build` y abre `http://127.0.0.1:8080`. Elasticsearch y Azurite se ejecutan localmente para desarrollo; PostgreSQL guarda la cola y Azurite los PDFs originales.
5. Subir PDFs digitales. Los escaneados se marcan como fallidos con mensaje de OCR desactivado. Buscar conceptos en lenguaje natural y abrir los PDFs originales desde los resultados.

La subida acepta una firma `%PDF-` dentro de los primeros 1024 bytes, para archivos con preámbulo MIME corto; guarda solo desde la firma. La API devuelve un resultado por fichero (`queued`, `duplicate` o `rejected`) para que un archivo inválido no cancele el resto del bloque. `MAX_UPLOAD_MB` se aplica a cada PDF y `MAX_BATCH_FILES` limita el número por petición. La interfaz muestra el envío, los errores HTTP y páginas de 50 documentos.

## API y MCP

Configura `API_TOKEN` en el entorno del servicio con un secreto aleatorio. Todos los endpoints `/api/...` salvo `/api/health`, y el endpoint MCP `/mcp`, requieren `Authorization: Bearer <token>`. La interfaz permite introducirlo y lo conserva solo en `sessionStorage`.

Swagger UI está disponible en `/api/docs`, la especificación OpenAPI en `/api/openapi.json` y ReDoc en `/api/redoc`. En Swagger, pulsa **Authorize** e introduce el token Bearer. La especificación describe la API REST; las herramientas MCP se consultan mediante MCP `tools/list`.

La API REST acepta PDFs como `multipart/form-data` en `POST /api/documents`, usando el campo repetido `files`. Devuelve identificadores para consultar individualmente `GET /api/documents/{id}`. El listado admite `limit` (1-200), `offset` y filtro `status`. La búsqueda es `POST /api/search`.

El servidor MCP usa Streamable HTTP en `/mcp` y publica las herramientas `index_invoice` (un PDF por llamada, codificado en base64), `get_invoice_status` y `semantic_search`. Para archivos grandes, usa el endpoint REST multipart: el transporte MCP tiene un límite de tamaño de mensaje del SDK, por lo que no debe usarse para PDFs grandes.

Para desplegar detrás de un proxy HTTPS, conserva `APP_BIND_ADDRESS=127.0.0.1` si el proxy corre en el host. Configúralo como `0.0.0.0` solo si el puerto está protegido por firewall y terminado TLS por el proxy. Elasticsearch se configura mediante `ELASTICSEARCH_URL` y `ELASTICSEARCH_INDEX`; para un clúster protegido se puede usar `ELASTICSEARCH_API_KEY` o `ELASTICSEARCH_USERNAME` junto con `ELASTICSEARCH_PASSWORD`. El servicio de embeddings se apunta mediante `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` y, si aplica, `EMBEDDING_API_KEY`; el índice vectorial actual requiere 1024 dimensiones. En despliegue, los PDFs se almacenan en Azure Blob Storage usando `AZURE_STORAGE_CONNECTION_STRING` y `AZURE_STORAGE_CONTAINER`.

### Portainer

Despliega `docker-compose.portainer.yml` desde el repositorio Git para que Portainer pueda construir las imágenes desde el `Dockerfile`. Define en las variables del Stack: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `DATABASE_URL`, `API_TOKEN`, `ELASTICSEARCH_URL`, `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_CONTAINER`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, y las claves opcionales de Elasticsearch/embeddings. Usa secretos de Portainer para contraseñas, tokens y connection string. `DATABASE_URL` debe apuntar al servicio `postgres` del stack y su contraseña debe coincidir con `POSTGRES_PASSWORD`. Elasticsearch y Azure Blob son dependencias externas y deben ser accesibles desde la red Docker. Configura `APP_BIND_ADDRESS=0.0.0.0` si el proxy no comparte el host; protege entonces el puerto con firewall y TLS en el proxy.

La cola transaccional está en PostgreSQL y los originales en Azure Blob Storage; `FOR UPDATE SKIP LOCKED` permite varios workers sin reclamar dos veces un trabajo.

El worker indexa un vector y el Markdown extraído por factura en Elasticsearch. La búsqueda combina BM25 y kNN cosine con RRF calculado en la aplicación (la licencia Basic de este nodo no permite RRF server-side); así los códigos/frases exactas se combinan con la similitud semántica. La puntuación RRF es un orden relativo, no certeza. No hay filtros CIF/cliente/fecha ni extracción de metadatos del nombre todavía.

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

## Diferencia con el requisito inicial

Esta ruta sustituye la entrada multimodal por extracción de texto del PDF. Es útil para avanzar con facturas digitales, pero **no satisface el requisito original de embeddings visuales del PDF**, especialmente para escaneos, sellos, distribución visual o información gráfica. MarkItDown declara que su salida es Markdown para análisis de texto, no una conversión visual fiel. Para volver a la aceptación multimodal hace falta un endpoint que procese PDF/páginas visuales con una ruta de modelo documentada y validada.

Fuentes: [MarkItDown](https://github.com/microsoft/markitdown), [implementación del conversor PDF](https://github.com/microsoft/markitdown/blob/main/packages/markitdown/src/markitdown/converters/_pdf_converter.py), [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Jina Omni MLX](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval-mlx).
