# Búsqueda semántica local de facturas

## Alcance actual

Hay dos modos: **texto** extrae texto del PDF con MarkItDown; **directo** envía el PDF o la imagen a Jina multimodal. En directo se conservan embeddings visuales incluso en PDFs escaneados; el texto extraíble sigue alimentando BM25. El worker almacena originales en Azure Blob Storage (Azurite en desarrollo).

El worker procesa el PDF localmente y envía el Markdown resultante al endpoint configurado de embeddings. Si el endpoint apunta a otro equipo, ese contenido sale del Mac; úsese contenido sintético/anonimizado hasta verificar la ubicación, autorización y controles del servidor. El texto extraído se conserva en Elasticsearch local para búsqueda léxica; no se escribe en logs.

## Prueba del proveedor local

La instancia local actual anuncia `jinaai/jina-embeddings-v5-omni-nano`. Verifiqué consultas de texto, un PDF y una imagen sintéticos; devuelve vectores de **768 dimensiones** y acepta `input_type=query|document`. La inferencia multimodal del PDF de prueba tardó ~175 s en CPU; el timeout predeterminado es 300 s.

Jina local envía PDFs como base64 junto con un `document_id`, e imágenes PNG/JPEG/WEBP como base64. El modo directo está disponible solo para el proveedor Jina local. El proveedor OpenAI-compatible conserva el flujo de texto.

## Ejecución

1. Copiar `.env.example` a `.env`, `.env.runtime.example` a `.env.runtime.app` y `.env.migration.example` a `.env.migration`. Usa contraseñas diferentes para `APP_DB_PASSWORD` y `POSTGRES_PASSWORD`; la contraseña del `DATABASE_URL` debe coincidir con `APP_DB_PASSWORD`. El URL admin solo se entrega a `schema-migrate`; PostgreSQL recibe las variables `POSTGRES_*` necesarias, nunca URLs de conexión ni la contraseña runtime. API/worker reciben solo `.env.runtime.app`. Configurar `EMBEDDING_API_KEY` sin versionar los archivos.
2. Para Jina local en el host, configura `EMBEDDING_PROVIDER=jina-local`, `EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1`, `EMBEDDING_MODEL=jinaai/jina-embeddings-v5-omni-nano`, `EMBEDDING_DOCUMENT_MODE=direct` y `EMBEDDING_DIMENSIONS=768`. En Ajustes elige Jina y modo multimodal directo. Otros proveedores requieren modo texto.
3. Cambiar de modelo/dimensiones requiere un índice Elasticsearch nuevo y reindexar documentos. El ejemplo usa `invoices-poc-jina-omni-v5-768`; la aplicación rechaza un índice cuyo vector no coincide con `EMBEDDING_DIMENSIONS`.
4. Ejecuta `docker compose --env-file .env --env-file .env.migration up --build` y abre `http://127.0.0.1:8080`. `schema-migrate` crea/actualiza esquema y aprovisiona el usuario runtime antes de API/workers. Elasticsearch y Azurite se ejecutan localmente para desarrollo; PostgreSQL guarda la cola y Azurite los PDFs originales.
5. En modo directo se admiten PDFs y PNG/JPEG/WEBP; en modo texto, PDFs con texto extraíble. Buscar conceptos y abrir los originales desde los resultados.

La subida acepta una firma `%PDF-` dentro de los primeros 1024 bytes, para archivos con preámbulo MIME corto; guarda solo desde la firma. La API devuelve un resultado por fichero (`queued`, `duplicate` o `rejected`) para que un archivo inválido no cancele el resto del bloque. `MAX_UPLOAD_MB` se aplica a cada archivo y `MAX_BATCH_FILES` limita el número por petición. La interfaz muestra el envío, los errores HTTP y páginas de 50 documentos.

## API y MCP

Configura `API_TOKEN` en el entorno del servicio con un secreto aleatorio. Todos los endpoints `/api/...` salvo `/api/health`, y el endpoint MCP `/mcp`, requieren `Authorization: Bearer <token>`. La interfaz permite introducirlo y lo conserva solo en `sessionStorage`.

Swagger UI está disponible en `/api/docs`, la especificación OpenAPI en `/api/openapi.json` y ReDoc en `/api/redoc`. En Swagger, pulsa **Authorize** e introduce el token Bearer. La especificación describe la API REST; las herramientas MCP se consultan mediante MCP `tools/list`.

La API REST acepta PDFs e imágenes compatibles como `multipart/form-data` en `POST /api/documents`, usando el campo repetido `files`. Las imágenes se validan como PNG/JPEG/WEBP y requieren Jina en modo directo. Devuelve identificadores para consultar individualmente `GET /api/documents/{id}`. El listado admite `limit` (1-200), `offset` y filtro `status`. La búsqueda de facturas es `POST /api/search`; la recuperación de pasajes de evidencia es `POST /api/search/evidence` con `{query, limit, invoice_id?}`, y los filtros estructurados se envían a `POST /api/invoices/filter` como JSON con campos de factura, cliente, fechas e intervalo de total a pagar.

La interfaz web permite alternar entre facturas, pasajes de evidencia, filtros estructurados y evaluación. Los pasajes muestran offsets sobre texto extraído (no páginas del PDF), score relativo y acceso al original. Los resultados filtrados incluyen la confianza heurística y la evidencia textual disponibles; estos campos no constituyen verificación contable. La interfaz presenta evidencia, pero no genera respuestas: esa responsabilidad corresponde al agente que consume MCP.

La pestaña **Evaluación** guarda los casos etiquetados en `DATA_DIR/evidence-evaluation.json`, permite inspeccionar el texto extraído y añadir citas revisadas manualmente, y ejecuta Recall@5/10 en el backend. Informa por separado (1) resolución de factura en la búsqueda de documentos, (2) recuperación global de pasajes y (3) recuperación de pasajes condicionada por los `invoice_id` gold. La tercera métrica es diagnóstica/oráculo: mide recuperación dentro de la factura una vez conocida, no Recall end-to-end. Los contratos REST son `GET/PUT /api/evaluation/evidence/cases`, `POST /api/evaluation/evidence/run` y `GET /api/documents/{id}/text`. El archivo contiene consultas e incluso citas de facturas: queda en el volumen local `DATA_DIR`, protegido por Bearer en la API; protege también los permisos y las copias de seguridad del volumen. La UI no marca resultados recuperados como relevantes automáticamente.

El servidor MCP usa Streamable HTTP en `/mcp` y publica `index_invoice` (un PDF por llamada, codificado en base64), `get_invoice_status`, `reindex_invoices`, `semantic_search`, `search_invoice_evidence` y `filter_invoices`. Para archivos grandes, usa el endpoint REST multipart: el transporte MCP tiene un límite de tamaño de mensaje del SDK, por lo que no debe usarse para PDFs grandes.

Para desplegar detrás de un proxy HTTPS, conserva `APP_BIND_ADDRESS=127.0.0.1` si el proxy corre en el host. Configúralo como `0.0.0.0` solo si el puerto está protegido por firewall y terminado TLS por el proxy. Elasticsearch se configura mediante `ELASTICSEARCH_URL` y `ELASTICSEARCH_INDEX`; para un clúster protegido se puede usar `ELASTICSEARCH_API_KEY` o `ELASTICSEARCH_USERNAME` junto con `ELASTICSEARCH_PASSWORD`. El servicio de embeddings se apunta mediante `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` y, si aplica, `EMBEDDING_API_KEY`; `EMBEDDING_DIMENSIONS` debe coincidir con la mapping. En despliegue, PDFs e imágenes originales se almacenan en Azure Blob Storage usando `AZURE_STORAGE_CONNECTION_STRING` y `AZURE_STORAGE_CONTAINER`.

### Portainer

Despliega `docker-compose.portainer.yml` desde el repositorio Git para que Portainer pueda construir las imágenes desde el `Dockerfile`. Define `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` como credencial administrativa solo de PostgreSQL/migrador; `APP_DB_USER`, `APP_DB_PASSWORD` para el runtime y `DATABASE_MIGRATION_URL` para el servicio one-shot de migración. `DATABASE_URL` debe contener `APP_DB_USER`/`APP_DB_PASSWORD`, y solo se entrega a API/worker. Añade además `API_TOKEN`, `ELASTICSEARCH_URL`, `ELASTICSEARCH_INDEX`, `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_CONTAINER`, `EMBEDDING_PROVIDER`, `EMBEDDING_DOCUMENT_MODE`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` y las claves opcionales. Para Jina Omni v5 usa un índice nuevo con `EMBEDDING_DIMENSIONS=768`. Usa secretos de Portainer para contraseñas, tokens y connection string. No incluyas `DATABASE_MIGRATION_URL` en el entorno de API/worker. En instalaciones con volumen PostgreSQL preexistente, define y verifica las credenciales runtime nuevas antes de desplegar; no reutilices como `DATABASE_URL` el usuario superuser histórico. Elasticsearch, Azure Blob y embeddings son dependencias externas. Configura `APP_BIND_ADDRESS=0.0.0.0` si el proxy no comparte host y protégelo con firewall/TLS.

La cola transaccional está en PostgreSQL y los originales en Azure Blob Storage; `FOR UPDATE SKIP LOCKED` permite varios workers sin reclamar dos veces un trabajo. La configuración del proveedor se persiste en el volumen compartido `embedding-settings-data` para que app y workers usen el mismo modo.

La API/worker usa el rol `APP_DB_USER`, sin superuser, CREATE de schema, ownership ni membresía en `extraction_retention_operator`. `schema-migrate` usa `DATABASE_MIGRATION_URL` para cambios DDL y aprovisiona el rol runtime con DML mínimo. La retención de runs terminales es una función SECURITY DEFINER propiedad del rol PostgreSQL NOLOGIN `extraction_retention_operator`; un DBA autorizado debe `SET ROLE extraction_retention_operator` y llamar `retire_extraction_run(run_id)`. Triggers exigen tanto el rol efectivo como el marcador transaccional interno de la función. Un GUC solo no concede privilegio. La retención elimina run e hijas en una única transacción; cualquier FK restrictiva revierte todo el borrado.

El worker extrae texto de PDFs digitales con MarkItDown y, mediante reglas deterministas, intenta identificar número/fecha de factura, código y NIF del cliente, albaranes, conceptos, moneda, base imponible, IVA, total de factura y total a pagar. Cada dato extraído incluye evidencia textual y una confianza heurística; los valores ausentes o ambiguos quedan vacíos. Los importes se guardan separados y no implican una decisión sobre qué sumar. El texto se divide en pasajes y se vectoriza por lotes; `search_invoice_evidence` devuelve los pasajes coincidentes con offsets de caracteres sobre el texto extraído (MarkItDown no conserva una ubicación fiable por página). Usa esta herramienta para respuestas fundamentadas en citas. `filter_invoices` permite filtrar por identificadores, conceptos, fechas, moneda e intervalo de total a pagar; `semantic_search` busca facturas completas. Una búsqueda semántica top-K no es exhaustiva y no debe usarse como conjunto para sumar. Ninguna puntuación o confianza es certeza contable. `reindex_invoices` reprocesa una factura por ID o todas las indexadas, necesario para generar pasajes de documentos previos. Antes de marcar una factura como `indexed`, el worker comprueba que todos los pasajes de su generación estén activos; conflictos o publicaciones incompletas dejan el trabajo en `failed`. La limpieza de generaciones inactivas sigue siendo best-effort. La credencial Elasticsearch del servicio necesita gestionar el índice de pasajes y ejecutar consultas de lectura/escritura, `update_by_query` y `delete_by_query` para publicar nuevas generaciones y limpiar fragmentos antiguos.

La extracción AS/400 Shadow es opcional (`AS400_SHADOW_ENABLED=false` por defecto); su persistencia especializada requiere además `AS400_SHADOW_PERSISTENCE_ENABLED=true` y está deshabilitada por defecto. El writer Shadow es experimental, no valida importes y no se conecta a REST/MCP: mantener ambas opciones apagadas en producción hasta completar revisión operativa y funcional. Usa fixtures sintéticos para probarlo.

## Diagnóstico y datos

`docker compose` publica la aplicación y Elasticsearch solo en loopback por defecto. Para acceso desde otro host, pon la API detrás de un proxy HTTPS/firewall y configura `APP_BIND_ADDRESS` según dónde se ejecute el proxy. Elasticsearch usa 1 GiB de heap en desarrollo; modelo e inferencia se ejecutan en servicios externos. Los trabajos viven en PostgreSQL; en desarrollo los PDFs persisten en Azurite y en producción en Azure Blob Storage. Para detener sin borrar datos usa `docker compose --env-file .env --env-file .env.migration down`. `down -v` con esos mismos archivos elimina los volúmenes locales de PostgreSQL, Azurite y Elasticsearch.

Los workers se pueden distribuir en el mismo servidor con `docker compose --env-file .env --env-file .env.migration up --build --scale worker=4`; PostgreSQL reclama cada trabajo de forma atómica para evitar que dos workers lo procesen a la vez. Cada tarea en `processing` tiene una lease temporal y se vuelve a encolar si el worker no la completa en una hora. Azurite es solo para desarrollo local; producción usa la cuenta externa de Azure Blob Storage.

El diagnóstico `/api/health` comprueba Elasticsearch, PostgreSQL, almacenamiento de objetos y la lista de modelos del proveedor, sin generar vectores. Se procesaron las seis facturas indicadas por el usuario: las seis quedaron `indexed`. Una regresión con consulta de frase y código de cliente, cuyo resultado relevante se confirmó con el usuario, pasó a posición 1 (Recall@5 y Recall@10: 1.0). También se validó el flujo extremo a extremo con un PDF sintético. La métrica cubre un único caso etiquetado; no es una evaluación representativa del corpus.

## Pruebas de búsqueda

Pruebas deterministas (sin enviar datos al proveedor):

```sh
docker compose --env-file .env --env-file .env.migration run --rm --no-deps app python -m unittest discover -s tests -v
```

La evaluación offline de extracción estructurada usa exclusivamente PDFs sintéticos/anonimizados y no llama servicios externos:

```sh
docker compose --env-file .env --env-file .env.migration run --rm --no-deps -v "$PWD:/app" app \
  python -m scripts.evaluate_extraction evaluation/structured-extraction-v1.json
```

Escribe `extraction-report.json` y `extraction-report.md` en el directorio actual. Formato del dataset v1 y opciones de salida: `evaluation/README.md`.

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

Para evaluar evidencia, etiqueta citas textuales relevantes por `invoice_id` estable y fragmento esperado, no por `passage_id` (cambia con cada generación):

```json
[
  {"query": "mantenimiento de impresoras", "relevant_evidence": [
    {"invoice_id": "ID-de-factura-etiquetada", "quote": "fragmento exacto que respalda la consulta"}
  ]}
]
```

Con `API_TOKEN` disponible en el entorno y la app activa, ejecuta `python3 scripts/evaluate_evidence.py evidencia.json`. El script consulta `/api/search/evidence` y calcula Recall@5/10 sobre citas: un resultado cuenta cuando pertenece a la factura etiquetada y contiene el fragmento esperado (sin distinguir mayúsculas ni espacios). Informa latencia y metadatos del top 10, pero no imprime el contenido de los pasajes. Las etiquetas deben ser juicios humanos; el ejemplo es un marcador, no un resultado esperado.

## Modos y límites

El modo texto sigue limitado por MarkItDown y no procesa PDFs escaneados sin OCR. El modo Jina directo ya envía PDFs e imágenes al modelo visual; las consultas de texto y documentos visuales deben usar el mismo modelo e índice vectorial. En CPU, la inferencia puede tardar minutos, por lo que el timeout se configura a 300 segundos por defecto.

Fuentes: [MarkItDown](https://github.com/microsoft/markitdown), [Jina Embeddings v5 Omni Nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano).
