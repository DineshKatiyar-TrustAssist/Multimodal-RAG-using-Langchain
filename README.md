# <h1 style="color:#2b6cb0">Multimodal RAG using LangChain (Multimodal-RAG-using-Langchain)</h1>

This repository demonstrates a small Multimodal Retrieval-Augmented Generation (RAG) pipeline using LangChain, OpenAI models (including GPT-4 / GPT-4 vision), and a FAISS vectorstore. It extracts text, tables and images from PDFs, summarizes them with an LLM, stores the summaries and metadata in FAISS, and exposes a FastAPI web interface to ask questions grounded in the ingested multimodal content.

<h2 style="color:#dd6b20">Contents</h2>
- `multimodal_gpt_x.py` — notebook-style script (also present as notebooks in the repo) that extracts text/tables/images from PDFs, summarizes them using OpenAI models, and builds a FAISS index (`faiss_index/`).
- `app.py` — FastAPI application that loads the FAISS index and provides a simple UI and API (`/` and `/get_answer`) to ask questions against the vectorstore and produce LLM-backed answers.
- `faiss_index/` — saved FAISS index (if present) used by the API.
- `templates/index.html` — minimal UI used by the FastAPI app.

<h2 style="color:#2f855a">Core ideas</h2>
- Input sources: PDFs (text, tables) and extracted images.
- Preprocessing: `unstructured` (PDF partitioning) to split and extract elements.
- Summarization: LLM summaries for text and tables; GPT-4 vision for images (optional in the notebook).
- Vector storage: embeddings via OpenAIEmbeddings + FAISS for similarity search.
- Serving: FastAPI for a small web frontend and a JSON API.

<h2 style="color:#dd6b20">Prerequisites</h2>
- Python 3.10+ recommended.
- An OpenAI API key with access to the required models (GPT-3.5/GPT-4/GPT-4-vision if you plan to use vision features).
- System packages for document/image extraction: on macOS use Homebrew, on Debian/Ubuntu the notebook shows apt commands. Examples:
  - macOS (Homebrew):
    - brew install tesseract
    - brew install poppler
  - Ubuntu/Debian (shown in notebook):
    - sudo apt install tesseract-ocr libtesseract-dev -y
    - sudo apt-get install poppler-utils -y

<h2 style="color:#2f855a">Python dependencies</h2>
- See `requirements.txt` for the canonical list. Key packages include:
  - langchain
  - openai
  - faiss-cpu
  - fastapi, uvicorn, jinja2 (for the web app)
  - python-dotenv (for .env support)
  - unstructured (used in the notebook for PDF partitioning)

<h2 style="color:#dd6b20">Quick setup (macOS / zsh)</h2>
1. Clone the repository and cd into it.
2. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
pip install unstructured[all-docs] pydantic lxml opencv-python
# If you need FAISS on some platforms: pip install faiss-cpu
```

4. Provide your OpenAI API key via environment variable or a `.env` file:

```bash
# zsh example
export OPENAI_API_KEY="sk-..."

# or create a .env file in the repo root with:
# OPENAI_API_KEY=sk-...
```

<h2 style="color:#2f855a">Creating / updating the FAISS index</h2>

The provided notebook/script (`multimodal_gpt_x.py` and the `.ipynb` versions) contains code that:
- loads a PDF using `unstructured.partition.pdf.partition_pdf` (splits into text/table/image elements),
- summarizes each element via a LangChain LLMChain, and
- uses `OpenAIEmbeddings` and `FAISS.from_documents(...).save_local("faiss_index")` to persist the index locally.

If you already have a ready `faiss_index/` folder in the repo, the FastAPI app will load it at startup. If not, run the notebook or adapt `multimodal_gpt_x.py` to process your PDFs and call `vectorstore.save_local("faiss_index")`.

<h2 style="color:#dd6b20">Notes on the notebook script</h2>
- The notebook uses system package installs (apt) and pip installs — those are there because the notebook was developed in a Colab/Ubuntu environment.
- Image summarization in the notebook uses GPT-4 vision (the snippet constructs a base64 data URL and sends it to GPT-4-vision). That requires access to GPT-4-vision and may cost more than text-only calls.

<h2 style="color:#2f855a">Run the FastAPI web app (local development)</h2>
1. Activate your virtualenv and ensure `OPENAI_API_KEY` is set.
2. Start the server (recommended with `uvicorn`):

```bash
# from the repo root
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

3. Open the UI in your browser: http://localhost:8000

<h2 style="color:#dd6b20">API</h2>
- GET / — returns the HTML UI (from `templates/index.html`).
- POST /get_answer — form parameter `question` (string). Returns JSON with two keys:
  - `result` — the LLM answer (string)
  - `relevant_images` — base64-encoded image data (string) for the top relevant image, if any

<h3 style="color:#6b46c1">Example (curl)</h3>

```bash
curl -X POST \
  -F "question=What is Gingivitis?" \
  http://localhost:8000/get_answer

# Example response (JSON):
# {"relevant_images":"/9j/4AAQ...","result":"Gingivitis is ..."}
```

<h2 style="color:#2f855a">How the FastAPI app works (summary)</h2>
- `app.py` loads environment variables (via `python-dotenv`), constructs `OpenAIEmbeddings`, and loads the FAISS vectorstore folder via `FAISS.load_local("faiss_index", embeddings)`.
- Incoming question -> similarity search on FAISS -> build a textual `context` by concatenating relevant document original contents -> pass `context` and `question` to a LangChain `LLMChain` (ChatOpenAI) with a prompt template that asks to act as a vet and answer only within the provided context -> return JSON with the answer and the top relevant image (if any).

<h2 style="color:#dd6b20">Environment variables / configuration</h2>
- OPENAI_API_KEY — required. Provide via environment or `.env`.
- The app expects to find a local `faiss_index/` folder. If missing, the app will fail to load the DB; create it by running the notebook/script that builds the vectorstore.

<h2 style="color:#2f855a">Costs and safety</h2>
- The code uses OpenAI models (GPT-3.5/GPT-4/GPT-4-vision). These calls will incur costs. Monitor usage.
- The prompt explicitly instructs the model to decline when unsure; however, follow-up validation may be required before using outputs in production.

<h2 style="color:#dd6b20">Troubleshooting</h2>
- Error: "FAISS index not found" — ensure `faiss_index/` exists and contains saved index files. Re-run the preprocessing notebook to recreate it.
- Error: missing API key / authentication errors — confirm `OPENAI_API_KEY` is exported or present in `.env`.
- Model errors (rate limits, model not found) — ensure your OpenAI account has access to the requested models and check rate limits or quotas.
- On macOS, replace apt installation instructions from the notebook with Homebrew installs for `tesseract` and `poppler`.

<h2 style="color:#2f855a">Extending this project</h2>
- Add more documents: update the ingestion script to iterate over multiple PDFs and save them into the FAISS index.
- Use a Persistent Vector DB: replace `FAISS` local store with a hosted/persistent vector DB if you need multi-user concurrency or production durability.
- Add authentication to FastAPI and limit access to your API key / endpoints.

<h2 style="color:#dd6b20">Security and privacy</h2>
- Do not commit your OpenAI API key to the repo. Use `.gitignore` to ignore `.env` files.

<h2 style="color:#dd6b20">License</h2>
- Check the repository root for `LICENSE` to verify repository licensing.
