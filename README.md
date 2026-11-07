# Sahaya Emergency Assistant

Voice-first emergency triage and first-aid assistant for rural India.

## Stack
- **Gnani Prisma** — Speech-to-Text (11 Indian languages)
- **Groq** — Free LLM triage reasoning (Llama 3.1 8B)
- **Gnani Timbre** — Text-to-Speech (10 Indian languages)
- **Gradio** — Web UI

## Deploy on Render

1. Push this repo to GitHub.
2. Sign in at [render.com](https://render.com) with GitHub.
3. Click **New → Web Service** → connect your repo.
4. Render auto-detects `render.yaml`. Pick **Docker** as the environment.
5. Under **Environment**, add two secret keys:
   - `GNANI_API_KEY` — your Gnani API key
   - `GROQ_API_KEY` — your free key from [console.groq.com](https://console.groq.com)
6. Click **Deploy**.

The app will be live at `https://sahaya-XXXX.onrender.com`.

## Local Development

```bash
pip install -r requirements.txt
export GNANI_API_KEY=your_key_here
export GROQ_API_KEY=your_key_here
python app.py
```
