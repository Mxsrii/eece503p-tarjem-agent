# Tarjem: "Layla", a business assistant agent

EECE 503P, Assignment C3. Layla is the virtual concierge of **Tarjem**, a fictional certified translation bureau in Hamra, Beirut. She answers questions about services, prices and policies, collects customer leads, and logs anything she cannot answer. Built with the OpenAI API and Gradio.

## How it works

Each customer message goes through four stages:

1. **Router.** A separate LLM call classifies the latest message as `general`, `lead` or `feedback` and returns JSON. If that call fails, a keyword heuristic takes over, so the bot never stalls.
2. **Agent with tools.** The reply is generated with a system prompt built from a persona, numbered rules, the business knowledge, and a playbook for the routed intent. Each intent only sees the tools it needs. The model can call tools in a loop (capped at 4 rounds per turn).
3. **Tools.**
   - `record_customer_interest(email, name, message)` validates the email and name, then saves a lead to `logs/leads.jsonl` and prints it to the console.
   - `record_feedback(question)` saves unanswered questions, complaints and suggestions to `logs/feedback.jsonl`.
4. **Reflection (bonus).** A reviewer call checks the draft reply against the rules (invented facts, promises of embassy acceptance, legal advice, missing email request, persona) and rewrites it if needed. Drafts and revisions are logged to `logs/reflection.jsonl`.

Short-term memory is the last 10 messages of the conversation. The Gradio UI is a clean customer chat; the router, tool and reflection details are printed to the console and written to `logs/`, not shown to the customer.

## Project layout

| Path | Purpose |
|---|---|
| `business_agent.ipynb` | Step-by-step notebook: knowledge loading, persona, tools, router, agent loop, reflection, tests, Gradio UI |
| `app.py` | The same agent as a standalone Gradio app (used for Hugging Face Spaces) |
| `me/about_business.pdf` | Business profile (read with `pypdf`) |
| `me/business_summary.txt` | Business summary: services, prices, policies, team |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for your API key |
| `logs/` | Created at runtime (leads, feedback, router, reflection and error logs). Gitignored |

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env            # macOS/Linux: cp .env.example .env
# edit .env and set OPENAI_API_KEY

python app.py                     # opens the Gradio app at http://127.0.0.1:7860
```

Or open `business_agent.ipynb` and run the cells top to bottom.

The model defaults to `gpt-4.1`. Set `OPENAI_MODEL` in `.env` to use another one. The router and reflection calls use `temperature=0`, so models that reject a custom temperature (the gpt-5 family) need that argument removed.

## Deploy to Hugging Face Spaces

1. Create a new Space at huggingface.co/new-space with the **Gradio** SDK, then push this repository's files to it (it must contain `app.py`, `requirements.txt`, `README.md` and the `me/` folder).
2. Put this configuration block at the very top of the **Space's** `README.md`. Hugging Face reads it to know how to run the app. It is not kept in the GitHub README because GitHub would show it as a table.

   ```yaml
   ---
   title: Tarjem - Layla
   emoji: 📄
   colorFrom: green
   colorTo: blue
   sdk: gradio
   sdk_version: 6.30.0
   app_file: app.py
   pinned: false
   short_description: Virtual concierge for a certified translation bureau
   ---
   ```

3. In the Space, open **Settings > Variables and secrets** and add a **secret** named `OPENAI_API_KEY`. Optionally add a variable `OPENAI_MODEL`.
4. The Space builds and starts automatically. Do not upload `.env`.

The Space's disk is temporary, so `logs/` is cleared whenever it restarts. Leads and feedback are also printed to the Space's container log.

## Try these messages

- "What languages do you translate?" (general)
- "How much for a 2-page diploma, Arabic to English?" (price arithmetic from published rates)
- "I need my birth certificate translated into French for a visa." (lead; she asks for the missing details and your email)
- "Will the Canadian embassy accept your translation?" (she will not promise acceptance, and logs the question)
- "My last order arrived a day late and I'm not happy." (feedback)

## Design notes

- **Grounding.** Layla answers only from the business files. Anything else is logged with `record_feedback` instead of guessed.
- **Safety rules.** She never promises acceptance by an authority, gives legal or immigration advice, or collects passport, ID or payment details in chat.
- **Privacy.** `.env` and `logs/` are gitignored because logs contain user-entered names and emails.
- **Language.** English only. Tarjem itself works in Arabic, English and French, but the bot politely declines other languages for now.
