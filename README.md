# INDRA — Integrated Neural Design and Research Assistant

> **AI-powered learning and rulebook intelligence platform for Indian students and Formula Student teams.**


---

## What is INDRA?

INDRA started as a rulebook search engine for Formula Bharat teams — a tool to replace `Ctrl+F` with actual intelligence. It has since evolved into something broader.

**INDRA is a curriculum-native AI learning platform** that converts dense technical documents — rulebooks, textbooks, engineering references — into structured, conversational learning experiences. Students don't search. They ask. INDRA answers with precision, context, and when applicable, visual explanation.

Two active use cases:

**1. Formula Bharat / Formula Student Teams**
Ask natural language questions against the 140+ page technical rulebook. Get exact rule references, material tables, and cross-referenced answers in seconds — without breaking your CAD focus.

**2. Indian Students (NCERT & Technical Curricula)**
Learn Physics, Chemistry, Mathematics and engineering concepts through a structured chatbook interface. No prompt engineering required. No PDF dumping. The curriculum is already inside — ask your doubt and get an answer built for your syllabus.

---

## The Problem

### For Formula Student Teams
Designing a Formula Student car requires strict adherence to a 140+ page technical rulebook. Relying on `Ctrl+F` is inefficient — it leads to missed cross-references, misinterpreted material specifications, and frustrating tech inspection failures.

A typical question like:

> *"What is the minimum wall thickness for the Front Hoop using 1020 steel?"*

...requires manually navigating multiple rule sections, cross-referencing material tables, and verifying exceptions. Engineers lose hours to this every week.

### For Indian Students
250 million students study NCERT curricula. Most of them:
- Cannot afford ₹40,000/year coaching institutes
- Learn passively from YouTube videos with no ability to ask doubts
- Struggle with concepts that require visualization — force vectors, thermodynamic cycles, geometric proofs
- Have no access to a teacher who will answer the same doubt five different ways until it clicks

The problem isn't intelligence. It's access.

---

## The Solution

### Architecture — Why INDRA Is Different

Most AI document tools work by dumping entire documents into context on every query. A 200-page rulebook = 30,000+ tokens per message. At 30 questions per session that's nearly 1,000,000 tokens — expensive, slow, and paradoxically less accurate because the model drowns in irrelevant context.

INDRA uses a **Graph RAG architecture with selective node activation:**

```
Document ingested
        ↓
Converted into a knowledge graph — nodes represent concepts, 
edges represent relationships and cross-references
        ↓
Student / engineer asks a question
        ↓
Relevant nodes activated based on semantic similarity
        ↓
Only activated node content sent as context (500–2,000 tokens)
        ↓
SLM answers with precise, minimal, accurate context
```

**Result:** 10–20x token reduction compared to full-document RAG. Higher accuracy because the model only receives relevant information. Lower cost per user. Faster responses.

### For Formula Student Teams — Rulebook Intelligence

INDRA has ingested the official Formula Bharat rulebook. Every rule is classified, cross-referenced, and stored as a node in the knowledge graph.

Ask:
> *"What is the minimum wall thickness for the Front Hoop using 1020 steel?"*

INDRA retrieves rule T 3.2, formats the material table correctly, flags any cross-referenced rules, and gives you a direct answer — in seconds.

### For Students — The Chatbook

INDRA converts textbooks into **chatbooks** — curriculum-native conversational learning experiences.

- NCERT Physics, Chemistry, Mathematics (Class 11 & 12)
- Formula Bharat & SAE rulebooks
- Expanding library under active development

Students ask questions in plain language. No prompt engineering. No PDF uploading. The curriculum is pre-processed, supervised, and structured before it reaches any student.

**Automated few-shot prompting layer:** INDRA maintains 200–300 curated prompt-response pairs across question types. When a student asks a question, it is automatically tagged and enriched with relevant examples before reaching the model — producing consistently high-quality responses without the student knowing any of this is happening.

---

## Key Features

### Currently Live (Phase 0)
- Natural language querying against Formula Bharat rulebook
- Graph RAG pipeline with selective context retrieval
- Automated few-shot prompt enrichment
- Guardrails — responses scoped strictly to curriculum content
- Backend proxy architecture — API keys never exposed to client
- Per-user rate limiting — cost-controlled usage

### In Active Development (Phase 1)
- NCERT chatbook for Class 11 & 12 Physics and Mathematics
- Fine-tuned SLM replacing frontier model for curriculum queries
- Expanded knowledge graph covering cross-subject concept relationships
- Mobile-optimized interface for low-bandwidth environments

### Roadmap (Phase 2–3)
- **Visual explanation engine** — Pre-built static asset library (force vectors, geometric shapes, mechanical components, human figures) assembled dynamically by a composition model trained on thousands of physics and engineering problems
- A student asks about an inclined plane problem — INDRA assembles the correct scene from existing assets rather than generating from scratch, making visual responses fast and accurate
- Hindi and regional language support
- Offline-capable progressive web app for rural access

---

## Who Is This For?

### Formula Student Engineers
- **Technical Directors & Scrutineers** — Verify rules instantly during design reviews
- **Chassis, Powertrain, EV, Aero Leads** — Check keep-out zones and material constraints without leaving CAD
- **New Team Members** — Get up to speed on rulebook structure in days, not weeks

### Indian Students
- **Underprivileged students** in Tier 2/3 cities and rural areas with internet access but no coaching access — INDRA is the first viable learning tool available to them
- **Students who struggle with passive learning** — The 80% of learners who cannot internally visualize what a teacher explains need active engagement. INDRA forces it through conversation
- **JEE / NEET aspirants** who need doubt resolution at 11 PM when no teacher is available

---

## Retention Evidence

The learning methodology behind INDRA is not theoretical.

The founder studied an entire 7-unit Mechanical Engineering Innovation syllabus using INDRA's conversational approach over a single session — asking doubts, challenging explanations, building understanding concept by concept.

Retention test two weeks later: **80% concept recall.**

This aligns with established learning science — active dialogue-based learning produces significantly stronger memory consolidation than passive reading or video watching. INDRA is built around this principle.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite |
| Backend | Node.js (API proxy, rate limiting, auth) |
| Database | Supabase (PostgreSQL + pgvector for embeddings) |
| Knowledge Graph | Python (networkx + custom taxonomy tagging) |
| AI Engine | Google Gemini 1.5 (extraction & chat) + text-embedding-004 |
| Ingestion Pipeline | Python 3 (pypdf, regex-based rule classification) |
| Auth | Supabase Auth |

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/hexawatts/indra.git
cd indra
```

### 2. Install Node Dependencies
```bash
npm install
```

### 3. Environment Variables
```bash
cp .env.example .env
```

Fill in your keys:
```
GOOGLE_AI_API_KEY=your_google_ai_studio_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

You will need:
- A free [Google AI Studio](https://aistudio.google.com) API Key
- A free [Supabase](https://supabase.com) Project URL and Service Role Key

### 4. Database Setup
Navigate to your Supabase project's SQL Editor and run:

```sql
-- Enable vector extension
create extension if not exists vector;

-- Create rules table
create table rules (
  id bigserial primary key,
  rule_id text,
  section text,
  content text,
  tags text[],
  embedding vector(768)
);

-- Create index for fast similarity search
create index on rules using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);
```

### 5. Ingest the Document
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your document in root directory
# For Formula Bharat: FB2027_Rules.pdf
# For NCERT: NCERT_Physics_11.pdf

python ingest.py
```

> Note: Full ingestion takes 1.5–2 hours for a complete rulebook. The pipeline carefully classifies, tags, and embeds each node without hitting free-tier API rate limits. This is a one-time process per document.

### 6. Run the Application
```bash
npm run dev
```

---

## Project Structure

```
indra/
├── frontend/          # React + Vite client
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── hooks/
├── backend/           # Node.js API proxy
│   ├── routes/
│   ├── middleware/    # Auth, rate limiting, guardrails
│   └── server.js
├── pipeline/          # Python ingestion engine
│   ├── ingest.py      # Main ingestion script
│   ├── graph.py       # Knowledge graph construction
│   ├── embed.py       # Embedding generation
│   └── classify.py    # Taxonomy tagging
├── prompts/           # Few-shot prompt pairs library
└── README.md
```

---

## Phases

| Phase | Status | Scope |
|---|---|---|
| Phase 0 | ✅ Live | Formula Bharat RAG pipeline, core chat interface, backend proxy |
| Phase 1 | 🔄 Active Development | NCERT chatbook, SLM fine-tuning, mobile optimization |
| Phase 2 | 📋 Planned | Visual explanation engine, static asset library, composition model |
| Phase 3 | 📋 Planned | Regional language support, offline PWA, expanded curriculum library |

---

## The Bigger Picture

India has 250 million students studying NCERT curricula. A significant portion of them are in cities and villages where quality teaching is inaccessible — but 4G internet is not.

INDRA is not trying to replace classrooms or teachers. The experience of learning from a human in front of you is irreplaceable and INDRA makes no claim otherwise.

INDRA is trying to serve the student for whom that experience was never available in the first place.

A ₹500/year product that runs on a basic Android phone and answers doubts at midnight is not competing with coaching institutes charging ₹40,000/year. It is reaching the student that coaching institute never reached and never will.

That is the actual goal.

---

## Contributing

INDRA is open source because the Formula Student and Indian student communities should own tools built for them.

If your team builds a useful feature — automated rule quiz generation, CAD constraint checking, Hindi language support — open a Pull Request.

```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/your-feature-name

# Commit your changes
git commit -m "Add: your feature description"

# Push to your branch
git push origin feature/your-feature-name

# Open a Pull Request
```

Areas where contributions are most needed:
- Regional language support (Hindi, Tamil, Telugu)
- Additional NCERT subject ingestion pipelines
- Formula Student rulebook variants (FSG, FSUK, FSAE)
- Visual asset library for Phase 2 animation engine

---

## Built By

**Robin 

INDRA started as an internal tool to stop our engineers from losing hours to rulebook navigation. It became something we realized every engineering student in India needed.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

*INDRA — Ask anything. Understand everything.*
