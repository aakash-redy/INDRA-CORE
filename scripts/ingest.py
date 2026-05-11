# ==========================================
# INDRA - INGESTION ENGINE v8.0
# ==========================================
# Changes from v7.0:
#   - Switched embedding model from Gemini to Nomic (nomic-embed-text-v1.5)
#   - Nomic outputs 768-dim vectors — Supabase schema unchanged
#   - Gemini API completely removed from ingestion (zero quota usage at ingest)
#   - NOMIC_API_KEY added to required env vars
#   - FREE_TIER_SLEEP removed — Nomic free tier is generous, no hard RPM cap
#   - Retry logic updated for Nomic API error responses
#   - Task type updated to "search_document" (Nomic convention)
#   - Everything else unchanged — PDF parsing, table detection, domain taxonomy
# ==========================================

import os
import re
import json
import time
import logging
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client
import pypdf

# ==========================================
# 1. SYSTEM SETUP & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR   = SCRIPT_DIR.parent
ENV_PATH   = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

_required_keys = ["NOMIC_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
if not all(os.getenv(k) for k in _required_keys):
    logger.error(
        "❌ FATAL: Missing one or more keys in .env\n"
        "   Required: NOMIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY"
    )
    exit(1)

NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
NOMIC_API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL   = "nomic-embed-text-v1.5"
NOMIC_DIMS    = 768   # matches existing Supabase pgvector column — no schema change needed

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

RULEBOOK_YEAR    = "2027"
NOMIC_BATCH_SIZE = 50   # Nomic supports batching — send up to 50 texts per call
                        # massively faster than one-by-one Gemini calls

# ==========================================
# 2. NOMIC EMBEDDING FUNCTION
# ==========================================

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using Nomic Atlas API.
    Task type 'search_document' is correct for rulebook content being stored.
    Returns a list of 768-dim float vectors matching Supabase schema.
    Retries up to 3 times with exponential backoff on failure.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                NOMIC_API_URL,
                headers={
                    "Authorization": f"Bearer {NOMIC_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": NOMIC_MODEL,
                    "texts": texts,
                    "task_type": "search_document",
                    "dimensionality": NOMIC_DIMS,
                },
                timeout=30,
            )

            if response.status_code == 429:
                wait = 60
                logger.warning(f"⏳ Nomic rate limit hit — cooling down {wait}s…")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                raise ValueError(
                    f"Nomic API error {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            embeddings = data.get("embeddings", [])

            if not embeddings or len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
                )

            # Validate dimensions
            if len(embeddings[0]) != NOMIC_DIMS:
                raise ValueError(
                    f"Dimension mismatch: got {len(embeddings[0])}, expected {NOMIC_DIMS}"
                )

            return embeddings

        except requests.exceptions.Timeout:
            logger.warning(f"⚠️  Nomic timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 2)
            else:
                raise

        except Exception as e:
            logger.warning(
                f"⚠️  Embed attempt {attempt + 1}/{max_retries} failed: {str(e)[:120]}"
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 2)
            else:
                raise

    raise RuntimeError("embed_texts: all retries exhausted")


def embed_single(text: str) -> list[float]:
    """Convenience wrapper for single text embedding."""
    return embed_texts([text])[0]

# ==========================================
# 3. SHARED REGEX (single source of truth)
# ==========================================
RULE_ID_RE = re.compile(r'([A-Z]{1,3}\d*\.\d+(?:\.\d+)*)')

# ==========================================
# 4. DOMAIN & SUBDOMAIN TAXONOMY
# ==========================================

DOMAIN_MAP = {
    "A":  "Administrative",
    "T":  "General_Technical",
    "EV": "Electric_Vehicle",
    "CV": "Combustion_Vehicle",
    "IN": "Inspection",
    "S":  "Static_Events",
    "D":  "Dynamic_Events",
}

SUBDOMAIN_KEYWORDS: list[tuple[list[str], str]] = [
    (["chassis", "frame", "monocoque", "roll hoop", "roll bar",
      "impact attenuator", "sis"],                                          "Chassis"),
    (["suspension", "wishbone", "damper", "spring", "upright",
      "kinematics", "anti-roll"],                                           "Suspension"),
    (["steering", "rack", "toe", "ackermann", "pinion", "tie rod"],        "Steering"),
    (["brake", "braking", "caliper", "rotor", "pedal",
      "overtravel", "master cylinder"],                                     "Brakes"),
    (["aero", "aerodynamic", "wing", "diffuser", "downforce",
      "splitter", "undertray", "drs"],                                      "Aerodynamics"),
    (["cockpit", "ergonomic", "harness", "seat", "headrest",
      "driver", "firewall", "egress"],                                      "Cockpit_and_Ergonomics"),
    (["fastener", "bolt", "weld", "material", "composite",
      "carbon", "nyloc", "safety wire"],                                    "Fasteners_and_Materials"),
    (["engine", "powertrain", "throttle", "intake",
      "exhaust", "fuel", "restrictor"],                                     "Powertrain"),
    (["drivetrain", "differential", "axle", "gearbox",
      "driveshaft", "chain", "belt"],                                       "Drivetrain"),
    (["cool", "radiator", "coolant", "thermal",
      "heat exchanger", "water pump", "fan"],                               "Cooling"),
    (["accumulator", "battery", "cell", "bms", "soc", "container"],        "Accumulator"),
    (["tractive", "hv", "high voltage", "inverter",
      "tsal", "imd", "motor controller"],                                   "Tractive_System"),
    (["ecu", "electronic", "wiring", "harness", "sensor",
      "can bus", "glv", "dashboard"],                                       "Electronics"),
    (["software", "control", "firmware", "algorithm",
      "pid", "vcu", "torque vectoring"],                                    "Software_and_Control"),
    (["dynamics", "handling", "lap time", "skidpad",
      "autocross", "tire data"],                                            "Vehicle_Dynamics"),
    (["safety", "fire", "extinguisher", "fia", "protection",
      "guard", "suit", "helmet"],                                           "Safety_Equipment"),
    (["scrutineer", "inspection", "scrutineering",
      "technical check", "test tool"],                                      "Scrutineering"),
]

def detect_domain(rule_id: str) -> str:
    m = re.match(r'^([A-Z]{1,3})', rule_id)
    return DOMAIN_MAP.get(m.group(1), "Unknown") if m else "Unknown"

def detect_subdomain(text: str) -> str:
    lower = text.lower()
    for keywords, subdomain in SUBDOMAIN_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return subdomain
    return "General"

# ==========================================
# 5. TEXT UTILITIES
# ==========================================

def sanitize_text(text: str) -> str:
    text = text.replace('\x00', '')
    return re.sub(r'[ \t]+', ' ', text).strip()

def format_rule(text: str, max_chars: int = 4000) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= max_chars else cleaned[:max_chars] + " [TRUNCATED]"

# ==========================================
# 6. TABLE DETECTION & PARSING
# ==========================================

_TABLE_LINE_RE = re.compile(r'(\|.+\||\t.+\t|(?:\S+\s{2,}){2,}\S+)')

def _is_table_line(line: str) -> bool:
    return bool(_TABLE_LINE_RE.search(line))

def _split_row(line: str) -> list[str]:
    if '|' in line:
        return [c.strip() for c in line.strip().strip('|').split('|') if c.strip()]
    normalised = re.sub(r'\s{2,}', '\t', line.strip())
    return [c.strip() for c in normalised.split('\t') if c.strip()]

def _parse_table_lines(lines: list[str]) -> list[dict]:
    if not lines:
        return []
    header = _split_row(lines[0])
    if not header:
        return []
    rows: list[dict] = []
    for line in lines[1:]:
        cols = _split_row(line)
        if not cols:
            continue
        if len(cols) < len(header) and rows:
            for i, col_text in enumerate(cols):
                if i < len(header) and col_text:
                    key = header[i]
                    rows[-1][key] = f"{rows[-1].get(key, '')} {col_text}".strip()
            continue
        cols = (cols + [''] * len(header))[:len(header)]
        rows.append(dict(zip(header, cols)))
    return rows


class TableBuffer:
    def __init__(self):
        self._lines: list[str] = []
        self._active           = False
        self._rule_id_at_start = "General Context"
        self._completed: list[tuple[str, list[dict]]] = []

    def feed_line(self, line: str, current_rule_id: str):
        if line.startswith("__PAGE_BREAK_"):
            return
        if _is_table_line(line):
            if not self._active:
                self._active           = True
                self._rule_id_at_start = current_rule_id
            self._lines.append(line)
        else:
            if self._active:
                self._flush()

    def end_of_document(self):
        if self._active:
            self._flush()

    def _flush(self):
        rows = _parse_table_lines(self._lines)
        if rows:
            self._completed.append((self._rule_id_at_start, rows))
            logger.info(
                f"📊 Table captured → rule [{self._rule_id_at_start}]: "
                f"{len(rows)} rows × {len(rows[0])} cols"
            )
        self._lines  = []
        self._active = False

    def pop_completed(self) -> list[tuple[str, list[dict]]]:
        result = list(self._completed)
        self._completed.clear()
        return result

# ==========================================
# 7. CORE INGESTION FUNCTION
# ==========================================

def ingest_domain(
    file_path:     Path,
    domain:        str,
    start_page:    int,
    end_page:      Optional[int] = None,
    force_refresh: bool = False,
):
    logger.info(f"🏁 INDRA INGESTION ENGINE v8.0 — targeting [{domain}]")
    logger.info(f"🧠 Embedding model : {NOMIC_MODEL} ({NOMIC_DIMS} dims)")
    logger.info(f"📦 Batch size      : {NOMIC_BATCH_SIZE} rules per API call")

    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return

    # ── SMART SYNC ──────────────────────────────────────────────────────────
    existing_rules: set[str] = set()
    if force_refresh:
        logger.warning(f"🧹 FORCE REFRESH: wiping all records for '{domain}'…")
        supabase.table("rulebook_chunks").delete().eq("source_domain", domain).execute()
    else:
        logger.info("🔍 Fetching existing rule IDs to prevent duplicates…")
        try:
            resp = supabase.table("rulebook_chunks") \
                           .select("rule_id") \
                           .eq("source_domain", domain) \
                           .execute()
            existing_rules = {row['rule_id'] for row in resp.data}
            logger.info(f"🛡️  {len(existing_rules)} existing rules found — will be skipped.")
        except Exception as e:
            logger.error(f"⚠️  Could not fetch existing rules: {e}")

    # ── STEP 1: PDF EXTRACTION ───────────────────────────────────────────────
    try:
        reader      = pypdf.PdfReader(file_path, strict=False)
        total_pages = len(reader.pages)
        final_page  = min(end_page, total_pages) if end_page else total_pages

        all_lines: list[str] = []
        for i in range(start_page - 1, final_page):
            page_text = reader.pages[i].extract_text() or ""
            all_lines.extend(page_text.splitlines())
            all_lines.append(f"__PAGE_BREAK_{i + 1}__")

        full_text = sanitize_text("\n".join(all_lines))
        logger.info(f"📚 Extracted {len(full_text):,} chars — pages {start_page}–{final_page}")

    except Exception as e:
        logger.error(f"❌ PDF extraction failed: {e}")
        return

    # ── STEP 2: TABLE PASS ───────────────────────────────────────────────────
    table_buffer = TableBuffer()
    _cur_rule_id = "General Context"

    for line in all_lines:
        m = RULE_ID_RE.match(line.strip())
        if m:
            _cur_rule_id = m.group(1)
        table_buffer.feed_line(line, _cur_rule_id)

    table_buffer.end_of_document()

    tables_by_rule: dict[str, list[list[dict]]] = {}
    for rule_id, rows in table_buffer.pop_completed():
        tables_by_rule.setdefault(rule_id, []).append(rows)

    logger.info(f"📊 Tables found: {sum(len(v) for v in tables_by_rule.values())} "
                f"across {len(tables_by_rule)} rules")

    # ── STEP 3: RULE SPLITTING ───────────────────────────────────────────────
    split_content = [
        chunk.strip()
        for chunk in re.split(r'\n\s*' + RULE_ID_RE.pattern, full_text)
        if chunk.strip()
    ]

    if len(split_content) <= 1:
        logger.warning("⚠️  No rules found — check page range or rule ID pattern.")
        return

    logger.info(f"✂️  Split into ~{len(split_content) // 2} rule chunks.")

    # ── STEP 4: BUILD RULE PAYLOADS ──────────────────────────────────────────
    # Collect all rules first then batch embed — far more efficient than
    # one API call per rule like the old Gemini approach
    pending: list[dict] = []
    current_rule_id = "General Context"

    for item in split_content:
        if RULE_ID_RE.fullmatch(item):
            current_rule_id = item
            continue

        rule_text = format_rule(item)
        if len(rule_text) < 15:
            continue

        if current_rule_id in existing_rules:
            continue

        rule_domain    = detect_domain(current_rule_id)
        rule_subdomain = detect_subdomain(rule_text)
        associated     = tables_by_rule.get(current_rule_id)
        tables_json    = json.dumps(associated) if associated else None

        enriched = (
            f"[Rule {current_rule_id} | Domain: {rule_domain} | "
            f"Subdomain: {rule_subdomain}] {rule_text}"
        )

        pending.append({
            "rule_id":       current_rule_id,
            "content":       enriched,
            "domain":        rule_domain,
            "subdomain":     rule_subdomain,
            "source_domain": domain,
            "year":          RULEBOOK_YEAR,
            "tables_json":   tables_json,
            "associated":    associated,
        })

    logger.info(f"📋 {len(pending)} rules to embed and upload.")

    if not pending:
        logger.info("✅ Nothing to do — all rules already in DB.")
        return

    # ── STEP 5: BATCH EMBED & UPLOAD ─────────────────────────────────────────
    # NEW v8.0: batch embedding means ~26 API calls for 1300 rules
    # vs 1300 individual calls in the old Gemini approach
    uploaded        = 0
    failed          = 0
    tables_attached = 0

    for batch_start in range(0, len(pending), NOMIC_BATCH_SIZE):
        batch = pending[batch_start:batch_start + NOMIC_BATCH_SIZE]
        texts = [r["content"] for r in batch]

        logger.info(
            f"🔢 Embedding batch {batch_start // NOMIC_BATCH_SIZE + 1}/"
            f"{(len(pending) + NOMIC_BATCH_SIZE - 1) // NOMIC_BATCH_SIZE} "
            f"({len(batch)} rules)…"
        )

        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            logger.error(f"❌ Batch embed failed: {e} — skipping {len(batch)} rules.")
            failed += len(batch)
            continue

        # Upload each rule in the batch with its embedding
        for rule, embedding in zip(batch, embeddings):
            try:
                payload = {
                    "content":       rule["content"],
                    "rule_id":       rule["rule_id"],
                    "domain":        rule["domain"],
                    "subdomain":     rule["subdomain"],
                    "source_domain": rule["source_domain"],
                    "year":          rule["year"],
                    "embedding":     embedding,
                }
                if rule["tables_json"]:
                    payload["tables_json"] = rule["tables_json"]

                supabase.table("rulebook_chunks").insert(payload).execute()
                uploaded += 1

                if rule["associated"]:
                    tables_attached += len(rule["associated"])

                logger.info(
                    f"✅  Rule {rule['rule_id']} "
                    f"[{rule['domain']} / {rule['subdomain']}]"
                    + (f" + {len(rule['associated'])} table(s)" if rule["associated"] else "")
                )

            except Exception as e:
                logger.error(f"❌ Upload failed for {rule['rule_id']}: {str(e)[:100]}")
                failed += 1

        # Small pause between batches — polite to the API
        # No hard sleep needed since Nomic free tier has no strict RPM cap
        time.sleep(0.5)

    skipped = len(existing_rules)

    logger.info(
        f"\n🏆 INGESTION COMPLETE — [{domain}]\n"
        f"   🧠  Model     : {NOMIC_MODEL} ({NOMIC_DIMS} dims)\n"
        f"   ✅  Uploaded  : {uploaded} rules\n"
        f"   ❌  Failed    : {failed} rules\n"
        f"   ⏭️  Skipped   : {skipped} rules (already in DB)\n"
        f"   📊  Tables    : {tables_attached} attached\n"
        f"   📦  API calls : ~{(uploaded + failed + NOMIC_BATCH_SIZE - 1) // NOMIC_BATCH_SIZE} "
        f"(vs {uploaded + failed} with old Gemini approach)\n"
    )

# ==========================================
# 8. MANUAL CONTROL DECK
# ==========================================
if __name__ == "__main__":
    TARGET_DOMAIN    = "Formula Bharat 2027 Full"
    START_PAGE       = 1
    END_PAGE         = None    # None = entire document
    WIPE_SLATE_CLEAN = True    # True = wipe Gemini vectors and reingest with Nomic

    logger.info("🚀 INDRA INGESTION ENGINE v8.0 — NOMIC REINGESTION SEQUENCE…")
    logger.info("⚠️  WIPE_SLATE_CLEAN=True — existing Gemini vectors will be deleted")
    logger.info("   This is required because Gemini and Nomic vectors are incompatible")

    ingest_domain(
        file_path     = ROOT_DIR / "FB2027_Rules.pdf",
        domain        = TARGET_DOMAIN,
        start_page    = START_PAGE,
        end_page      = END_PAGE,
        force_refresh = WIPE_SLATE_CLEAN,
    )