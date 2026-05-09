# ==========================================
# HEXAWATTS SORA - ULTIMATE INGESTION ENGINE (v7.0)
# ==========================================
# Changes from v6.5:
#   - Fixed multi-line cell continuation bug (i < len(header) + .get() guard)
#   - sleep(4.5) now ONLY fires after a real API call, not on skips
#   - Retry loop: quota cooldown no longer eats the retry budget
#   - Table buffer: page-break markers no longer interrupt active tables
#   - Rule ID regex unified across all three places it appears
#   - Minor: year hardened to constant, log messages cleaned up
# ==========================================

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
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

_required_keys = ["GEMINI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
if not all(os.getenv(k) for k in _required_keys):
    logger.error("❌ FATAL: Missing one or more API keys in .env — check GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY")
    exit(1)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

RULEBOOK_YEAR   = "2027"
FREE_TIER_SLEEP = 4.5   # seconds — keeps RPM safely under Gemini free-tier limit (13 RPM)

# ==========================================
# 2. SHARED REGEX (single source of truth)
# ==========================================
# Matches rule IDs like: T3.2 / EV4.1.3 / A1 / IN2.3
RULE_ID_RE = re.compile(r'([A-Z]{1,3}\d*\.\d+(?:\.\d+)*)')

# ==========================================
# 3. DOMAIN & SUBDOMAIN TAXONOMY
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

# Order matters: first match wins.
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
    """Map rule ID prefix to top-level domain (e.g. 'EV4.1' → 'Electric_Vehicle')."""
    m = re.match(r'^([A-Z]{1,3})', rule_id)
    return DOMAIN_MAP.get(m.group(1), "Unknown") if m else "Unknown"

def detect_subdomain(text: str) -> str:
    """Keyword-scan rule text for the best-matching subdomain."""
    lower = text.lower()
    for keywords, subdomain in SUBDOMAIN_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return subdomain
    return "General"

# ==========================================
# 4. TEXT UTILITIES
# ==========================================

def sanitize_text(text: str) -> str:
    """Strip null bytes; collapse horizontal whitespace; preserve newlines for regex."""
    text = text.replace('\x00', '')
    return re.sub(r'[ \t]+', ' ', text).strip()

def format_rule(text: str, max_chars: int = 4000) -> str:
    """Collapse whitespace; hard-truncate overly long rules."""
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= max_chars else cleaned[:max_chars] + " [TRUNCATED]"

# ==========================================
# 5. TABLE DETECTION & PARSING
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
    """
    Convert raw table lines into a list-of-dicts (JSON-ready).
    Handles multi-line cell continuation: if a row has fewer columns than
    the header, its content is appended to the previous row's cells.
    """
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

        # ✅ FIX: continuation row → append to last row's cells
        if len(cols) < len(header) and rows:
            for i, col_text in enumerate(cols):
                if i < len(header) and col_text:          # guard: stay within header width
                    key = header[i]
                    rows[-1][key] = f"{rows[-1].get(key, '')} {col_text}".strip()
            continue

        # Normal row: pad or trim to header width
        cols = (cols + [''] * len(header))[:len(header)]
        rows.append(dict(zip(header, cols)))

    return rows


class TableBuffer:
    """
    Two-pass cross-page table stitcher.
    Feed it every line of the document; it accumulates table lines across
    page boundaries and flushes only when a non-table line interrupts.
    Page-break markers are transparent — they never close an active table.
    """

    def __init__(self):
        self._lines: list[str] = []
        self._active           = False
        self._rule_id_at_start = "General Context"
        self._completed: list[tuple[str, list[dict]]] = []

    def feed_line(self, line: str, current_rule_id: str):
        # Page-break markers must NEVER interrupt an active table
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
# 6. CORE INGESTION FUNCTION
# ==========================================

def ingest_domain(
    file_path:     Path,
    domain:        str,
    start_page:    int,
    end_page:      Optional[int] = None,
    force_refresh: bool = False,
):
    logger.info(f"🏁 SYSTEM ONLINE — targeting [{domain}]")

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

    # ── STEP 2: TABLE PASS (zero API cost) ──────────────────────────────────
    table_buffer    = TableBuffer()
    _cur_rule_id    = "General Context"

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

    # ── STEP 4: EMBED & UPLOAD ───────────────────────────────────────────────
    current_rule_id = "General Context"
    uploaded        = 0
    skipped         = 0
    tables_attached = 0

    for item in split_content:

        # Rule ID header token — just update state, no upload
        if RULE_ID_RE.fullmatch(item):
            current_rule_id = item
            continue

        rule_text = format_rule(item)
        if len(rule_text) < 15:
            continue

        # Deduplication
        if current_rule_id in existing_rules:
            skipped += 1
            continue

        # Tagging
        rule_domain    = detect_domain(current_rule_id)
        rule_subdomain = detect_subdomain(rule_text)

        # Tables
        associated = tables_by_rule.get(current_rule_id)
        tables_json = json.dumps(associated) if associated else None
        if associated:
            tables_attached += len(associated)

        enriched = (
            f"[Rule {current_rule_id} | Domain: {rule_domain} | "
            f"Subdomain: {rule_subdomain}] {rule_text}"
        )

        # ── RETRY LOOP ───────────────────────────────────────────────────────
        max_retries    = 3
        api_call_made  = False

        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=enriched,
                    task_type="retrieval_document",
                    output_dimensionality=768,
                )
                api_call_made = True

                payload: dict = {
                    "content":       enriched,
                    "rule_id":       current_rule_id,
                    "domain":        rule_domain,
                    "subdomain":     rule_subdomain,
                    "source_domain": domain,
                    "year":          RULEBOOK_YEAR,
                    "embedding":     result['embedding'],
                }
                if tables_json:
                    payload["tables_json"] = tables_json   # JSONB column required

                supabase.table("rulebook_chunks").insert(payload).execute()
                uploaded += 1
                logger.info(
                    f"✅  Rule {current_rule_id} "
                    f"[{rule_domain} / {rule_subdomain}]"
                    + (f" + {len(associated)} table(s)" if associated else "")
                )
                break  # success — exit retry loop

            except Exception as e:
                err = str(e)
                logger.warning(
                    f"⚠️  Attempt {attempt + 1}/{max_retries} failed "
                    f"for {current_rule_id}: {err[:100]}"
                )

                if "429" in err or "quota" in err.lower():
                    # Quota hit: cool down but DON'T burn a retry slot
                    logger.info("⏳ Rate limit hit — cooling down 60 s…")
                    time.sleep(60)
                    # Reset attempt counter so we get a full 3 retries after cooldown
                    attempt = -1   # loop will increment to 0
                elif attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 2)   # exponential back-off: 2 s, 4 s
                else:
                    logger.error(f"❌ Permanent failure on rule {current_rule_id} — skipping.")

        # ── RATE LIMIT GUARD ─────────────────────────────────────────────────
        # Only sleep when we actually hit the Gemini API (skipped rules cost nothing)
        if api_call_made:
            time.sleep(FREE_TIER_SLEEP)

    logger.info(
        f"\n🏆 INGESTION COMPLETE — [{domain}]\n"
        f"   ✅  Uploaded : {uploaded} rules\n"
        f"   ⏭️  Skipped  : {skipped} rules (already in DB)\n"
        f"   📊  Tables   : {tables_attached} attached\n"
    )

# ==========================================
# 7. MANUAL CONTROL DECK
# ==========================================
if __name__ == "__main__":
    TARGET_DOMAIN    = "Formula Bharat 2027 Full"
    START_PAGE       = 1
    END_PAGE         = None    # None = entire document
    WIPE_SLATE_CLEAN = False  # True = delete all existing records first

    logger.info("🚀 INITIATING FULL RULEBOOK LAUNCH SEQUENCE…")

    ingest_domain(
        file_path     = ROOT_DIR / "FB2027_Rules.pdf",
        domain        = TARGET_DOMAIN,
        start_page    = START_PAGE,
        end_page      = END_PAGE,
        force_refresh = WIPE_SLATE_CLEAN,
    )