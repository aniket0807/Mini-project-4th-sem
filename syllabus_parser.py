"""
===================================================================
SYLLABUS_PARSER.PY — Gemini API PDF Syllabus Parser
===================================================================
Primary:  Gemini 2.0 Flash — structured JSON extraction from PDF text
Fallback: Regex/heuristic parser — runs when Gemini is unavailable
           (rate-limited, quota exhausted, no API key, etc.)
===================================================================
"""

import os, re, json, time, fitz
from dotenv import load_dotenv
from typing import Dict, List


# ─────────────────────────────────────────────
#  PDF TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes using PyMuPDF."""
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text.strip()


# ─────────────────────────────────────────────
#  GEMINI CLIENT
# ─────────────────────────────────────────────
def _get_gemini_client():
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key in ("your_gemini_api_key_here", ""):
        raise ValueError("GEMINI_API_KEY is not set in your .env file.")
    from google import genai
    return genai.Client(api_key=api_key)


_GEMINI_PROMPT = """
You are an expert academic curriculum analyst. Parse the syllabus and return
ONLY a valid JSON object (no markdown, no extra text) with this schema:

{
  "course_name": "<course name>",
  "topics": ["<all individual topics as a flat list>"],
  "units": [{"unit": "<unit name>", "topics": ["<topics>"]}]
}

Rules:
1. Extract EVERY topic, subtopic, and concept.
2. Exclude admin info (dates, credits, faculty names, contact hours).
3. Keep topic names concise but complete.
4. If no clear units exist, use a single unit named "General".
5. Return ONLY JSON.
"""

_GEMINI_MODELS = [
    "gemini-2.5-flash",        # Best — latest flash
    "gemini-2.0-flash",        # Stable flash
    "gemini-2.0-flash-lite",   # Lighter flash
    "gemini-1.5-flash",        # Older but very stable
]


def _try_gemini(raw_text: str) -> Dict:
    """Try Gemini API with model fallback chain. Returns result dict."""
    try:
        client = _get_gemini_client()
    except ValueError as e:
        return {"success": False, "error": str(e), "_use_fallback": True}

    prompt = f"{_GEMINI_PROMPT}\n\n--- SYLLABUS ---\n{raw_text[:12000]}\n--- END ---"

    last_error = ""
    for model_name in _GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = response.text.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                text = "\n".join(
                    l for l in text.splitlines()
                    if not l.strip().startswith("```")
                ).strip()

            parsed = json.loads(text)
            topics = parsed.get("topics", [])
            if not isinstance(topics, list) or len(topics) == 0:
                continue   # Try next model

            return {
                "success":     True,
                "course_name": parsed.get("course_name", "Unknown Course"),
                "topics":      topics,
                "units":       parsed.get("units", []),
                "source":      f"Gemini ({model_name})",
            }

        except json.JSONDecodeError:
            last_error = f"Gemini ({model_name}) returned non-JSON output."
            continue
        except Exception as e:
            err_str = str(e)
            # Retryable errors — try next model (different quotas / endpoints)
            if any(code in err_str for code in
                   ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                    "500", "INTERNAL", "overloaded")):
                last_error = f"{model_name}: temporarily unavailable, trying next model…"
                time.sleep(1)
                continue
            # Hard error (auth, network) — bail out
            last_error = f"API error on {model_name}: {e}"
            break

    # All models failed — signal to use fallback
    return {"success": False, "error": last_error, "_use_fallback": True}


# ─────────────────────────────────────────────
#  FALLBACK HEURISTIC PARSER
# ─────────────────────────────────────────────
_SKIP_PATTERNS = re.compile(
    r"""(^\s*$                      # blank line
    | ^\d+[\.\)]\s*$               # just a number
    | page\s*\d+                   # page numbers
    | \d{1,2}[\/\-]\d{1,2}[\/\-]  # dates
    | credits?|hours?|duration     # admin words
    | faculty|professor|dr\.|mr\.|ms\.
    | university|college|department
    | exam|test\s+date|assessment\s+date
    | total\s+marks|internal|external
    | prerequisite|co.requisite
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_UNIT_HEADER = re.compile(
    r"""^(unit|module|chapter|section|part|topic)\s*[\-–:]?\s*(\w.*)|
         ^(i{1,3}v?|vi{0,3}|ix|x{1,3})\s*[\.\)]\s+(\w.*)""",
    re.IGNORECASE | re.VERBOSE,
)

_TOPIC_LINE = re.compile(
    r"""^[\s\-\•\*\d\.\)]+              # leading bullet / number / dash
        ([A-Z][a-zA-Z0-9\s\(\)\/\-,]+) # topic starting with capital letter
    """,
    re.VERBOSE,
)


def _heuristic_parse(raw_text: str) -> Dict:
    """
    Pure-regex fallback that extracts topics without any AI API call.
    Returns the same schema as _try_gemini on success.
    """
    lines = raw_text.splitlines()

    # Guess course name from first non-empty, reasonably long line
    course_name = "Unknown Course"
    for line in lines[:20]:
        stripped = line.strip()
        if 5 < len(stripped) < 80 and not re.search(r"\d{4}", stripped):
            course_name = stripped
            break

    units: List[Dict] = []
    current_unit = "General"
    current_topics: List[str] = []
    all_topics: List[str] = []

    for line in lines:
        line = line.strip()
        if not line or _SKIP_PATTERNS.search(line):
            continue

        # Detect unit header
        m = _UNIT_HEADER.match(line)
        if m:
            if current_topics:
                units.append({"unit": current_unit, "topics": list(current_topics)})
                all_topics.extend(current_topics)
                current_topics = []
            # Use the header title as unit name
            title = (m.group(2) or m.group(4) or line).strip()
            current_unit = re.sub(r"\s+", " ", title)[:60]
            continue

        # Detect topic line (bullet, numbered, or plain capitalized phrase)
        m2 = _TOPIC_LINE.match(line)
        if m2:
            topic = re.sub(r"\s+", " ", m2.group(1)).strip()
            if 3 < len(topic) < 80:
                current_topics.append(topic)
        else:
            # Plain line — include if it looks like a concept (≥2 words, not all caps header)
            words = line.split()
            if (2 <= len(words) <= 10
                    and not line.isupper()
                    and line[0].isupper()
                    and len(line) < 80
                    and not re.search(r"\d{3,}", line)):
                topic = re.sub(r"\s+", " ", line).strip()
                current_topics.append(topic)

    # Flush last unit
    if current_topics:
        units.append({"unit": current_unit, "topics": list(current_topics)})
        all_topics.extend(current_topics)

    # Deduplicate while preserving order
    seen = set()
    unique_topics = []
    for t in all_topics:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique_topics.append(t)

    if not unique_topics:
        return {"success": False, "error": "Could not extract any topics from the PDF text."}

    return {
        "success":     True,
        "course_name": course_name,
        "topics":      unique_topics,
        "units":       units if units else [{"unit": "General", "topics": unique_topics}],
        "source":      "Local Text Parser (Gemini unavailable)",
    }


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────
def parse_syllabus_pdf(pdf_bytes: bytes) -> Dict:
    """
    Full pipeline:
      1. Extract PDF text (PyMuPDF)
      2. Try Gemini API (with model fallback chain)
      3. If Gemini fails → use heuristic text parser

    Returns:
      {"success": True,  "course_name": ..., "topics": [...],
       "units": [...], "source": "...", "raw_text_preview": "..."}
      {"success": False, "error": "..."}
    """
    # Step 1: Extract text
    try:
        raw_text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        return {"success": False, "error": f"Failed to read PDF: {e}"}

    if len(raw_text.strip()) < 50:
        return {"success": False,
                "error": "PDF appears to be empty or image-only (no extractable text found)."}

    raw_preview = raw_text[:500]

    # Step 2: Try Gemini
    gemini_result = _try_gemini(raw_text)

    if gemini_result.get("success"):
        gemini_result["raw_text_preview"] = raw_preview
        return gemini_result

    # Step 3: Fallback
    fallback_result = _heuristic_parse(raw_text)
    fallback_result["raw_text_preview"] = raw_preview

    # Attach Gemini failure reason as a note (non-blocking)
    if not gemini_result.get("success") and gemini_result.get("error"):
        fallback_result["gemini_note"] = gemini_result["error"]

    return fallback_result
