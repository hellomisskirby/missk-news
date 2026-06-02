"""
Miss K NewsRoom — Daily News Generator
scripts/generate_news.py

Calls the Anthropic API (claude-haiku-4-5-20251001) with web_search enabled
to find and rewrite 4 real news articles (2 EN + 2 ZH) into student-friendly
summaries, then saves the result as news-data/today.json and archives it.

Required environment variable:
  ANTHROPIC_API_KEY — set as a GitHub Actions secret

Optional environment variable:
  DATE_OVERRIDE — YYYY-MM-DD string; defaults to today (HKT)
"""

import anthropic
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL          = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 8192
MAX_RETRIES    = 3
OUTPUT_FILE    = Path("news-data/today.json")
ARCHIVE_DIR    = Path("news-data/archive")
HKT            = timezone(timedelta(hours=8))
# ─────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are an educational news editor for Miss K NewsRoom, a Hong Kong-based interview preparation platform for primary and secondary school students (ages 6–18).

Your job is to find 4 real news articles published TODAY and rewrite them for student reading, following strict guidelines.

══════════════════════════════════════════
ARTICLE SELECTION
══════════════════════════════════════════
Select exactly 4 articles:
- 2 in English (from reputable sources such as BBC News, Reuters, The Guardian, South China Morning Post English, AP News)
- 2 in Chinese Traditional (from reputable sources such as 明報, 星島日報, 香港01, 東方日報, 文匯報)

Choose articles that are:
✓ Published TODAY or within the last 48 hours
✓ Suitable for students aged 6–18 (no graphic violence, no adult content)
✓ Relevant to Hong Kong, Asia, science, environment, technology, society, education, or current affairs
✓ Interesting enough to spark discussion and critical thinking
✗ Avoid: sports match results, celebrity gossip, stock market numbers, election vote counts

══════════════════════════════════════════
REWRITING RULES — STRICTLY FOLLOW
══════════════════════════════════════════
For each article:

1. DO NOT copy any sentence verbatim from the original source.
2. REWRITE the content entirely in your own words — change sentence structure, vocabulary and order.
3. Simplify language to suit the reading level of PRIMARY school students (ages 8–12) while keeping the core facts accurate.
4. Length: 80–120 words per summary (English) or 100–140 characters per summary (Chinese).
5. Tone: clear, neutral, educational. No sensationalism.
6. The rewritten summary must convey the same key facts as the original but read as an original educational text.
7. Always credit the original source in the adaptedFrom field.

══════════════════════════════════════════
THINKING QUESTIONS — 3 LEVELS
══════════════════════════════════════════
For each article, write 2 thinking questions at each of 3 levels:

Lv1 (P2–P3, ages 7–9): Simple recall and basic understanding. One sentence. For English articles, add a Chinese translation of the question in brackets.
Lv2 (P4–P6, ages 9–12): Why/how questions requiring some reasoning. For English articles, add Chinese translation.
Lv3 (S1–S3, ages 12–15): Open-ended critical thinking. Multiple perspectives. Controversial or debatable. For English articles, add Chinese translation.

Questions must be original — do not copy question phrasings from the article.

══════════════════════════════════════════
OUTPUT FORMAT — JSON ONLY
══════════════════════════════════════════
Return ONLY a valid JSON object. No markdown, no explanation, no preamble, no code fences.

{
  "articles": [
    {
      "id": "YYYY-MM-DD-en-1",
      "lang": "en",
      "source": "BBC News",
      "sourceUrl": "https://www.bbc.com/news/[article-path]",
      "flag": "🇬🇧",
      "date": "YYYY-MM-DD",
      "adaptedFrom": "BBC News",
      "title": "[Rewritten headline — clear and student-friendly]",
      "summary": "[Rewritten article body — 80–120 words, original language, not copied from source]",
      "questions": {
        "lv1": ["Question 1 (Chinese translation)", "Question 2 (Chinese translation)"],
        "lv2": ["Question 1 (Chinese translation)", "Question 2 (Chinese translation)"],
        "lv3": ["Question 1 (Chinese translation)", "Question 2 (Chinese translation)"]
      }
    },
    {
      "id": "YYYY-MM-DD-zh-1",
      "lang": "zh",
      "source": "明報",
      "sourceUrl": "https://www.mingpao.com/[article-path]",
      "flag": "🇭🇰",
      "date": "YYYY-MM-DD",
      "adaptedFrom": "明報",
      "title": "[改寫後的標題——清晰易懂]",
      "summary": "[改寫後的文章內容——100至140字，用自己文字表達，不可直接複製原文]",
      "questions": {
        "lv1": ["問題一", "問題二"],
        "lv2": ["問題一", "問題二"],
        "lv3": ["問題一", "問題二"]
      }
    },
    {
      "id": "YYYY-MM-DD-en-2",
      "lang": "en",
      "source": "Reuters",
      "sourceUrl": "https://www.reuters.com/[article-path]",
      "flag": "🇬🇧",
      "date": "YYYY-MM-DD",
      "adaptedFrom": "Reuters",
      "title": "[Rewritten headline]",
      "summary": "[Rewritten summary]",
      "questions": {
        "lv1": ["...", "..."],
        "lv2": ["...", "..."],
        "lv3": ["...", "..."]
      }
    },
    {
      "id": "YYYY-MM-DD-zh-2",
      "lang": "zh",
      "source": "星島日報",
      "sourceUrl": "https://www.singtao.com/[article-path]",
      "flag": "🇭🇰",
      "date": "YYYY-MM-DD",
      "adaptedFrom": "星島日報",
      "title": "[改寫後的標題]",
      "summary": "[改寫後的文章內容]",
      "questions": {
        "lv1": ["...", "..."],
        "lv2": ["...", "..."],
        "lv3": ["...", "..."]
      }
    }
  ]
}

══════════════════════════════════════════
QUALITY CHECKLIST — before outputting
══════════════════════════════════════════
□ All 4 articles selected from today's or last 48 hours' news
□ 2 English + 2 Chinese
□ No sentence copied verbatim from any source
□ Each summary 80–120 words (EN) or 100–140 chars (ZH)
□ adaptedFrom field present on every article
□ sourceUrl is the actual article URL, not just the homepage
□ All 6 thinking questions present per article (2 per level × 3 levels)
□ Lv1 and Lv2 English questions include Chinese translations in brackets
□ Output is valid JSON only — no extra text, no markdown code fences
"""


def get_today_hkt(override: str = "") -> str:
    """Return today's date in YYYY-MM-DD format (HKT), with optional override."""
    if override:
        return override.strip()
    return datetime.now(HKT).strftime("%Y-%m-%d")


def build_user_message(today: str) -> str:
    return (
        f"Today's date is {today} (Hong Kong Time, HKT).\n\n"
        "Please search for 4 real news articles published today or in the last 48 hours "
        "(2 in English, 2 in Traditional Chinese), then rewrite and format them exactly "
        "as specified in your instructions.\n\n"
        "Search for today's top news from BBC, Reuters, SCMP, 明報, and 星島日報. "
        "Pick stories about Hong Kong, Asia, science, environment, technology, society, or education.\n\n"
        "Return ONLY the JSON object. No markdown, no explanation."
    )


def extract_json(raw: str) -> dict:
    """
    Extract and parse JSON from the model response.
    Handles cases where the model wraps output in markdown code fences.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError(f"Response was empty after stripping code fences. Raw: {raw[:300]!r}")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def validate_articles(data: dict, today: str) -> list:
    articles = data.get("articles", [])
    if len(articles) != 4:
        raise ValueError(f"Expected 4 articles, got {len(articles)}")

    required_fields = {"id", "lang", "source", "sourceUrl", "flag", "date",
                       "adaptedFrom", "title", "summary", "questions"}
    for i, art in enumerate(articles):
        missing = required_fields - set(art.keys())
        if missing:
            raise ValueError(f"Article {i+1} missing fields: {missing}")
        qs = art.get("questions", {})
        for lvl in ("lv1", "lv2", "lv3"):
            if lvl not in qs or len(qs[lvl]) < 2:
                raise ValueError(f"Article {i+1} missing questions for {lvl}")

    langs = [a["lang"] for a in articles]
    if langs.count("en") < 2 or langs.count("zh") < 2:
        raise ValueError(f"Expected 2 EN + 2 ZH articles, got: {langs}")

    return articles


def save_output(data: dict, today: str) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved: {OUTPUT_FILE}")

    archive_path = ARCHIVE_DIR / f"{today}.json"
    shutil.copy(OUTPUT_FILE, archive_path)
    print(f"✓ Archived: {archive_path}")


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

    today = get_today_hkt(os.environ.get("DATE_OVERRIDE", ""))
    print(f"Generating news for: {today}")

    client = anthropic.Anthropic(api_key=api_key)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\nAttempt {attempt}/{MAX_RETRIES}: Calling Anthropic API with web search...")
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": build_user_message(today)}]
            )

            print(f"  stop_reason: {response.stop_reason}")
            print(f"  content blocks: {[b.type for b in response.content]}")

            raw_text = ""
            for block in response.content:
                if block.type == "text":
                    raw_text += block.text

            if not raw_text.strip():
                raise ValueError(
                    f"API returned no text content. "
                    f"stop_reason={response.stop_reason}, "
                    f"blocks={[b.type for b in response.content]}"
                )

            print("  Parsing response...")
            data = extract_json(raw_text)

            print("  Validating articles...")
            articles = validate_articles(data, today)
            print(f"✓ {len(articles)} articles validated")
            for a in articles:
                print(f"  [{a['lang'].upper()}] {a['source']} — {a['title'][:60]}...")

            output = {
                "generated": datetime.now(HKT).isoformat(),
                "date": today,
                "count": len(articles),
                "articles": articles
            }

            save_output(output, today)
            print("✓ Done!")
            return

        except Exception as e:
            last_error = e
            print(f"  ⚠ Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                wait = 10 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}") from last_error


if __name__ == "__main__":
    main()
