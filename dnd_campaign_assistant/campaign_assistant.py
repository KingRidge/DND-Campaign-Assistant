#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Campaign Assistant for D&D 5e.py

A local-only D&D campaign assistant with blank templates to customize.
Uses Ollama for local LLM-powered content generation.

Run:
  python campaign_assistant.py init
  python campaign_assistant.py add-note "Session 1" data/notes/s001.md
  python campaign_assistant.py generate --kind scene --prompt "Write a tense scene in the [Your Location] hall"
  # Ensure Ollama is running: `ollama serve` and pull a model: `ollama pull qwen3:8b` (or your choice)
"""

import argparse, json, os, re, sys, time, math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import re
import json

# -------------------------------
# Config
# -------------------------------
DEFAULT_DATA_DIR = Path("./data")
CANON_FILE = "canon.json"
INDEX_FILE = "embeddings.json"
NOTES_DIR = "notes"
STYLES_FILE = "styles.md"
SYSTEM_PROMPT_FILE = "system_prompt.txt"

# Ollama settings (local only)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("MODEL", "qwen3:8b")  # Default model, can be overridden
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_KEY = os.getenv("OLLAMA_KEY", "")  # Optional API key for Ollama


# -------------------------------
# Utilities
# -------------------------------
def ensure_dirs(data_dir: Path) -> None:
    (data_dir / NOTES_DIR).mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def write_text(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def load_json(p: Path, default: Any) -> Any:
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return s or f"file_{int(time.time())}"


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return [t for t in re.findall(r"[a-z0-9']+", text) if t and not t.isdigit()]


# -------------------------------
# Canon Store (BLANK by default)
# -------------------------------
def default_canon() -> Dict[str, Any]:
    return {
        "players": [],  # DROP YOUR PCs HERE
        "npcs": [],  # DROP YOUR NPCS HERE
        "locations": [],  # DROP YOUR LOCATIONS HERE
        "items": [],  # DROP YOUR MAGIC ITEMS HERE
        "quests": [],  # DROP YOUR QUESTS HERE
        "timeline": [],  # DROP DATED EVENTS HERE
        "rules": {"house_rules": [], "sources": []},  # DROP HOUSE RULES/SOURCES HERE
        "_meta": {"created": now_iso(), "updated": now_iso(), "version": 1},
    }


class CanonStore:
    def __init__(self, path: Path):
        self.path = path
        self.data = load_json(path, default_canon())

    def save(self):
        self.data["_meta"]["updated"] = now_iso()
        save_json(self.path, self.data)

    def upsert(self, path_expr: str, value: Any) -> None:
        def segparse(seg):
            m = re.match(r"^([a-zA-Z0-9_]+)(\[(.*)\])?$", seg)
            if not m:
                return seg, None
            return m.group(1), m.group(3)

        target = self.data
        parts = path_expr.split(".")
        for i, seg in enumerate(parts):
            key, filt = segparse(seg)
            last = i == len(parts) - 1
            if key not in target:
                target[key] = [] if filt is not None else {}
            node = target[key]
            if filt is None:
                if last:
                    target[key] = value
                else:
                    if not isinstance(target[key], dict):
                        target[key] = {}
                    target = target[key]
            else:
                if not isinstance(node, list):
                    target[key] = []
                arr = target[key]
                if filt.strip() == "+":
                    if last:
                        arr.append(value)
                    else:
                        arr.append({})
                        target = arr[-1]
                    continue
                try:
                    crit = json.loads(filt)
                    assert isinstance(crit, dict)
                except Exception:
                    raise ValueError(f"Invalid filter in path: {seg}")
                idx = None
                for j, el in enumerate(arr):
                    if isinstance(el, dict) and all(
                        el.get(k) == v for k, v in crit.items()
                    ):
                        idx = j
                        break
                if idx is None:
                    arr.append(crit.copy())
                    idx = len(arr) - 1
                if last:
                    if isinstance(arr[idx], dict) and isinstance(value, dict):
                        arr[idx] = {**arr[idx], **value}
                    else:
                        arr[idx] = value
                else:
                    if not isinstance(arr[idx], dict):
                        arr[idx] = {}
                    target = arr[idx]
        self.save()


# -------------------------------
# Tiny TF-IDF Notes Index
# -------------------------------
class NotesIndex:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.docs: Dict[str, Any] = {}
        self.df: Dict[str, int] = {}
        self.N: int = 0
        self.load()

    def load(self):
        if self.index_path.exists():
            raw = load_json(self.index_path, {})
            self.docs = raw.get("docs", {})
            self.df = raw.get("df", {})
            self.N = raw.get("N", 0)

    def save(self):
        save_json(self.index_path, {"docs": self.docs, "df": self.df, "N": self.N})

    def add_document(self, doc_id: str, title: str, text: str, path: str):
        tf = Counter(tokenize(text))
        seen = set(tf.keys())
        for t in seen:
            self.df[t] = self.df.get(t, 0) + 1
        self.N += 1
        self.docs[doc_id] = {
            "path": path,
            "title": title,
            "tf": tf,
            "len": sum(tf.values()),
        }
        self.save()

    def _tfidf_vec(self, tf: Counter) -> Dict[str, float]:
        total = sum(tf.values()) or 1
        vec = {}
        for t, f in tf.items():
            df = self.df.get(t, 0)
            if df == 0:
                continue
            idf = math.log((self.N + 1) / (df + 0.5)) + 1.0
            vec[t] = (f / total) * idf
        return vec

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        qtf = Counter(tokenize(query))
        qvec = self._tfidf_vec(qtf)
        scored = []
        for doc_id, meta in self.docs.items():
            dvec = self._tfidf_vec(meta["tf"])
            scored.append((self._cosine(qvec, dvec), doc_id))
        scored.sort(reverse=True)
        out = []
        for score, doc_id in scored[:k]:
            meta = self.docs[doc_id]
            snippet = make_snippet(
                Path(meta["path"]).read_text(encoding="utf-8"), query
            )
            out.append(
                {
                    "source": "notes",
                    "doc_id": doc_id,
                    "title": meta["title"],
                    "path": meta["path"],
                    "score": round(score, 4),
                    "snippet": snippet,
                }
            )
        return out


def make_snippet(text: str, query: str, width: int = 160) -> str:
    if not text:
        return ""
    for token in tokenize(query):
        m = re.search(re.escape(token), text, flags=re.IGNORECASE)
        if m:
            start = max(0, m.start() - width // 2)
            return text[start : start + width].replace("\n", " ")
    return text[:width].replace("\n", " ")


# -------------------------------
# Retrieval helpers
# -------------------------------
def search_canon(
    canon: Dict[str, Any], query: str, max_items: int = 20
) -> List[Tuple[str, Any]]:
    q = set(tokenize(query))
    hits: List[Tuple[str, Any]] = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        else:
            if q & set(tokenize(str(node))):
                hits.append((path, node))

    walk(canon)
    hits.sort(key=lambda x: (-(len(re.findall(r"\[(\d+)\]", x[0]))), len(x[0])))
    return hits[:max_items]


def rerank(canon_hits, notes_hits, k=8):
    merged = [
        {"source": "canon", "path": p, "value": v, "freshness": 1.0}
        for p, v in canon_hits
    ]
    merged += [{**h, "freshness": 0.7} for h in notes_hits]
    return merged[:k]


def default_system_prompt() -> str:
    return (
        "You are a D&D campaign assistant, guardian of Canon and generator of flavorful content.\n"
        "Rules: 1) Never contradict Canon. 2) If Canon is silent, use Notes but flag uncertainty. "
        "3) Maintain established voices. 4) Output JSON when asked; else clean prose. "
        "If user wants to change canon, require explicit 'Update Canon' confirmation.\n"
    )


def build_context_block(retrieved) -> str:
    lines = []
    for r in retrieved:
        if r.get("source") == "canon":
            lines.append(f"- [CANON] {r['path']}: {r['value']}")
        else:
            lines.append(
                f"- [NOTES s={r.get('score')}] {r.get('title')} :: {r.get('snippet')}"
            )
    return "\n".join(lines)


# -------------------------------
# Ollama LLM plumbing (local only)
# -------------------------------
def llm_chat_ollama(
    messages,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    expect_json: bool = False,
) -> str:

    model_name, server, key = model
    url = f"{server}/api/chat"
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }

    if expect_json:
        # Add JSON-specific prompt elements
        messages = [
            {
                "role": "system",
                "content": "You are a JSON generator. Your response must be a single, complete, valid JSON object. No other text.",
            },
            *messages,
        ]
        options["format"] = "json"  # Tell Ollama we want JSON
        options["stop"] = ["\n\n", "</s>", "```"]  # Stop markers

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    print(f"Sending request to Ollama...", file=sys.__stderr__)
    print(f"Model: {payload['model']}", file=sys.__stderr__)
    print(f"Temperature: {temperature}", file=sys.__stderr__)
    print(f"Expecting JSON: {expect_json}", file=sys.__stderr__)

    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT, headers=headers)
    r.raise_for_status()
    data = r.json()

    content = data.get("message", {}).get("content", "").strip()

    if not content:
        print("Received empty response from Ollama", file=sys.__stderr__)
        raise ValueError("Empty response from LLM")

    print(f"Raw response from Ollama:", file=sys.__stderr__)
    print("-" * 40, file=sys.__stderr__)
    print(content[:500], file=sys.__stderr__)
    print("-" * 40, file=sys.__stderr__)

    if expect_json:
        # Try to find a JSON object in the response
        try:
            # First, try to parse the whole response as JSON
            json.loads(content)
            return content
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the response
            try:
                # Look for content between triple backticks
                matches = re.findall(r"```(?:json)?\s*(.+?)\s*```", content, re.DOTALL)
                if matches:
                    content = matches[0]

                # Find the first { and last }
                start = content.find("{")
                end = content.rfind("}")

                if start >= 0 and end > start:
                    content = content[start : end + 1]
                    # Validate that this is valid JSON
                    json.loads(content)  # This will raise JSONDecodeError if invalid
                    return content

                raise ValueError("No JSON object found in response")
            except Exception as e:
                print(f"Error extracting JSON: {e}", file=sys.__stderr__)
                print("Attempted to parse:", file=sys.__stderr__)
                print(content, file=sys.__stderr__)
                raise

    return content


def run_llm_with_context(
    user_request: str,
    retrieved_blocks: list,
    system_prompt: str,
    model: Optional[str] = None,
) -> str:
    ctx = build_context_block(retrieved_blocks)
    prompt = (
        f"Create a vivid and dynamic scene for a D&D campaign. Focus on making it engaging and memorable.\n\n"
        f"CONTEXT (Canon first, then Notes):\n{ctx}\n\n"
        f"SCENE REQUEST:\n{user_request}\n\n"
        f"GUIDELINES:\n"
        f"- Paint a rich picture using all five senses\n"
        f"- Create a strong atmosphere and mood\n"
        f"- Include dynamic elements that could change the scene\n"
        f"- Add environmental details that could be used in interesting ways\n"
        f"- Include subtle hooks or mysteries\n"
        f"- Consider the emotional impact on the players\n"
        f"- Add at least one unexpected or unique element\n\n"
        f"Follow the rules strictly and flag uncertainty when Canon is silent. Focus on creating an immersive and memorable experience."
    )
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = llm_chat_ollama(
            messages, model=model, temperature=0.8, max_tokens=4096
        )
        if not response.strip():
            print("Empty response from LLM", file=sys.__stderr__)
            return generate_scene_stub(user_request, retrieved_blocks)
        return response
    except Exception as e:
        print(f"LLM error: {e}", file=sys.__stderr__)
        return f"[Fallback draft due to LLM error] " + generate_scene_stub(
            user_request, retrieved_blocks
        )


# -------------------------------
# Generators (LLM + neutral stubs; NO setting-specific lore)
# -------------------------------
def generate_scene_stub(query: str, context: List[Dict[str, Any]]) -> str:
    return (
        "A quiet corridor stretches ahead, lit by unsteady lanterns. "
        "Footsteps pause just out of sight as a hushed voice weighs the next move."
    )


def generate_npc_llm(
    hook: str,
    role: str,
    faction: str,
    relationship_to_party: str,
    secret: Optional[str],
    model: Optional[str],
) -> str:
    """Generate a D&D NPC description with rich narrative detail."""
    from datetime import datetime

    system_prompt = "You are an expert D&D 5e character creator, skilled at crafting memorable NPCs."

    # Add a nonce to break repetitive responses
    nonce = datetime.now().strftime("%H%M%S")

    prompt = f"""Create a vivid and compelling NPC description. Start with a unique name as the main header (# Name).

CONTEXT:
Role Concept: {role}
Faction: {faction}
Party Dynamic: {relationship_to_party}
Known Secret: {secret or 'Unknown'}
Hook: {hook}
Nonce: {nonce}

REQUIREMENTS:
1. Start with a unique name as main header
2. Include D&D 5e specific sections:
   - Game Information (race, class, level, alignment)
   - Stats & Abilities (key skills, notable features)
   - Background & Proficiencies
3. Include narrative sections:
   - Physical Description
   - Personality & Motivations
   - Voice & Mannerisms (with quote)
   - Relationships & Connections
   - Secrets & Story Hooks
4. Make them:
   - Complex and layered
   - Mechanically valid for 5e
   - Rich in specific details
   
Format your response in clean markdown with clear headers and sections."""

    try:
        content = llm_chat_ollama(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.8,
            max_tokens=4096,
        )

        if not content or len(content) < 100:
            return f"Error: Generated content too short or empty"

        # Verify we have a name header
        if not re.search(r"^#\s+\w+", content, re.M):
            return f"Error: Missing name header in generated content"

        return content

    except Exception as e:
        return f"Error generating NPC: {str(e)}"


def generate_location_llm(
    name_hint: str, tags: List[str], model: Optional[str]
) -> Dict[str, Any]:
    from datetime import datetime

    system_prompt = (
        "You are a master worldbuilder for D&D. Create unique locations with unexpected elements "
        "and rich storytelling potential. Never repeat input text verbatim. Focus on original, "
        "surprising details that make each location memorable. Return ONLY valid JSON."
    )

    schema = {
        "name": "string (unique, evocative name)",
        "region": "string (broader geographical context)",
        "tags": ["array of defining characteristics"],
        "atmosphere": "string (overall mood and feeling)",
        "sensory_5": {
            "sight": "string (visual details)",
            "sound": "string (ambient sounds)",
            "smell": "string (distinct odors)",
            "touch": "string (textures and temperatures)",
            "taste": "string (air taste or local flavors)",
        },
        "lore": ["array of 3-4 historical or mythical elements"],
        "secrets": ["array of 2-3 hidden truths about the location"],
        "inhabitants": ["array of unique residents or creatures"],
        "points_of_interest": ["array of 3-4 notable features or areas"],
        "encounter_table": ["array of 4-5 possible encounters"],
        "travel_hooks": ["array of 2-3 reasons to visit/return"],
        "rumors": ["array of 2-3 local whispers or legends"],
        "history": ["array of 2-3 significant past events"],
    }

    # Add a nonce to break repetitive responses
    nonce = datetime.now().strftime("%H%M%S")

    prompt = (
        f"Create an evocative and mysterious location. Nonce: {nonce}\n\n"
        f"Context (DO NOT REPEAT VERBATIM):\n"
        f"- Suggested Name: {name_hint}\n"
        f"- Theme Elements: {', '.join(tags)}\n\n"
        f"Requirements:\n"
        "1. Create a rich, multi-layered environment\n"
        "2. Include unexpected features and phenomena\n"
        "3. Add multiple storytelling hooks\n"
        "4. Design dynamic elements that change over time\n"
        "5. Include mysteries and secrets to uncover\n"
        "6. Make the location feel alive and interactive\n\n"
        f"Return valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
    )

    try:
        txt = llm_chat_ollama(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.8,
            max_tokens=4096,
            expect_json=True,
        )
        try:
            loc = json.loads(txt)
        except json.JSONDecodeError as je:
            print(f"Invalid JSON from LLM: {je}", file=sys.__stderr__)
            print(f"Raw response: {txt[:500]}...", file=sys.__stderr__)
            return generate_location_stub(name_hint, tags)

        # Ensure required fields exist with non-empty values
        required_fields = ["name", "region", "atmosphere", "sensory_5", "lore"]
        if not all(loc.get(field) for field in required_fields):
            print(
                f"Missing or empty required fields in JSON response",
                file=sys.__stderr__,
            )
            return generate_location_stub(name_hint, tags)

        loc.setdefault("id", f"loc_{slugify(loc.get('name', name_hint))}")
        return loc
    except Exception as e:
        print(f"LLM error: {e}", file=sys.__stderr__)
        return generate_location_stub(name_hint, tags)


def generate_item_llm(
    name_hint: str, rarity: str, model: Optional[str]
) -> Dict[str, Any]:
    sys = default_system_prompt()
    schema = "Return ONLY JSON: id,name,rarity,attunement,origin,properties,quirk,complication,owners_timeline."
    user = f"Create a magic item named '{name_hint}' with rarity '{rarity}'.\n{schema}"
    try:
        txt = llm_chat_ollama(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            model=model,
            max_tokens=4096,
        )
        item = json.loads(txt)
        item.setdefault("id", f"item_{slugify(item.get('name', name_hint))}")
        return item
    except Exception:
        return generate_item_stub(name_hint, rarity)


# Neutral, system-agnostic stubs
def generate_npc_stub(hook, role, faction, relationship_to_party, one_secret=None):
    name = f"{role.title()} of {faction}".strip()
    return {
        "id": f"npc_{slugify(name)}",
        "name": name,
        "role": role,
        "faction": faction,
        "motivations": ["Protect their position", "Advance faction interests"],
        "secrets": [
            one_secret or "Hiding a compromise that could flip their loyalties"
        ],
        "relationships": {"party": relationship_to_party},
        "last_seen": "Unseen (newly introduced)",
        "hook": hook,
        "voice": "Distinct cadence and a repeating verbal tic",
        "voice_line": "“State your aim—clearly, if you can.”",
    }


def generate_location_stub(name_hint, tags):
    return {
        "id": f"loc_{slugify(name_hint)}",
        "name": name_hint,
        "region": "A realm where nature and magic intertwine",
        "tags": tags,
        "lore": [
            "Built upon ruins of an ancient civilization",
            "Site of a mysterious magical phenomenon",
            "Local legends speak of hidden treasures and dire warnings",
        ],
        "sensory_5": {
            "sight": "Ethereal lights dance between twisted stone archways",
            "sound": "Whispers echo from unknown sources, carrying fragments of forgotten songs",
            "smell": "The air carries a mix of exotic spices and magical resonance",
            "touch": "Surfaces pulse with subtle magical energy",
            "taste": "The air tastes of ozone and ancient mysteries",
        },
        "encounter_table": [
            "Mysterious trader with impossible wares",
            "Echo of a past event playing out",
            "Local resident with an unusual request",
            "Strange phenomenon that defies explanation",
        ],
        "travel_hooks": [
            "Rumors of valuable artifacts surfacing",
            "Strange lights seen during specific lunar phases",
            "Locals reporting unusual dreams",
        ],
        "secrets": [
            "Hidden doorways to other realms",
            "Ancient prophecy written in the architecture",
        ],
        "inhabitants": [
            "Keeper of forgotten lore",
            "Mysterious entities that appear at twilight",
        ],
        "points_of_interest": [
            "The Whispering Wall",
            "The Ethereal Garden",
            "The Chamber of Echoes",
        ],
    }


def generate_item_stub(name_hint, rarity="rare"):
    return {
        "id": f"item_{slugify(name_hint)}",
        "name": name_hint,
        "rarity": rarity,
        "attunement": True,
        "origin": "",
        "properties": [],
        "quirk": "",
        "complication": "",
        "owners_timeline": [],
    }


# -------------------------------
# Proposals & RAG
# -------------------------------
def propose_updates_from_text(text: str) -> Dict[str, Any]:
    # Generic extractor with no setting assumptions
    updates = []
    # Example: "NPC <Name> last seen: Session 3, docks"
    m = re.search(r"npc\s+(.+?)\s+last\s+seen\s*:\s*(.+)$", text, re.I)
    if m:
        updates.append(
            {
                "op": "upsert",
                "path": f'npcs[{{"name":"{m.group(1).strip()}"}}]',
                "value": {"name": m.group(1).strip(), "last_seen": m.group(2).strip()},
            }
        )
    return {"canon_suggestions": updates}


def retrieve_and_generate(
    canon: Dict[str, Any],
    index: NotesIndex,
    query: str,
    model: Optional[str],
    k: int = 6,
) -> Dict[str, Any]:
    c_hits = search_canon(canon, query, max_items=10)
    n_hits = index.search(query, k=k)
    merged = rerank(c_hits, n_hits, k=k)
    system_prompt = default_system_prompt()
    draft = run_llm_with_context(query, merged, system_prompt, model=model)
    return {
        "prompt": f"SYSTEM:\n{system_prompt}\n\nCONTEXT:\n{build_context_block(merged)}\n\nUSER:\n{query}",
        "draft": draft,
        "new_hooks": [],  # INTENTIONALLY EMPTY: ADD YOUR HOOK SUGGESTION LOGIC OR LEAVE BLANK
        "canon_suggestions": [],  # INTENTIONALLY EMPTY: ADD YOUR DIFFS OR LEAVE BLANK
        "context_used": merged,
    }


# -------------------------------
# CLI
# -------------------------------
def cmd_init(args):
    data_dir = Path(args.data_dir)
    ensure_dirs(data_dir)
    canon_path = data_dir / CANON_FILE
    if not canon_path.exists():
        save_json(canon_path, default_canon())
    if not (data_dir / INDEX_FILE).exists():
        save_json(data_dir / INDEX_FILE, {"docs": {}, "df": {}, "N": 0})
    if not (data_dir / STYLES_FILE).exists():
        write_text(
            data_dir / STYLES_FILE,
            "# Styles (fill me)\n\n- Narrator: \n- Major NPCs:\n  - <Name>: <voice cues>\n",
        )
    if not (data_dir / SYSTEM_PROMPT_FILE).exists():
        write_text(data_dir / SYSTEM_PROMPT_FILE, default_system_prompt())
    # Drop a blank session template to copy
    template = """# Session <Number> – <Title>

**Date:** <YYYY-MM-DD>  
**Party:** <PC1>, <PC2>, <PC3>  

## Summary
- <beats>

## NPC Interactions
- **<NPC Name>** (<voice/tic>): <what happened>

## Loot & Hooks
- <loot>
- <hook>

## Locations Visited
- <place>: <details>
"""
    tpath = data_dir / "session_template.md"
    if not tpath.exists():
        write_text(tpath, template)
    print(f"Initialized data dir at {data_dir}")


def cmd_add_note(args):
    data_dir = Path(args.data_dir)
    ensure_dirs(data_dir)
    idx = NotesIndex(data_dir / INDEX_FILE)
    src = Path(args.src)
    if not src.exists():
        print(f"Note file not found: {src}", file=sys.stderr)
        sys.exit(1)
    text = read_text(src)
    doc_id = f"note_{slugify(args.title)}_{int(time.time())}"
    idx.add_document(doc_id, args.title, text, str(src))
    print(f"Indexed note '{args.title}' from {src} as {doc_id}")


def cmd_search(args):
    data_dir = Path(args.data_dir)
    canon = load_json(data_dir / CANON_FILE, default_canon())
    idx = NotesIndex(data_dir / INDEX_FILE)
    c_hits = search_canon(canon, args.q, max_items=10)
    n_hits = idx.search(args.q, k=args.k)
    merged = rerank(c_hits, n_hits, k=args.k)
    print("=== RESULTS (Canon first, then Notes) ===")
    for r in merged:
        if r["source"] == "canon":
            print(f"[CANON] {r['path']}: {r['value']}")
        else:
            print(
                f"[NOTES {r['score']}] {r['title']} -> {r['path']}\n  ... {r['snippet']}"
            )


def cmd_generate(args):
    data_dir = Path(args.data_dir)
    ensure_dirs(data_dir)
    from narrative_generator import (
        generate_npc_description,
        generate_random_npc,
        generate_location_description,
        generate_scene_description,
        save_narrative_content,
    )

    model = (
        args.model or OLLAMA_MODEL,
        args.ollama_server or OLLAMA_HOST,
        args.ollama_key or OLLAMA_KEY,
    )

    try:
        if args.kind == "npc":
            if args.random:
                content = generate_random_npc(model)
                default_name = "random_npc"
            else:
                content = generate_npc_description(
                    args.hook or "",
                    args.role or "",
                    args.faction or "",
                    args.rel or "",
                    args.secret,
                    model,
                )
                default_name = args.role or "mysterious_npc"

            if content and not content.startswith("Error"):
                filepath = save_narrative_content(
                    content, "npcs", default_name, data_dir
                )
                print(f"NPC description saved to: {filepath}")
                print("\n" + content)
            else:
                print(content, file=sys.__stderr__)

        elif args.kind == "location":
            content = generate_location_description(
                args.name or "mysterious_place", args.tags or [], model
            )
            if content and not content.startswith("Error"):
                filepath = save_narrative_content(
                    content, "locations", args.name or "mysterious_place", data_dir
                )
                print(f"Location description saved to: {filepath}")
                print("\n" + content)
            else:
                print(content, file=sys.__stderr__)

        elif args.kind == "scene":
            content = generate_scene_description(
                args.prompt or "A mysterious scene unfolds...", model
            )
            if content and not content.startswith("Error"):
                scene_name = f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                filepath = save_narrative_content(
                    content, "scenes", scene_name, data_dir
                )
                print(f"Scene description saved to: {filepath}")
                print("\n" + content)
            else:
                print(content, file=sys.__stderr__)

        elif args.kind == "item":
            # TODO: Implement narrative item generation
            print(
                "Item generation not yet implemented in narrative format",
                file=sys.__stderr__,
            )

        else:
            print(
                "Unknown kind; choose one of: npc | location | scene | item",
                file=sys.__stderr__,
            )
            sys.exit(1)

    except Exception as e:
        print(f"Error generating content: {e}", file=sys.__stderr__)
        sys.exit(1)


def cmd_propose_updates(args):
    text = args.text
    suggestions = propose_updates_from_text(text)
    print(json.dumps(suggestions, indent=2, ensure_ascii=False))


def cmd_apply_update(args):
    data_dir = Path(args.data_dir)
    canon = CanonStore(data_dir / CANON_FILE)
    try:
        value = json.loads(args.value)
    except Exception:
        print("Value must be valid JSON.", file=sys.stderr)
        sys.exit(1)
    canon.upsert(args.path, value)
    print(f"Applied upsert at {args.path}")


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Local-only D&D Campaign Assistant (Blank Canon)"
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Data directory (default: ./data)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize data folder (blank canon + templates)")

    p_add = sub.add_parser("add-note", help="Add/Index a session note file")
    p_add.add_argument("title", help="Human title for the note (e.g., 'Session 1')")
    p_add.add_argument("src", help="Path to a .txt/.md file to index")

    p_search = sub.add_parser("search", help="Search canon + notes")
    p_search.add_argument("--q", required=True, help="Query string")
    p_search.add_argument("--k", type=int, default=6, help="Top-k notes to return")

    p_gen = sub.add_parser("generate", help="Generate content")
    p_gen.add_argument(
        "--kind", required=True, choices=["npc", "location", "item", "scene"]
    )
    p_gen.add_argument(
        "--random",
        action="store_true",
        help="Generate random content (currently supported for NPCs)",
    )
    # NPC
    p_gen.add_argument("--hook")
    p_gen.add_argument("--role")
    p_gen.add_argument("--faction")
    p_gen.add_argument("--rel")
    p_gen.add_argument("--secret")
    # Location/Item
    p_gen.add_argument("--name")
    p_gen.add_argument("--rarity")
    p_gen.add_argument("--tags", nargs="*", default=None)
    # Scene
    p_gen.add_argument("--prompt")
    p_gen.add_argument("--k", type=int, default=6)
    p_gen.add_argument("--model", help="Override Ollama model (default from env MODEL)")
    p_gen.add_argument(
        "--ollama-server",
        help="Override Ollama model server URL",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )
    p_gen.add_argument(
        "--ollama-key",
        help="Set the Ollama Key used for authentication (if required)",
        default=None,
    )
    p_prop = sub.add_parser(
        "propose-updates", help="Suggest structured canon updates from plain text"
    )
    p_prop.add_argument("--text", required=True)

    p_apply = sub.add_parser(
        "apply-update", help="Apply a single upsert to canon by path"
    )
    p_apply.add_argument(
        "--path", required=True, help='e.g., npcs[{"id":"npc_x"}].last_seen'
    )
    p_apply.add_argument(
        "--value",
        required=True,
        help='JSON value, e.g., ""Session 3, docks"" or {"last_seen":"..."}',
    )

    args = parser.parse_args()

    if args.cmd == "init":
        cmd_init(args)
    elif args.cmd == "add-note":
        cmd_add_note(args)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "propose-updates":
        cmd_propose_updates(args)
    elif args.cmd == "apply-update":
        cmd_apply_update(args)


if __name__ == "__main__":
    main()
