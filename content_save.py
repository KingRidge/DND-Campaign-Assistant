import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def save_narrative_content(
    content: str, category: str, name: str, data_dir: Path
) -> str:
    """Save narrative content to a markdown file in the appropriate directory."""
    # Create category directory (npcs, locations, scenes, etc.)
    category_dir = data_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Create a filename from the name
    filename = f"{datetime.now().strftime('%Y%m%d')}_{slugify(name)}.md"
    filepath = category_dir / filename

    # Add creation date and metadata
    dated_content = f"""---
created: {datetime.utcnow().isoformat()}Z
category: {category}
name: {name}
---

{content}"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(dated_content)

    return str(filepath)


def slugify(text: str) -> str:
    """Convert text to a URL and filename-friendly format."""
    import re

    # Convert to lowercase and replace spaces with underscores
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text)
    return text or "untitled"
