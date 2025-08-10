"""
Narrative content generator for D&D campaign elements.
Focuses on creative, narrative descriptions rather than structured data.
"""

import os
import re
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from dnd5e_data import (
    RACES,
    CLASSES,
    OCCUPATIONS,
    FACTIONS,
    PERSONALITY_TRAITS,
    HOOKS,
    RELATIONSHIPS,
)


def save_narrative_content(
    content: str, category: str, default_name: str, data_dir: Path
) -> Path:
    """Save generated narrative content to a markdown file.

    Attempts to extract a title from the content's first header,
    falling back to the default name if none is found.
    """
    # Try to extract title from first markdown header
    title_match = re.search(r"^#\s+(.+)$", content, re.M)
    if title_match:
        name = title_match.group(1).strip()
    else:
        name = default_name

    # Convert title to filename-safe format
    filename = slugify(name)
    if not filename:
        filename = slugify(default_name)

    # Add timestamp to ensure uniqueness
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Create category directory
    category_dir = data_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Create final path with category, name and timestamp
    filepath = category_dir / f"{filename}_{timestamp}.md"

    # Add metadata and save content
    dated_content = f"""---
created: {datetime.utcnow().isoformat()}Z
category: {category}
name: {name}
---

{content}"""

    with filepath.open("w", encoding="utf-8") as f:
        f.write(dated_content)

    return filepath


def generate_npc_description(
    hook: str,
    role: str,
    faction: str,
    relationship_to_party: str,
    secret: Optional[str],
    model: Optional[str],
) -> str:
    """Generate a narrative NPC description."""
    # Add a nonce to prevent repetitive responses
    nonce = datetime.utcnow().strftime("%H%M%S")

    system_prompt = """You are an expert D&D 5th Edition character creator who writes vivid, narrative descriptions.
IMPORTANT RESPONSE FORMAT:
1. DO NOT include any meta-commentary, <think> tags, or explanations
2. DO NOT explain your thought process
3. ALWAYS start with a Level 1 markdown header containing a UNIQUE, ORIGINAL character name
4. NEVER reuse character names from previous responses
5. Create culturally appropriate names based on the character's race and background
6. Output ONLY the character description in markdown format
7. Make sure all game mechanics are valid D&D 5e content
8. Focus on making the NPC unique and memorable"""

    prompt = f"""Create a D&D character description in markdown format. YOU MUST START WITH A LEVEL 1 HEADER CONTAINING THE CHARACTER NAME, like this:
# Aldrich Blackwood

CONTEXT:
Role Concept: {role}
Faction Affiliation: {faction}
Party Dynamic: {relationship_to_party}
Known Secret: {secret or 'Unknown'}
Story Hook: {hook}
Nonce: {nonce}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
# [Character Name]

## Game Information
- Race: [PHB Race]
- Class: [Class and Level]
- Alignment: [Alignment]
- Background: [Background]

## Role & Status
- Role: [Role in society]
- Faction: [Faction affiliation]
- Status: [Current standing]

## Physical Description
[Vivid description of appearance]

## Abilities & Skills
- Key Skills: [Notable proficiencies]
- Special Abilities: [Class/racial features]
- Combat Style: [If applicable]

## Personality
[Personality traits and behaviors]

## Voice & Mannerisms
[Description of speech and gestures]

> "[A memorable quote that shows their personality]"

## Relationships
[Connections to others and the party]

## Secrets & Motivations
[Hidden aspects and driving forces]

## Story Hooks
[Adventure seeds and plot hooks]

REQUIREMENTS:
1. MUST start with a Level 1 header (# Name)
2. Use valid D&D 5e content only
3. Make them complex and memorable
4. Include specific, vivid details
5. Follow markdown formatting exactly"""

    try:
        from .llm_utils import llm_chat_ollama

        content = llm_chat_ollama(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.8,
            max_tokens=4096,  # Increased token limit to ensure complete output
        )

        if not content or len(content) < 100:
            return f"Error: Generated content too short or empty"

        # Verify we have a name header
        if not re.search(r"^#\s+\w+", content, re.M):
            return f"Error: Missing name header in generated content"

        return content

    except Exception as e:
        return f"Error generating NPC: {str(e)}"


def generate_location_description(
    name: str, tags: list[str], model: Optional[str]
) -> str:
    """Generate a narrative location description."""
    # Add a nonce to prevent repetitive responses
    nonce = datetime.utcnow().strftime("%H%M%S")

    system_prompt = """You are a master D&D location creator who writes rich, atmospheric descriptions.
Focus on creating unique and memorable places that feel alive and dynamic.
Your descriptions should paint vivid pictures and suggest countless stories.
Include both obvious features and subtle details that reward careful exploration."""

    prompt = f"""Create an evocative and intriguing location description. Start with a unique name as the main header (# Name).

CONTEXT:
Suggested Name: {name}
Thematic Elements: {', '.join(tags) if tags else 'mysterious, atmospheric'}
Nonce: {nonce}

REQUIREMENTS:
1. Start with a distinctive name as the main header
2. Include sections for:
   - Overview & First Impressions
   - Atmosphere & Environment
   - Notable Features
   - Hidden Secrets
   - Current Inhabitants
   - Historical Significance
   - Rumors & Mysteries
   - Potential Encounters
3. Make the location:
   - Rich in sensory details
   - Dynamic and changing
   - Full of interaction points
   - Layered with history
   
Format your response in clean markdown with clear headers and sections."""

    try:
        from .llm_utils import llm_chat_ollama

        content = llm_chat_ollama(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.8,
            max_tokens=4096,  # Increased token limit to ensure complete output
        )

        if not content or len(content) < 100:
            return f"Error: Generated content too short or empty"

        # Verify we have a name header
        if not re.search(r"^#\s+\w+", content, re.M):
            return f"Error: Missing name header in generated content"

        return content

    except Exception as e:
        return f"Error generating location: {str(e)}"


def generate_scene_description(prompt: str, model: Optional[str]) -> str:
    """Generate a narrative scene description."""
    # Add a nonce to prevent repetitive responses
    nonce = datetime.utcnow().strftime("%H%M%S")

    system_prompt = """You are a master D&D scene creator who writes vivid, dynamic descriptions.
Focus on creating engaging scenes that draw players in and offer multiple interaction points.
Your descriptions should engage all the senses and suggest immediate action possibilities.
Include both atmosphere and practical details DMs can use."""

    enhanced_prompt = f"""Create a rich and dynamic scene description. Start with an evocative title as the main header (# Title).

SCENE PROMPT: {prompt}
Nonce: {nonce}

REQUIREMENTS:
1. Start with an evocative title as the main header
2. Include sections for:
   - Initial Impression
   - Atmosphere & Mood
   - Key Features & Elements
   - Dynamic Aspects
   - Sensory Details
   - Immediate Options
   - Hidden Details
3. Make the scene:
   - Immediately engaging
   - Rich in sensory detail
   - Dynamic and interactive
   - Full of possibilities
   
Format your response in clean markdown with clear headers and sections."""

    try:
        from .llm_utils import llm_chat_ollama

        content = llm_chat_ollama(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_prompt},
            ],
            model=model,
            temperature=0.8,
            max_tokens=4096,  # Increased token limit to ensure complete output
        )

        if not content or len(content) < 100:
            return f"Error: Generated content too short or empty"

        # Verify we have a title header
        if not re.search(r"^#\s+\w+", content, re.M):
            return f"Error: Missing title header in generated content"

        return content

    except Exception as e:
        return f"Error generating scene: {str(e)}"


def generate_random_npc(model: Optional[str] = None) -> str:
    """Generate a random NPC using D&D 5e elements."""
    # Generate timestamp for uniqueness
    timestamp = datetime.utcnow().strftime("%H%M%S")

    # Generate random NPC elements
    race = random.choice(RACES)
    char_class = random.choice(CLASSES)
    level = random.randint(1, 12)  # Most NPCs are lower level
    occupation = random.choice(OCCUPATIONS)
    faction = random.choice(FACTIONS)
    traits = random.sample(PERSONALITY_TRAITS, k=3)
    hook = random.choice(HOOKS)
    relationship = random.choice(RELATIONSHIPS)

    # Generate unique name suggestion based on race and role
    race_based_name = f"{race}-{timestamp}"  # This will be replaced by LLM but helps ensure uniqueness

    # Generate an appropriate role combining class and occupation
    if random.random() < 0.7:  # 70% chance to be primarily their occupation
        role = f"{occupation} (former {char_class}) - ID:{timestamp}"  # Added timestamp to role
    else:
        role = f"Level {level} {char_class} serving as {occupation} - ID:{timestamp}"  # Added timestamp to role

    # Add racial element to hook sometimes
    if random.random() < 0.3:  # 30% chance
        hook = f"{hook} (complicated by their {race.lower()} heritage)"

    # Generate the NPC
    return generate_npc_description(
        hook=hook,
        role=role,
        faction=faction,
        relationship_to_party=relationship,
        secret=f"Their {', '.join(traits)} nature leads them to {random.choice(HOOKS).lower()}",
        model=model,
    )


def slugify(text: str) -> str:
    """Convert text to a URL and filename-friendly format."""
    import re

    # Convert to lowercase and replace spaces with underscores
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text)
    return text
