import streamlit as st
from pathlib import Path
from campaign_assistant import (
    NotesIndex,
    CanonStore,
    DEFAULT_DATA_DIR,
    CANON_FILE,
    INDEX_FILE,
    retrieve_and_generate,
    generate_npc_llm,
    generate_location_llm,
    generate_item_llm,
    ensure_dirs,
    load_json,
    default_canon,
    slugify,
)
from narrative_generator import (
    generate_npc_description,
    generate_location_description,
    generate_scene_description,
    generate_random_npc,
    save_narrative_content,
)
import os

# Initialize session state
if "data_dir" not in st.session_state:
    st.session_state.data_dir = Path(DEFAULT_DATA_DIR)
    ensure_dirs(st.session_state.data_dir)

# Page configuration
st.set_page_config(page_title="D&D Campaign Assistant", page_icon="🎲", layout="wide")

# Title and description
st.title("🎲 D&D Campaign Assistant")
st.markdown(
    """
This tool helps you manage and generate content for your D&D campaign using AI assistance.
Make sure you have Ollama running locally with your preferred model.
"""
)

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Model selection
    model_name = st.text_input(
        "Ollama Model",
        value=os.getenv("MODEL", "qwen3:8b"),
        help="The name of the Ollama model to use",
    )

    # Ollama server URL
    ollama_host = st.text_input(
        "Ollama Server URL",
        value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="The URL of your Ollama server",
    )

    model = (model_name, ollama_host, os.getenv("OLLAMA_KEY", ""))

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📝 Notes & Search", "👤 NPCs", "🏰 Locations", "⚔️ Scenes", "📊 Campaign Data"]
)

# Notes & Search Tab
with tab1:
    st.header("Notes & Search")

    search_query = st.text_input(
        "Search your campaign notes and canon", placeholder="Enter your search query..."
    )

    if search_query:
        # Load canon and index
        canon = load_json(st.session_state.data_dir / CANON_FILE, default_canon())
        index = NotesIndex(st.session_state.data_dir / INDEX_FILE)

        # Perform search
        results = retrieve_and_generate(canon, index, search_query, model, k=6)

        # Display results
        for item in results["context_used"]:
            with st.expander(f"🔍 {item.get('title', item.get('path', 'Result'))}"):
                if item["source"] == "canon":
                    st.write(f"**Canon Entry:** {item['value']}")
                else:
                    st.write(f"**Source:** {item['path']}")
                    st.write(f"**Score:** {item['score']}")
                    st.write(f"**Snippet:** {item['snippet']}")

# NPCs Tab
with tab2:
    st.header("NPC Generator")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎲 Generate Random NPC"):
            with st.spinner("Generating random NPC..."):
                npc_content = generate_random_npc(model)
                if not npc_content.startswith("Error"):
                    st.markdown(npc_content)
                    if st.button("💾 Save NPC"):
                        filepath = save_narrative_content(
                            npc_content, "npcs", "random_npc", st.session_state.data_dir
                        )
                        st.success(f"NPC saved to: {filepath}")
                else:
                    st.error(npc_content)

    with col2:
        st.subheader("Custom NPC")
        hook = st.text_input("Hook", placeholder="What makes this NPC interesting?")
        role = st.text_input("Role", placeholder="Their role in society")
        faction = st.text_input("Faction", placeholder="Their affiliated group")
        relationship = st.text_input(
            "Relationship to Party", placeholder="How they relate to the players"
        )
        secret = st.text_input(
            "Secret (optional)", placeholder="A hidden aspect of the NPC"
        )

        if st.button("Generate Custom NPC"):
            with st.spinner("Generating custom NPC..."):
                npc_content = generate_npc_description(
                    hook, role, faction, relationship, secret, model
                )
                if not npc_content.startswith("Error"):
                    st.markdown(npc_content)
                    if st.button("💾 Save Custom NPC"):
                        filepath = save_narrative_content(
                            npc_content,
                            "npcs",
                            role or "custom_npc",
                            st.session_state.data_dir,
                        )
                        st.success(f"NPC saved to: {filepath}")
                else:
                    st.error(npc_content)

# Locations Tab
with tab3:
    st.header("Location Generator")

    location_name = st.text_input(
        "Location Name", placeholder="Name or type of location"
    )
    tags = st.text_input(
        "Tags (comma-separated)", placeholder="mysterious, ancient, magical"
    )

    if st.button("Generate Location"):
        with st.spinner("Generating location..."):
            tag_list = [t.strip() for t in tags.split(",")] if tags else []
            location_content = generate_location_description(
                location_name, tag_list, model
            )

            if not location_content.startswith("Error"):
                st.markdown(location_content)
                if st.button("💾 Save Location"):
                    filepath = save_narrative_content(
                        location_content,
                        "locations",
                        location_name or "mysterious_location",
                        st.session_state.data_dir,
                    )
                    st.success(f"Location saved to: {filepath}")
            else:
                st.error(location_content)

# Scenes Tab
with tab4:
    st.header("Scene Generator")

    scene_prompt = st.text_area(
        "Scene Description", placeholder="Describe the scene you want to generate..."
    )

    if st.button("Generate Scene"):
        with st.spinner("Generating scene..."):
            scene_content = generate_scene_description(scene_prompt, model)

            if not scene_content.startswith("Error"):
                st.markdown(scene_content)
                if st.button("💾 Save Scene"):
                    filepath = save_narrative_content(
                        scene_content, "scenes", "scene", st.session_state.data_dir
                    )
                    st.success(f"Scene saved to: {filepath}")
            else:
                st.error(scene_content)

# Campaign Data Tab
with tab5:
    st.header("Campaign Data")

    # Load canon data
    canon_store = CanonStore(st.session_state.data_dir / CANON_FILE)

    # Function to read markdown content
    def read_markdown_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    # Display generated content sections
    content_sections = {
        "NPCs": st.session_state.data_dir / "npcs",
        "Locations": st.session_state.data_dir / "locations",
        "Scenes": st.session_state.data_dir / "scenes",
        "Notes": st.session_state.data_dir / "notes"
    }

    st.subheader("Generated Content")
    for section_name, directory in content_sections.items():
        with st.expander(f"📝 {section_name}"):
            if directory.exists():
                files = list(directory.glob("*.md"))
                if files:
                    for file in files:
                        with st.expander(f"📄 {file.stem}"):
                            content = read_markdown_file(file)
                            st.markdown(content)
                else:
                    st.info(f"No {section_name.lower()} files found.")
            else:
                st.info(f"Directory for {section_name.lower()} does not exist yet.")

    st.subheader("Campaign Canon")
    # Display canon sections
    sections = ["players", "npcs", "locations", "items", "quests", "timeline", "rules"]

    for section in sections:
        with st.expander(f"📚 {section.title()}"):
            st.json(canon_store.data.get(section, []))
