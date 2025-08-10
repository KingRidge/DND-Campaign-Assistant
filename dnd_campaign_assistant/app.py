import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid
import random
from dnd_campaign_assistant.campaign_assistant import (
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
from dnd_campaign_assistant.narrative_generator import (
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

# Create a unique session ID if it doesn't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()).replace('-', '')

def get_unique_key(base_name):
    """Generate a simple unique key for Streamlit elements"""
    # Use a simple approach that doesn't depend on session state
    return base_name

def main():
    """Main Streamlit app function"""
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
            key=get_unique_key("model_name_input")
        )

        # Ollama server URL
        ollama_host = st.text_input(
            "Ollama Server URL",
            value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            help="The URL of your Ollama server",
            key=get_unique_key("ollama_host_input")
        )

        model = (model_name, ollama_host, os.getenv("OLLAMA_KEY", ""))
        
        # Debug information
        st.subheader("Debug Info")
        st.write(f"Model tuple: {model}")
        
        # Test Ollama connection
        if st.button("🔍 Test Ollama Connection", key=get_unique_key("test_ollama_button")):
            try:
                import requests
                response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    st.success(f"✅ Connected! Available models: {len(models)}")
                    for model_info in models[:3]:  # Show first 3 models
                        st.write(f"- {model_info.get('name', 'Unknown')}")
                else:
                    st.error(f"❌ HTTP {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📝 Notes & Search", "👤 NPCs", "🏰 Locations", "⚔️ Scenes", "📊 Campaign Data"]
    )

    # Notes & Search Tab
    with tab1:
        st.header("Notes & Search")

        search_query = st.text_input(
            "Search your campaign notes and canon", 
            placeholder="Enter your search query...",
            key=get_unique_key("search_query_input")
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
            st.subheader("Random NPC")
            if st.button("🎲 Generate Random NPC", key=get_unique_key("generate_random_npc_button")):
                with st.spinner("Generating random NPC..."):
                    npc_content = generate_random_npc(model)
                    st.session_state.generated_random_npc = npc_content
                    st.session_state.show_save_random_npc = True
            
            # Display generated random NPC if it exists
            if hasattr(st.session_state, 'generated_random_npc'):
                if not st.session_state.generated_random_npc.startswith("Error"):
                    st.markdown(st.session_state.generated_random_npc)
                    if st.button("💾 Save Random NPC", key=get_unique_key("save_random_npc_button")):
                        filepath = save_narrative_content(
                            st.session_state.generated_random_npc, "npcs", "random_npc", st.session_state.data_dir
                        )
                        st.success(f"NPC saved to: {filepath}")
                        # Clear the generated content after saving
                        del st.session_state.generated_random_npc
                        st.rerun()
                else:
                    st.error(st.session_state.generated_random_npc)

        with col2:
            st.subheader("Custom NPC")
            hook = st.text_input("Hook", placeholder="What makes this NPC interesting?", key=get_unique_key("npc_hook_input"))
            role = st.text_input("Role", placeholder="Their role in society", key=get_unique_key("npc_role_input"))
            faction = st.text_input("Faction", placeholder="Their affiliated group", key=get_unique_key("npc_faction_input"))
            relationship = st.text_input(
                "Relationship to Party", placeholder="How they relate to the players", key=get_unique_key("npc_relationship_input")
            )
            secret = st.text_input(
                "Secret (optional)", placeholder="A hidden aspect of the NPC", key=get_unique_key("npc_secret_input")
            )

            if st.button("Generate Custom NPC", key=get_unique_key("generate_custom_npc_button")):
                with st.spinner("Generating custom NPC..."):
                    npc_content = generate_npc_description(
                        hook, role, faction, relationship, secret, model
                    )
                    st.session_state.generated_custom_npc = npc_content
                    st.session_state.custom_npc_role = role or "custom_npc"
            
            # Display generated custom NPC if it exists
            if hasattr(st.session_state, 'generated_custom_npc'):
                if not st.session_state.generated_custom_npc.startswith("Error"):
                    st.markdown(st.session_state.generated_custom_npc)
                    if st.button("💾 Save Custom NPC", key=get_unique_key("save_custom_npc_button")):
                        filepath = save_narrative_content(
                            st.session_state.generated_custom_npc,
                            "npcs",
                            st.session_state.custom_npc_role,
                            st.session_state.data_dir,
                        )
                        st.success(f"NPC saved to: {filepath}")
                        # Clear the generated content after saving
                        del st.session_state.generated_custom_npc
                        del st.session_state.custom_npc_role
                        st.rerun()
                else:
                    st.error(st.session_state.generated_custom_npc)

    # Locations Tab
    with tab3:
        st.header("Location Generator")

        location_name = st.text_input(
            "Location Name", placeholder="Name or type of location", key=get_unique_key("location_name_input")
        )
        tags = st.text_input(
            "Tags (comma-separated)", placeholder="mysterious, ancient, magical", key=get_unique_key("location_tags_input")
        )

        if st.button("Generate Location", key=get_unique_key("generate_location_button")):
            with st.spinner("Generating location..."):
                tag_list = [t.strip() for t in tags.split(",")] if tags else []
                location_content = generate_location_description(
                    location_name, tag_list, model
                )
                st.session_state.generated_location = location_content
                st.session_state.location_name_for_save = location_name or "mysterious_location"

        # Display generated location if it exists
        if hasattr(st.session_state, 'generated_location'):
            if not st.session_state.generated_location.startswith("Error"):
                st.markdown(st.session_state.generated_location)
                if st.button("💾 Save Location", key=get_unique_key("save_location_button")):
                    filepath = save_narrative_content(
                        st.session_state.generated_location,
                        "locations",
                        st.session_state.location_name_for_save,
                        st.session_state.data_dir,
                    )
                    st.success(f"Location saved to: {filepath}")
                    # Clear the generated content after saving
                    del st.session_state.generated_location
                    del st.session_state.location_name_for_save
                    st.rerun()
            else:
                st.error(st.session_state.generated_location)

    # Scenes Tab
    with tab4:
        st.header("Scene Generator")

        scene_prompt = st.text_area(
            "Scene Description", 
            placeholder="Describe the scene you want to generate...",
            key=get_unique_key("scene_description_input")
        )

        if st.button("Generate Scene", key=get_unique_key("generate_scene_button")):
            with st.spinner("Generating scene..."):
                scene_content = generate_scene_description(scene_prompt, model)
                st.session_state.generated_scene = scene_content

        # Display generated scene if it exists
        if hasattr(st.session_state, 'generated_scene'):
            if not st.session_state.generated_scene.startswith("Error"):
                st.markdown(st.session_state.generated_scene)
                if st.button("💾 Save Scene", key=get_unique_key("save_scene_button")):
                    filepath = save_narrative_content(
                        st.session_state.generated_scene, "scenes", "scene", st.session_state.data_dir
                    )
                    st.success(f"Scene saved to: {filepath}")
                    # Clear the generated content after saving
                    del st.session_state.generated_scene
                    st.rerun()
            else:
                st.error(st.session_state.generated_scene)

    # Campaign Data Tab
    with tab5:
        st.header("Campaign Data")

        # Load canon data
        canon_store = CanonStore(st.session_state.data_dir / CANON_FILE)

        # Function to read markdown content
        def read_markdown_file(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    return file.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"

        # Display generated content sections
        content_sections = {
            "NPCs": st.session_state.data_dir / "npcs",
            "Locations": st.session_state.data_dir / "locations",
            "Scenes": st.session_state.data_dir / "scenes",
            "Notes": st.session_state.data_dir / "notes",
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

if __name__ == "__main__":
    main()
