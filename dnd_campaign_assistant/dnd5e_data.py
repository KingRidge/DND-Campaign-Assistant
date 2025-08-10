"""D&D 5e data structures for random generation."""

# Basic D&D 5e races (PHB)
RACES = [
    "Human",
    "Elf",
    "Dwarf",
    "Halfling",
    "Half-Elf",
    "Half-Orc",
    "Gnome",
    "Tiefling",
    "Dragonborn",
]

# Basic D&D 5e classes (PHB)
CLASSES = [
    "Fighter",
    "Wizard",
    "Cleric",
    "Rogue",
    "Ranger",
    "Paladin",
    "Barbarian",
    "Bard",
    "Druid",
    "Monk",
    "Sorcerer",
    "Warlock",
]

# Common NPC occupations in D&D settings
OCCUPATIONS = [
    # Merchants and Craftspeople
    "Merchant",
    "Blacksmith",
    "Carpenter",
    "Tailor",
    "Jeweler",
    "Potter",
    "Leatherworker",
    "Weaver",
    "Alchemist",
    "Herbalist",
    # Service Providers
    "Innkeeper",
    "Bartender",
    "Cook",
    "Healer",
    "Sage",
    "Scribe",
    "Guide",
    "Messenger",
    "Stablemaster",
    "Shipwright",
    # Religious and Magical
    "Priest",
    "Acolyte",
    "Fortune Teller",
    "Hedge Wizard",
    "Scholar",
    "Temple Guardian",
    "Magic Item Dealer",
    "Potion Maker",
    # Entertainment
    "Bard",
    "Storyteller",
    "Actor",
    "Artist",
    "Musician",
    "Dancer",
    # Criminal Elements
    "Fence",
    "Smuggler",
    "Information Broker",
    "Guild Thief",
    "Black Market Dealer",
    # Military and Security
    "Guard Captain",
    "City Watch",
    "Mercenary",
    "Veteran",
    "Scout",
    "Weapon Master",
    "Arena Fighter",
    # Political and Noble
    "Noble",
    "Diplomat",
    "Court Advisor",
    "Magistrate",
    "Tax Collector",
    "Town Official",
    "Guild Master",
]

# Common factions in D&D settings
FACTIONS = [
    # Government and Military
    "City Guard",
    "Royal Court",
    "Noble House",
    "Merchant's Guild",
    # Religious Organizations
    "Temple of Light",
    "Druidic Circle",
    "Monastic Order",
    "Holy Order",
    "Dark Cult",
    # Arcane Groups
    "Mage's Guild",
    "Arcane Academy",
    "Wizard's Circle",
    "Sorcerer's Cabal",
    # Criminal Organizations
    "Thieves' Guild",
    "Assassins' League",
    "Smuggler's Ring",
    "Criminal Syndicate",
    # Mercenary Companies
    "Mercenary Company",
    "Adventurers' Guild",
    "Bounty Hunters' League",
    # Trade Organizations
    "Merchant's League",
    "Craftsmen's Guild",
    "Trading Company",
    # Secret Societies
    "Secret Society",
    "Ancient Order",
    "Hidden Circle",
    "Shadow Council",
]

# Personality traits for NPCs
PERSONALITY_TRAITS = [
    # Positive Traits
    "Brave",
    "Honest",
    "Loyal",
    "Kind",
    "Generous",
    "Wise",
    "Patient",
    "Curious",
    "Creative",
    "Determined",
    "Diplomatic",
    "Energetic",
    # Neutral Traits
    "Cautious",
    "Reserved",
    "Practical",
    "Traditional",
    "Ambitious",
    "Proud",
    "Independent",
    "Mysterious",
    "Eccentric",
    "Calculated",
    # Flawed Traits
    "Greedy",
    "Suspicious",
    "Arrogant",
    "Stubborn",
    "Vengeful",
    "Paranoid",
    "Impulsive",
    "Cowardly",
    "Manipulative",
    "Jealous",
]

# NPC hooks and secrets
HOOKS = [
    # Quests and Missions
    "Seeks a rare artifact",
    "Needs help with a dangerous mission",
    "Looking for missing person",
    "Wants revenge against enemies",
    "Protecting valuable secret",
    "Planning a heist or mission",
    # Personal Problems
    "In debt to dangerous people",
    "Cursed by ancient magic",
    "Haunted by past mistakes",
    "Running from powerful enemies",
    "Trapped in magical contract",
    "Lost family inheritance",
    # Knowledge and Secrets
    "Knows forbidden knowledge",
    "Discovered ancient prophecy",
    "Holds evidence of conspiracy",
    "Found magical anomaly",
    "Uncovered noble's secret",
    "Has map to lost treasure",
    # Relationships
    "Caught between rival factions",
    "Secret love affair",
    "Betrayed by trusted ally",
    "Indebted to crime lord",
    "Serving multiple masters",
    "Family in danger",
]

# NPC relationships to party
RELATIONSHIPS = [
    # Positive
    "Potential ally seeking help",
    "Grateful for past aid",
    "Shares common enemy",
    "Impressed by party's reputation",
    "Needs their specific skills",
    "Friend of a friend",
    # Neutral
    "Professional interest only",
    "Cautiously evaluating",
    "Diplomatic necessity",
    "Mutual benefit arrangement",
    "Reserved but respectful",
    "Testing their trustworthiness",
    # Complex
    "Friend despite faction rivalry",
    "Forced to work together",
    "Respects but can't fully trust",
    "Conflicting loyalties",
    "Professional rivals",
    "Complicated past history",
    # Potentially Negative
    "Suspicious of motives",
    "Past grievance to overcome",
    "Competing interests",
    "Bound by circumstance",
    "Reluctant cooperation",
    "Testing their worth",
]
