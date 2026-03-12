"""Mesoamerican entity taxonomy for structured extraction.

Provides pre-classified entities (people, places, deities, etc.) and
disambiguation rules so the extraction pipeline can correctly distinguish
e.g. "Mexica" (ethnic group) from "Mexico" (place).
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Entity type definitions
# ---------------------------------------------------------------------------

ENTITY_TYPES = {
    "person": "An individual historical or mythological figure",
    "people_group": "An ethnic group, nation, or people (e.g., Mexica, Nahua, Tlaxcalteca)",
    "place": "A geographic location — city, region, body of water, mountain",
    "political_entity": "A state, altepetl, or political unit",
    "deity": "A god, goddess, or supernatural being",
    "title_role": "A political, religious, or social title (e.g., tlatoani, cihuacoatl)",
    "date": "A calendrical date (Gregorian or Aztec tonalpohualli/xiuhpohualli)",
    "event": "A historical event, battle, ceremony, or migration",
}


# ---------------------------------------------------------------------------
# Pre-classified Mesoamerican entities (~100 key entries)
# ---------------------------------------------------------------------------

KNOWN_ENTITIES: List[Dict[str, str]] = [
    # --- People / Ethnic groups ---
    {"name": "Mexica", "type": "people_group", "note": "The Aztec people of Tenochtitlan. NOT a place."},
    {"name": "Nahua", "type": "people_group", "note": "Speakers of Nahuatl languages broadly."},
    {"name": "Tlaxcalteca", "type": "people_group", "note": "People of Tlaxcala, allies of the Spanish."},
    {"name": "Acolhua", "type": "people_group", "note": "People centered at Texcoco."},
    {"name": "Tepaneca", "type": "people_group", "note": "People centered at Azcapotzalco."},
    {"name": "Chalca", "type": "people_group", "note": "People of Chalco."},
    {"name": "Xochimilca", "type": "people_group", "note": "People of Xochimilco."},
    {"name": "Culhua", "type": "people_group", "note": "People of Culhuacan, claimed Toltec descent."},
    {"name": "Tolteca", "type": "people_group", "note": "Ancient people of Tula, semi-legendary."},
    {"name": "Chichimeca", "type": "people_group", "note": "Northern nomadic/semi-nomadic peoples."},
    {"name": "Otomi", "type": "people_group", "note": "Non-Nahuatl people of central Mexico."},
    {"name": "Mixteca", "type": "people_group", "note": "People of the Mixtec region, Oaxaca."},
    {"name": "Zapoteca", "type": "people_group", "note": "People of Oaxaca, centered at Monte Alban."},
    {"name": "Maya", "type": "people_group", "note": "People of Yucatan and Guatemala."},
    {"name": "Pipil", "type": "people_group", "note": "Nahua people of El Salvador."},
    {"name": "Totonaca", "type": "people_group", "note": "People of coastal Veracruz."},
    {"name": "Purepecha", "type": "people_group", "note": "People of Michoacan (also called Tarascan)."},

    # --- Historical individuals ---
    {"name": "Motecuhzoma", "type": "person", "note": "Moctezuma — ruler of Tenochtitlan. Variants: Montezuma, Moteuczoma."},
    {"name": "Motecuhzoma Xocoyotzin", "type": "person", "note": "Moctezuma II, ruler at time of Spanish contact."},
    {"name": "Motecuhzoma Ilhuicamina", "type": "person", "note": "Moctezuma I, 5th tlatoani of Tenochtitlan."},
    {"name": "Cuauhtemoc", "type": "person", "note": "Last tlatoani of Tenochtitlan."},
    {"name": "Cuitlahuac", "type": "person", "note": "Penultimate tlatoani, ruled briefly during siege."},
    {"name": "Itzcoatl", "type": "person", "note": "4th tlatoani, founder of the Triple Alliance."},
    {"name": "Nezahualcoyotl", "type": "person", "note": "Philosopher-king of Texcoco, poet and lawgiver."},
    {"name": "Nezahualpilli", "type": "person", "note": "Son of Nezahualcoyotl, ruler of Texcoco."},
    {"name": "Tlacaelel", "type": "person", "note": "Cihuacoatl (advisor) who reshaped Aztec ideology."},
    {"name": "Hernan Cortes", "type": "person", "note": "Spanish conquistador. Nahuatl: Malintzin's companion."},
    {"name": "Malintzin", "type": "person", "note": "La Malinche / Dona Marina, interpreter for Cortes."},
    {"name": "Bernardino de Sahagun", "type": "person", "note": "Franciscan friar, compiled the Florentine Codex."},
    {"name": "Acamapichtli", "type": "person", "note": "First tlatoani of Tenochtitlan."},
    {"name": "Huitzilihuitl", "type": "person", "note": "Second tlatoani of Tenochtitlan."},
    {"name": "Chimalpopoca", "type": "person", "note": "Third tlatoani of Tenochtitlan."},
    {"name": "Axayacatl", "type": "person", "note": "Sixth tlatoani of Tenochtitlan."},
    {"name": "Tizoc", "type": "person", "note": "Seventh tlatoani of Tenochtitlan."},
    {"name": "Ahuitzotl", "type": "person", "note": "Eighth tlatoani, expanded empire greatly."},

    # --- Places ---
    {"name": "Mexico", "type": "place", "note": "Place name (modern country or Mexico-Tenochtitlan). NOT the people — that is 'Mexica'."},
    {"name": "Tenochtitlan", "type": "place", "note": "Capital of the Aztec empire, now Mexico City."},
    {"name": "Tlatelolco", "type": "place", "note": "Twin city of Tenochtitlan, major market center."},
    {"name": "Texcoco", "type": "place", "note": "Capital of Acolhua, part of Triple Alliance."},
    {"name": "Tlacopan", "type": "place", "note": "Third member of Triple Alliance (also Tacuba)."},
    {"name": "Tlaxcala", "type": "place", "note": "BOTH a place AND political entity. Independent state that allied with Spain."},
    {"name": "Cholula", "type": "place", "note": "Sacred city, major temple to Quetzalcoatl."},
    {"name": "Tula", "type": "place", "note": "Capital of the Toltecs (Tollan)."},
    {"name": "Azcapotzalco", "type": "place", "note": "Tepanec capital, defeated by Triple Alliance."},
    {"name": "Culhuacan", "type": "place", "note": "Ancient city claiming Toltec heritage."},
    {"name": "Chapultepec", "type": "place", "note": "'Grasshopper hill' — important site in Mexica migration."},
    {"name": "Xochimilco", "type": "place", "note": "City famous for chinampas (floating gardens)."},
    {"name": "Chalco", "type": "place", "note": "City-state south of Lake Texcoco."},
    {"name": "Coatepec", "type": "place", "note": "'Serpent hill' — mythological birthplace of Huitzilopochtli."},
    {"name": "Aztlan", "type": "place", "note": "Mythical homeland of the Mexica."},
    {"name": "Mictlan", "type": "place", "note": "Underworld / realm of the dead."},
    {"name": "Tamoanchan", "type": "place", "note": "Mythical paradise of origin."},
    {"name": "Anahuac", "type": "place", "note": "'Near the water' — the Valley of Mexico."},
    {"name": "Cemanahuac", "type": "place", "note": "The known world / earth."},

    # --- Deities ---
    {"name": "Huitzilopochtli", "type": "deity", "note": "Sun and war god, patron of the Mexica."},
    {"name": "Quetzalcoatl", "type": "deity", "note": "Feathered Serpent, god of wind and learning."},
    {"name": "Tezcatlipoca", "type": "deity", "note": "Smoking Mirror, god of fate and sorcery."},
    {"name": "Tlaloc", "type": "deity", "note": "Rain god, one of the oldest Mesoamerican deities."},
    {"name": "Chalchiuhtlicue", "type": "deity", "note": "Goddess of water and rivers, consort of Tlaloc."},
    {"name": "Xipe Totec", "type": "deity", "note": "Flayed Lord, god of spring and renewal."},
    {"name": "Mictlantecuhtli", "type": "deity", "note": "Lord of the dead, ruler of Mictlan."},
    {"name": "Coatlicue", "type": "deity", "note": "Mother of Huitzilopochtli, earth goddess."},
    {"name": "Tonatiuh", "type": "deity", "note": "Sun god (the current Fifth Sun)."},
    {"name": "Tlazolteotl", "type": "deity", "note": "Goddess of filth, purification, and confession."},
    {"name": "Xochiquetzal", "type": "deity", "note": "Goddess of beauty, flowers, and love."},
    {"name": "Centeotl", "type": "deity", "note": "Maize god."},
    {"name": "Ehecatl", "type": "deity", "note": "Wind god, aspect of Quetzalcoatl."},
    {"name": "Xiuhtecuhtli", "type": "deity", "note": "Fire god, lord of the year."},
    {"name": "Ometeotl", "type": "deity", "note": "Dual god of creation (Ometecuhtli/Omecihuatl)."},

    # --- Titles and roles ---
    {"name": "tlatoani", "type": "title_role", "note": "Ruler / speaker / king. Plural: tlatoque."},
    {"name": "huey tlatoani", "type": "title_role", "note": "Great speaker — supreme ruler of Tenochtitlan."},
    {"name": "cihuacoatl", "type": "title_role", "note": "Woman-serpent — second-in-command, chief advisor."},
    {"name": "tlacochcalcatl", "type": "title_role", "note": "Military general, one of two top commanders."},
    {"name": "tlacateccatl", "type": "title_role", "note": "Military judge/commander, co-general."},
    {"name": "calpixqui", "type": "title_role", "note": "Tax collector / steward."},
    {"name": "pochteca", "type": "title_role", "note": "Long-distance merchants (also intelligence gatherers)."},
    {"name": "tlamacazqui", "type": "title_role", "note": "Priest / one who offers things."},
    {"name": "tlamatini", "type": "title_role", "note": "Sage / wise person / philosopher."},
    {"name": "cuicani", "type": "title_role", "note": "Singer / poet."},
    {"name": "tlacuilo", "type": "title_role", "note": "Scribe / painter of codices."},
    {"name": "pilli", "type": "title_role", "note": "Noble / prince. Plural: pipiltin."},
    {"name": "macehualtin", "type": "title_role", "note": "Commoners (plural of macehualli)."},
]


# ---------------------------------------------------------------------------
# Disambiguation rules (injected into extraction prompts)
# ---------------------------------------------------------------------------

DISAMBIGUATION_RULES = """\
ENTITY DISAMBIGUATION RULES FOR MESOAMERICAN TEXTS:
- "Mexica" = people/ethnic group (the Aztec people). NEVER classify as a place.
- "Mexico" = place (the city Mexico-Tenochtitlan, or the modern country). NEVER classify as a people.
- "Tlaxcala" = BOTH a place AND a political entity. Classify based on context.
- "Nahua" / "Nahuatl" = people/language family, NOT a place.
- "Tolteca" / "Toltec" = people group (ancient), NOT just a place (Tula is the place).
- "Chichimeca" = people group (northern peoples), NOT a place.
- Names ending in "-ca" or "-tecatl" are usually PEOPLE demonyms (e.g., Chalca = people of Chalco).
- Names ending in "-tlan", "-co", "-pan", "-can" are usually PLACES (locative suffixes).
- "tlatoani" is a TITLE/ROLE, not a personal name.
- When a name like "Motecuhzoma" appears, classify as a PERSON, not as a title.
- Aztec calendar names (e.g., "Ce Acatl", "Nahui Ollin") are DATES, not people.
- If a term appears as both a deity name and a title (e.g., "Cihuacoatl"), use context to decide.\
"""


def format_entity_reference(max_per_type: int = 8) -> str:
    """Format the entity taxonomy as a reference block for prompt injection."""
    by_type: Dict[str, List[Dict[str, str]]] = {}
    for e in KNOWN_ENTITIES:
        by_type.setdefault(e["type"], []).append(e)

    lines = ["KNOWN MESOAMERICAN ENTITIES (use for classification guidance):"]
    for etype, label in ENTITY_TYPES.items():
        entries = by_type.get(etype, [])
        if not entries:
            continue
        lines.append(f"\n{etype.upper()} ({label}):")
        for e in entries[:max_per_type]:
            lines.append(f"  - {e['name']}: {e.get('note', '')}")
        if len(entries) > max_per_type:
            lines.append(f"  ... and {len(entries) - max_per_type} more")

    return "\n".join(lines)


# Default Mesoamerican-aware extraction schema
MESOAMERICAN_SCHEMA = {
    "people": [{"name": "", "type": "individual|ethnic_group|title"}],
    "places": [{"name": "", "type": "city|region|geographic_feature"}],
    "deities": [{"name": "", "domain": ""}],
    "dates": [{"text": "", "calendar": "gregorian|aztec"}],
    "events": [{"description": "", "participants": []}],
    "titles_roles": [{"title": "", "holder": ""}],
}
