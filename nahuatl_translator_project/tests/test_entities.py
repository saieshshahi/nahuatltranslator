"""Unit tests for webapp/entities.py — Mesoamerican entity taxonomy and disambiguation."""

import pytest
from webapp.entities import (
    ENTITY_TYPES,
    KNOWN_ENTITIES,
    DISAMBIGUATION_RULES,
    MESOAMERICAN_SCHEMA,
    format_entity_reference,
)


# ===================================================================
# Entity type definitions
# ===================================================================

class TestEntityTypes:
    def test_required_types_exist(self):
        required = [
            "person", "people_group", "place", "political_entity",
            "deity", "title_role", "date", "event",
        ]
        for t in required:
            assert t in ENTITY_TYPES, f"Missing entity type: {t}"

    def test_all_types_have_descriptions(self):
        for type_name, description in ENTITY_TYPES.items():
            assert len(description) > 10, f"Type '{type_name}' has no proper description"


# ===================================================================
# Known entities
# ===================================================================

class TestKnownEntities:
    def test_minimum_entity_count(self):
        assert len(KNOWN_ENTITIES) >= 80, f"Expected 80+ entities, got {len(KNOWN_ENTITIES)}"

    def test_entity_structure(self):
        for entity in KNOWN_ENTITIES:
            assert "name" in entity, f"Entity missing 'name': {entity}"
            assert "type" in entity, f"Entity missing 'type': {entity}"
            assert "note" in entity, f"Entity missing 'note': {entity}"

    def test_entity_types_are_valid(self):
        for entity in KNOWN_ENTITIES:
            assert entity["type"] in ENTITY_TYPES, (
                f"Entity '{entity['name']}' has invalid type: {entity['type']}"
            )

    def test_mexica_is_people_group(self):
        mexica = [e for e in KNOWN_ENTITIES if e["name"] == "Mexica"]
        assert len(mexica) == 1
        assert mexica[0]["type"] == "people_group"

    def test_mexico_is_place(self):
        mexico = [e for e in KNOWN_ENTITIES if e["name"] == "Mexico"]
        assert len(mexico) == 1
        assert mexico[0]["type"] == "place"

    def test_tenochtitlan_is_place(self):
        t = [e for e in KNOWN_ENTITIES if e["name"] == "Tenochtitlan"]
        assert len(t) == 1
        assert t[0]["type"] == "place"

    def test_huitzilopochtli_is_deity(self):
        h = [e for e in KNOWN_ENTITIES if e["name"] == "Huitzilopochtli"]
        assert len(h) == 1
        assert h[0]["type"] == "deity"

    def test_quetzalcoatl_is_deity(self):
        q = [e for e in KNOWN_ENTITIES if e["name"] == "Quetzalcoatl"]
        assert len(q) == 1
        assert q[0]["type"] == "deity"

    def test_tlatoani_is_title(self):
        t = [e for e in KNOWN_ENTITIES if e["name"] == "tlatoani"]
        assert len(t) == 1
        assert t[0]["type"] == "title_role"

    def test_motecuhzoma_is_person(self):
        m = [e for e in KNOWN_ENTITIES if e["name"] == "Motecuhzoma"]
        assert len(m) == 1
        assert m[0]["type"] == "person"

    def test_cuauhtemoc_is_person(self):
        c = [e for e in KNOWN_ENTITIES if e["name"] == "Cuauhtemoc"]
        assert len(c) == 1
        assert c[0]["type"] == "person"

    def test_tlaxcalteca_is_people_group(self):
        t = [e for e in KNOWN_ENTITIES if e["name"] == "Tlaxcalteca"]
        assert len(t) == 1
        assert t[0]["type"] == "people_group"

    def test_mictlan_is_place(self):
        m = [e for e in KNOWN_ENTITIES if e["name"] == "Mictlan"]
        assert len(m) == 1
        assert m[0]["type"] == "place"

    def test_no_duplicate_entities(self):
        names = [e["name"] for e in KNOWN_ENTITIES]
        duplicates = [n for n in names if names.count(n) > 1]
        assert len(duplicates) == 0, f"Duplicate entities found: {set(duplicates)}"

    def test_has_people_groups(self):
        groups = [e for e in KNOWN_ENTITIES if e["type"] == "people_group"]
        assert len(groups) >= 10, "Expected at least 10 people groups"

    def test_has_persons(self):
        persons = [e for e in KNOWN_ENTITIES if e["type"] == "person"]
        assert len(persons) >= 10, "Expected at least 10 historical persons"

    def test_has_places(self):
        places = [e for e in KNOWN_ENTITIES if e["type"] == "place"]
        assert len(places) >= 10, "Expected at least 10 places"

    def test_has_deities(self):
        deities = [e for e in KNOWN_ENTITIES if e["type"] == "deity"]
        assert len(deities) >= 10, "Expected at least 10 deities"

    def test_has_titles(self):
        titles = [e for e in KNOWN_ENTITIES if e["type"] == "title_role"]
        assert len(titles) >= 5, "Expected at least 5 titles/roles"


# ===================================================================
# Disambiguation rules
# ===================================================================

class TestDisambiguationRules:
    def test_mexica_rule(self):
        assert "Mexica" in DISAMBIGUATION_RULES
        assert "people" in DISAMBIGUATION_RULES.lower()

    def test_mexico_rule(self):
        assert "Mexico" in DISAMBIGUATION_RULES
        assert "place" in DISAMBIGUATION_RULES.lower()

    def test_ca_suffix_rule(self):
        assert "-ca" in DISAMBIGUATION_RULES

    def test_tlan_suffix_rule(self):
        assert "-tlan" in DISAMBIGUATION_RULES

    def test_tlatoani_rule(self):
        assert "tlatoani" in DISAMBIGUATION_RULES
        assert "TITLE" in DISAMBIGUATION_RULES

    def test_calendar_names_rule(self):
        assert "calendar" in DISAMBIGUATION_RULES.lower()
        assert "DATE" in DISAMBIGUATION_RULES

    def test_locative_suffixes_rule(self):
        locatives = ["-tlan", "-co", "-pan", "-can"]
        for loc in locatives:
            assert loc in DISAMBIGUATION_RULES, f"Missing locative rule for {loc}"


# ===================================================================
# Entity reference formatting
# ===================================================================

class TestFormatEntityReference:
    def test_produces_output(self):
        ref = format_entity_reference()
        assert len(ref) > 100
        assert "KNOWN MESOAMERICAN ENTITIES" in ref

    def test_contains_type_headers(self):
        ref = format_entity_reference()
        assert "PEOPLE_GROUP" in ref
        assert "PERSON" in ref
        assert "PLACE" in ref
        assert "DEITY" in ref

    def test_max_per_type_limit(self):
        ref_small = format_entity_reference(max_per_type=2)
        ref_large = format_entity_reference(max_per_type=20)
        assert len(ref_small) < len(ref_large)

    def test_contains_entity_names(self):
        ref = format_entity_reference()
        assert "Mexica" in ref
        assert "Tenochtitlan" in ref
        assert "Huitzilopochtli" in ref
        assert "tlatoani" in ref

    def test_truncation_indicator(self):
        ref = format_entity_reference(max_per_type=2)
        assert "... and" in ref


# ===================================================================
# Mesoamerican schema
# ===================================================================

class TestMesoamericanSchema:
    def test_schema_has_required_fields(self):
        required_keys = ["people", "places", "deities", "dates", "events", "titles_roles"]
        for key in required_keys:
            assert key in MESOAMERICAN_SCHEMA, f"Missing schema key: {key}"

    def test_schema_values_are_lists(self):
        for key, value in MESOAMERICAN_SCHEMA.items():
            assert isinstance(value, list), f"Schema '{key}' should be a list"

    def test_people_schema_structure(self):
        assert "name" in MESOAMERICAN_SCHEMA["people"][0]
        assert "type" in MESOAMERICAN_SCHEMA["people"][0]

    def test_places_schema_structure(self):
        assert "name" in MESOAMERICAN_SCHEMA["places"][0]
        assert "type" in MESOAMERICAN_SCHEMA["places"][0]

    def test_dates_schema_structure(self):
        assert "text" in MESOAMERICAN_SCHEMA["dates"][0]
        assert "calendar" in MESOAMERICAN_SCHEMA["dates"][0]
