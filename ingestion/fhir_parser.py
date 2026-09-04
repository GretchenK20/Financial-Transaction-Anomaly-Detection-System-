"""
Parse Synthea FHIR R4 patient bundles into structured records.
Extracts: Patient demographics, Conditions, Observations, Encounters, MedicationRequests.
"""
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional
from loguru import logger


def _safe_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str[:10]).date()
    except (ValueError, TypeError):
        return None


def _age_from_birthdate(birthdate_str: Optional[str]) -> Optional[int]:
    d = _safe_date(birthdate_str)
    if not d:
        return None
    today = date.today()
    return (today - d).days // 365


def _extract_extension_text(extensions: list, url_fragment: str) -> Optional[str]:
    """
    Find extension by URL fragment match (handles both full and short URL forms).
    Synthea uses full HL7 URLs; test bundles may use short forms.
    """
    for ext in extensions:
        url = ext.get("url", "")
        if url_fragment in url:
            for sub in ext.get("extension", []):
                if sub.get("url") == "text":
                    return sub.get("valueString")
            # Also handle flat valueString directly on the extension
            if "valueString" in ext:
                return ext["valueString"]
    return None


def parse_patient(resource: dict) -> dict:
    name = resource.get("name", [{}])[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")

    extensions = resource.get("extension", [])
    race = _extract_extension_text(extensions, "race")
    ethnicity = _extract_extension_text(extensions, "ethnicity")

    address = resource.get("address", [{}])[0]

    return {
        "patient_id": resource.get("id"),
        "given_name": given,
        "family_name": family,
        "birth_date": resource.get("birthDate"),
        "age_years": _age_from_birthdate(resource.get("birthDate")),
        "gender": resource.get("gender"),
        "race": race,
        "ethnicity": ethnicity,
        "marital_status": resource.get("maritalStatus", {}).get("text"),
        "city": address.get("city"),
        "state": address.get("state"),
        "postal_code": address.get("postalCode"),
        "deceased": resource.get("deceasedBoolean", False) or bool(resource.get("deceasedDateTime")),
    }


def parse_condition(resource: dict, patient_id: str) -> dict:
    code_obj = resource.get("code", {})
    codings = code_obj.get("coding", [{}])
    primary = codings[0] if codings else {}

    return {
        "condition_id": resource.get("id"),
        "patient_id": patient_id,
        "code": primary.get("code"),
        "display": primary.get("display"),
        "system": primary.get("system"),
        "onset_date": _safe_date(resource.get("onsetDateTime")),
        "abatement_date": _safe_date(resource.get("abatementDateTime")),
        "clinical_status": resource.get("clinicalStatus", {})
            .get("coding", [{}])[0].get("code"),
        "category": resource.get("category", [{}])[0]
            .get("coding", [{}])[0].get("display"),
    }


def parse_observation(resource: dict, patient_id: str) -> dict:
    code_obj = resource.get("code", {})
    codings = code_obj.get("coding", [{}])
    primary = codings[0] if codings else {}

    value_quantity = resource.get("valueQuantity", {})
    value_concept = resource.get("valueCodeableConcept", {})

    return {
        "observation_id": resource.get("id"),
        "patient_id": patient_id,
        "code": primary.get("code"),
        "display": primary.get("display"),
        "system": primary.get("system"),
        "effective_date": _safe_date(resource.get("effectiveDateTime")),
        "status": resource.get("status"),
        "value_numeric": value_quantity.get("value"),
        "value_unit": value_quantity.get("unit"),
        "value_text": value_concept.get("text"),
        "category": resource.get("category", [{}])[0]
            .get("coding", [{}])[0].get("display"),
    }


def parse_encounter(resource: dict, patient_id: str) -> dict:
    period = resource.get("period", {})
    type_list = resource.get("type", [{}])
    type_coding = type_list[0].get("coding", [{}])[0] if type_list else {}

    return {
        "encounter_id": resource.get("id"),
        "patient_id": patient_id,
        "type_code": type_coding.get("code"),
        "type_display": type_coding.get("display"),
        "class": resource.get("class", {}).get("code"),
        "status": resource.get("status"),
        "start_date": _safe_date(period.get("start")),
        "end_date": _safe_date(period.get("end")),
    }


def parse_medication_request(resource: dict, patient_id: str) -> dict:
    medication = resource.get("medicationCodeableConcept", {})
    codings = medication.get("coding", [{}])
    primary = codings[0] if codings else {}

    return {
        "medication_id": resource.get("id"),
        "patient_id": patient_id,
        "code": primary.get("code"),
        "display": primary.get("display"),
        "status": resource.get("status"),
        "authored_date": _safe_date(resource.get("authoredOn")),
        "intent": resource.get("intent"),
    }


def parse_bundle(bundle_path: Path) -> dict[str, list]:
    records: dict[str, list] = {
        "patients": [],
        "conditions": [],
        "observations": [],
        "encounters": [],
        "medication_requests": [],
    }

    with open(bundle_path) as f:
        bundle = json.load(f)

    if bundle.get("resourceType") != "Bundle":
        logger.warning(f"Skipping {bundle_path.name} — not a Bundle")
        return records

    entries = bundle.get("entry", [])

    patient_id = None
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            patient_id = resource.get("id")
            break

    if not patient_id:
        logger.warning(f"No Patient resource in {bundle_path.name}")
        return records

    for entry in entries:
        resource = entry.get("resource", {})
        rt = resource.get("resourceType")

        if rt == "Patient":
            records["patients"].append(parse_patient(resource))
        elif rt == "Condition":
            records["conditions"].append(parse_condition(resource, patient_id))
        elif rt == "Observation":
            records["observations"].append(parse_observation(resource, patient_id))
        elif rt == "Encounter":
            records["encounters"].append(parse_encounter(resource, patient_id))
        elif rt == "MedicationRequest":
            records["medication_requests"].append(
                parse_medication_request(resource, patient_id)
            )

    return records
