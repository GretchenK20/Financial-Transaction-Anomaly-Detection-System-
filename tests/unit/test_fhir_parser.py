"""Unit tests for FHIR parser — runs against a minimal synthetic bundle."""
import json
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.fhir_parser import parse_bundle, parse_patient, parse_condition

SYNTHETIC_BUNDLE = {
    "resourceType": "Bundle",
    "type": "transaction",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "test-patient-001",
                "name": [{"given": ["Ada"], "family": "Howell"}],
                "birthDate": "1970-03-15",
                "gender": "female",
                "address": [{"city": "Boston", "state": "MA", "postalCode": "02101"}],
                "maritalStatus": {"text": "M"},
                "extension": [
                    {
                        "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                        "extension": [{"url": "text", "valueString": "White"}],
                    },
                    {
                        "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                        "extension": [{"url": "text", "valueString": "Not Hispanic or Latino"}],
                    },
                ],
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-001",
                "subject": {"reference": "Patient/test-patient-001"},
                "code": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "44054006",
                        "display": "Diabetes mellitus type 2",
                    }]
                },
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "onsetDateTime": "2015-06-01",
                "category": [{"coding": [{"display": "Encounter Diagnosis"}]}],
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-001",
                "subject": {"reference": "Patient/test-patient-001"},
                "status": "final",
                "code": {"coding": [{"system": "http://loinc.org", "code": "8302-2", "display": "Body Height"}]},
                "valueQuantity": {"value": 167.5, "unit": "cm"},
                "effectiveDateTime": "2023-01-10",
                "category": [{"coding": [{"display": "vital-signs"}]}],
            }
        },
        {
            "resource": {
                "resourceType": "Encounter",
                "id": "enc-001",
                "subject": {"reference": "Patient/test-patient-001"},
                "status": "finished",
                "class": {"code": "AMB"},
                "type": [{"coding": [{"code": "185349003", "display": "Encounter for check up"}]}],
                "period": {"start": "2023-01-10", "end": "2023-01-10"},
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-001",
                "subject": {"reference": "Patient/test-patient-001"},
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [{"code": "860975", "display": "Metformin 500mg"}]
                },
                "authoredOn": "2023-01-10",
            }
        },
    ],
}


@pytest.fixture
def bundle_file(tmp_path):
    p = tmp_path / "test_patient.json"
    p.write_text(json.dumps(SYNTHETIC_BUNDLE))
    return p


def test_parse_patient_fields():
    resource = SYNTHETIC_BUNDLE["entry"][0]["resource"]
    patient = parse_patient(resource)
    assert patient["patient_id"] == "test-patient-001"
    assert patient["given_name"] == "Ada"
    assert patient["family_name"] == "Howell"
    assert patient["gender"] == "female"
    assert patient["city"] == "Boston"
    assert patient["state"] == "MA"
    assert patient["race"] == "White"
    assert patient["ethnicity"] == "Not Hispanic or Latino"
    assert patient["age_years"] is not None and patient["age_years"] > 50
    assert patient["deceased"] is False


def test_parse_condition_fields():
    resource = SYNTHETIC_BUNDLE["entry"][1]["resource"]
    condition = parse_condition(resource, "test-patient-001")
    assert condition["patient_id"] == "test-patient-001"
    assert condition["code"] == "44054006"
    assert "Diabetes" in condition["display"]
    assert condition["clinical_status"] == "active"


def test_parse_bundle_returns_all_resource_types(bundle_file):
    result = parse_bundle(bundle_file)
    assert len(result["patients"]) == 1
    assert len(result["conditions"]) == 1
    assert len(result["observations"]) == 1
    assert len(result["encounters"]) == 1
    assert len(result["medication_requests"]) == 1


def test_parse_bundle_patient_id_propagated(bundle_file):
    result = parse_bundle(bundle_file)
    for cond in result["conditions"]:
        assert cond["patient_id"] == "test-patient-001"
    for obs in result["observations"]:
        assert obs["patient_id"] == "test-patient-001"


def test_missing_patient_returns_empty(tmp_path):
    bundle_no_patient = {"resourceType": "Bundle", "type": "transaction", "entry": []}
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(bundle_no_patient))
    result = parse_bundle(p)
    assert result["patients"] == []
    assert result["conditions"] == []


def test_bronze_loader_with_synthetic_bundle(tmp_path, bundle_file):
    import duckdb
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ingestion.bronze_loader import load_bundles

    db_path = tmp_path / "test.duckdb"
    counts = load_bundles(
        fhir_dir=bundle_file.parent,
        db_path=db_path,
        limit=1,
    )
    assert counts["bronze_patients"] == 1
    assert counts["bronze_conditions"] == 1
    assert counts["bronze_observations"] == 1
    assert counts["bronze_encounters"] == 1
    assert counts["bronze_medication_requests"] == 1

    with duckdb.connect(str(db_path)) as conn:
        patient = conn.execute(
            "SELECT * FROM bronze_patients WHERE patient_id = 'test-patient-001'"
        ).fetchdf()
        assert len(patient) == 1
        assert patient.iloc[0]["gender"] == "female"
        assert patient.iloc[0]["race"] == "White"
