select
    encounter_id,
    patient_id,
    type_code,
    type_display,
    class,
    status,
    start_date,
    end_date,
    ingested_at
from bronze_encounters
