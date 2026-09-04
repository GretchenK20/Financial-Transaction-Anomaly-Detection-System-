/*
  Silver: encounter summary per patient.
  Counts and patterns used as utilization features.
*/
with enc as (
    select * from {{ ref('stg_encounters') }}
    where status = 'finished'
),

summary as (
    select
        patient_id,
        count(*)                                                    as total_encounters,
        count(case when class = 'AMB' then 1 end)                  as ambulatory_count,
        count(case when class = 'EMER' then 1 end)                 as emergency_count,
        count(case when class = 'IMP' then 1 end)                  as inpatient_count,
        min(start_date)                                             as first_encounter_date,
        max(start_date)                                             as last_encounter_date,
        -- Days between first and last encounter (care continuity proxy)
        datediff('day', min(start_date), max(start_date))          as care_span_days,
        -- Encounters in last 12 months
        count(case when start_date >= current_date - interval '12 months'
                   then 1 end)                                      as encounters_last_12m
    from enc
    group by patient_id
)

select
    *,
    case when emergency_count >= 2 then 1 else 0 end as high_ed_utilizer
from summary
