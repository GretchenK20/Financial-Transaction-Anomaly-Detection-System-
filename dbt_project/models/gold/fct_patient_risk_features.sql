/*
  Gold: final feature mart for ML model consumption.
  One row per patient. All features are numeric or binary — ready for
  PyTorch autoencoder and XGBoost without further preprocessing.

  Joins:
    bronze_patients → demographics
    int_patient_conditions → chronic condition flags
    int_patient_vitals → latest vitals
    int_patient_encounters → utilization patterns
*/
with patients as (
    select * from {{ ref('stg_patients') }}
),

conditions as (
    select * from {{ ref('int_patient_conditions') }}
),

vitals as (
    select * from {{ ref('int_patient_vitals') }}
),

encounters as (
    select * from {{ ref('int_patient_encounters') }}
),

joined as (
    select
        p.patient_id,

        -- Demographics
        p.age_years,
        case when p.gender = 'male'   then 1
             when p.gender = 'female' then 0
             else null end                              as gender_male,
        case when p.deceased then 1 else 0 end         as is_deceased,
        p.race,
        p.ethnicity,
        p.state,

        -- Condition burden
        coalesce(c.has_diabetes, 0)               as has_diabetes,
        coalesce(c.has_hypertension, 0)           as has_hypertension,
        coalesce(c.has_coronary_artery_disease, 0) as has_cad,
        coalesce(c.has_heart_failure, 0)          as has_heart_failure,
        coalesce(c.has_asthma_or_copd, 0)         as has_asthma_or_copd,
        coalesce(c.has_ckd, 0)                    as has_ckd,
        coalesce(c.has_depression, 0)             as has_depression,
        coalesce(c.has_anxiety, 0)                as has_anxiety,
        coalesce(c.has_cancer, 0)                 as has_cancer,
        coalesce(c.total_active_conditions, 0)    as total_active_conditions,

        -- Vitals
        v.bmi,
        v.systolic_bp,
        v.diastolic_bp,
        v.cholesterol_total,
        coalesce(v.is_obese, 0)                   as is_obese,
        coalesce(v.elevated_systolic, 0)          as elevated_systolic,
        coalesce(v.high_cholesterol, 0)           as high_cholesterol,

        -- Utilization
        coalesce(e.total_encounters, 0)           as total_encounters,
        coalesce(e.emergency_count, 0)            as emergency_count,
        coalesce(e.inpatient_count, 0)            as inpatient_count,
        coalesce(e.encounters_last_12m, 0)        as encounters_last_12m,
        coalesce(e.care_span_days, 0)             as care_span_days,
        coalesce(e.high_ed_utilizer, 0)           as high_ed_utilizer,

        -- Composite risk score (rule-based baseline for validation)
        (
            coalesce(c.has_diabetes, 0)
            + coalesce(c.has_hypertension, 0)
            + coalesce(c.has_coronary_artery_disease, 0)
            + coalesce(c.has_heart_failure, 0)
            + coalesce(c.has_ckd, 0)
            + coalesce(c.has_cancer, 0)
            + coalesce(v.is_obese, 0)
            + coalesce(v.elevated_systolic, 0)
            + coalesce(e.high_ed_utilizer, 0)
        )::integer                                as composite_risk_score,

        current_timestamp                         as feature_generated_at

    from patients p
    left join conditions c using (patient_id)
    left join vitals     v using (patient_id)
    left join encounters e using (patient_id)
)

select * from joined
