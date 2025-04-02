-- create_tables.sql

-- 1) Create table for esg_extraction_results.csv
CREATE TABLE esg_extraction_results (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    total_energy_consumption_in_production NUMERIC,
    energy_consumption_per_vehicle_production NUMERIC,
    total_water_usage NUMERIC,
    total_wastewater_volume_generated NUMERIC,
    water_recycling_and_reuse_rate NUMERIC,
    total_ghg_emissions NUMERIC,
    ghg_emissions_and_intensity_per_vehicle NUMERIC,
    total_manufacturing_waste_generation NUMERIC,
    waste_recycling_and_diversion_rate NUMERIC,
    battery_recycling_rate NUMERIC,
    employee_count NUMERIC,
    employee_turnover_rate NUMERIC,
    number_of_workplace_accidents NUMERIC,
    employee_injury_rate NUMERIC,
    average_training_hours_employee NUMERIC,
    training_investment_employee NUMERIC,
    workforce_gender_ratios TEXT,
    workforce_minority_ratios TEXT,
    number_of_corruption_incidents NUMERIC,
    anti_corruption_compliance_rate NUMERIC,
    number_of_anti_competitive_practices NUMERIC,
    monetary_value_of_fines_imposed NUMERIC,
    political_contributions_and_lobbying_expenditures NUMERIC,
    number_of_marketing_compliance_and_ethical_advertising_violation_incidents NUMERIC,
    sales_weighted_average_fuel_economy_emissions NUMERIC,
    zero_emission_and_alternative_fuel_vehicle_sales NUMERIC,
    narrative_on_sustainability_goals_and_actions TEXT,
    progress_updates_on_emission_reduction_targets TEXT,
    disclosure_on_renewable_energy_initiatives_and_resource_efficiency_practices TEXT,
    narrative_on_workforce_diversity_employee_well_being_and_safety TEXT,
    disclosure_on_community_engagement_and_social_impact_initiatives TEXT,
    narrative_on_governance_framework_and_board_diversity TEXT,
    disclosure_on_esg_risk_management_and_stakeholder_engagement TEXT,
    narrative_on_innovations_in_sustainable_technologies_and_product_design TEXT,
    disclosure_on_sustainable_supply_chain_management_practices TEXT,
    size TEXT,
    industry TEXT,
    company TEXT,
    year INT
);

-- 2) Create table for esg_scoring_results.csv
CREATE TABLE esg_scoring_results (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    total_energy_consumption_in_production NUMERIC,
    energy_consumption_per_vehicle_production NUMERIC,
    total_water_usage NUMERIC,
    total_wastewater_volume_generated NUMERIC,
    water_recycling_and_reuse_rate NUMERIC,
    total_ghg_emissions NUMERIC,
    ghg_emissions_and_intensity_per_vehicle NUMERIC,
    total_manufacturing_waste_generation NUMERIC,
    waste_recycling_and_diversion_rate NUMERIC,
    battery_recycling_rate NUMERIC,
    employee_turnover_rate NUMERIC,
    number_of_workplace_accidents NUMERIC,
    employee_injury_rate NUMERIC,
    average_training_hours_employee NUMERIC,
    training_investment_employee NUMERIC,
    workforce_gender_ratios TEXT,
    workforce_minority_ratios TEXT,
    number_of_corruption_incidents NUMERIC,
    anti_corruption_compliance_rate NUMERIC,
    number_of_anti_competitive_practices NUMERIC,
    monetary_value_of_fines_imposed NUMERIC,
    political_contributions_and_lobbying_expenditures NUMERIC,
    number_of_marketing_compliance_and_ethical_advertising_violation_incidents NUMERIC,
    sales_weighted_average_fuel_economy_emissions NUMERIC,
    zero_emission_and_alternative_fuel_vehicle_sales NUMERIC,
    narrative_on_sustainability_goals_and_actions TEXT,
    progress_updates_on_emission_reduction_targets TEXT,
    disclosure_on_renewable_energy_initiatives_and_resource_efficiency_practices TEXT,
    narrative_on_workforce_diversity_employee_well_being_and_safety TEXT,
    disclosure_on_community_engagement_and_social_impact_initiatives TEXT,
    narrative_on_governance_framework_and_board_diversity TEXT,
    disclosure_on_esg_risk_management_and_stakeholder_engagement TEXT,
    narrative_on_innovations_in_sustainable_technologies_and_product_design TEXT,
    disclosure_on_sustainable_supply_chain_management_practices TEXT,
    e_score NUMERIC,
    s_score NUMERIC,
    g_score NUMERIC,
    esg_score NUMERIC,
    industry TEXT,
    company TEXT,
    year INT
);

-- 3) Create table for sentiment_analysis_results.csv
CREATE TABLE sentiment_analysis_results (
    company TEXT,
    extracted_field TEXT,
    sentiment TEXT
);

-- 4) Create table for metrics.csv
CREATE TABLE metrics (
    metric TEXT,
    unit TEXT
);

-- 5) Import data from CSV files
COPY esg_extraction_results
FROM '/docker-entrypoint-initdb.d/esg_extraction_results.csv'
DELIMITER ','
CSV HEADER;

COPY esg_scoring_results
FROM '/docker-entrypoint-initdb.d/esg_scoring_results.csv'
DELIMITER ','
CSV HEADER;

COPY sentiment_analysis_results
FROM '/docker-entrypoint-initdb.d/sentiment_analysis_results.csv'
DELIMITER ','
CSV HEADER;

COPY metrics
FROM '/docker-entrypoint-initdb.d/metrics.csv'
DELIMITER ','
CSV HEADER;
