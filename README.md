# KENYA-MATERNAL-CHILD-HEALTH
# Equity in Maternal and Child Health Services Access Across Kenyan Counties

## Problem Statement

Despite significant investments in healthcare infrastructure and policy reforms, Kenya continues to face alarming disparities in maternal and child health outcomes across its 47 counties. **Kenya's maternal mortality ratio of 342 deaths per 100,000 live births remains nearly twice the global average, while under-5 mortality rates vary dramatically from 24 deaths per 1,000 live births in Nairobi to over 94 deaths per 1,000 in Mandera County** - a four-fold difference that represents thousands of preventable deaths annually.

These disparities are not merely statistical variations but reflect **systematic inequities in healthcare access, quality, and utilization** that disproportionately affect rural, pastoralist, and economically disadvantaged communities. Current health planning and resource allocation decisions are often made without comprehensive, county-specific evidence of where the greatest needs exist, leading to:

- **Misaligned resource distribution**: Counties with the highest mortality rates may not receive proportional health investments
- **Ineffective intervention targeting**: One-size-fits-all approaches fail to address county-specific barriers to care
- **Persistent equity gaps**: Without data-driven identification of underserved populations, existing disparities continue to widen
- **Suboptimal SDG progress**: Kenya risks falling short of Sustainable Development Goal 3 targets for maternal and child health by 2030

**The critical gap is the lack of systematic, county-level analysis that identifies specific disparities, quantifies equity gaps, and provides actionable insights for targeted interventions.** While national-level statistics show overall progress, they mask the reality that millions of women and children in certain counties face health outcomes comparable to the world's poorest nations, even as others approach middle-income country standards.

This analysis addresses this evidence gap by leveraging the most recent comprehensive health data to map inequities, identify priority areas for intervention, and provide the analytical foundation for evidence-based health policy and resource allocation decisions that can save lives and reduce preventable suffering.

## Project Overview

This project analyzes disparities in maternal and child health outcomes and service utilization across Kenya's 47 counties, with a focus on identifying underserved populations and informing targeted interventions.

## Key Research Questions

1. Which counties have the greatest disparities in maternal mortality ratios and child mortality rates?
2. How do socioeconomic factors, education levels, and geographic location correlate with health service utilization?
3. What are the coverage gaps in essential services like skilled birth attendance, immunization, and antenatal care?
4. Which interventions show the strongest correlation with improved outcomes across different regions?

## Data Sources

### Primary Dataset
- **Kenya Demographic and Health Survey (DHS) 2022**
  - Coverage: 37,911 households with county-level data
  - Includes: immunization, child and maternal health, family planning, nutrition, and healthcare access
  - Source: [Global Health Data Exchange](https://ghdx.healthdata.org/) | [Kenya National Bureau of Statistics](https://knbs.or.ke/)

### Supplementary Datasets
- **WHO Health Indicators for Kenya (HDX)**
  - Contains: maternal and reproductive health indicators, hemoglobin levels, birth rates, anemia prevalence
  - Source: [Humanitarian Data Exchange](https://data.humdata.org/)
  
- **DHS Historical Data**
  - Multiple survey rounds for trend analysis
  - Source: [Kenya National Demographic and Health Data | HDX](https://data.humdata.org/dataset/kenya-demographic-and-health-survey)

## Key Performance Indicators

### Maternal Health Indicators
- Maternal mortality ratios by county
- Skilled birth attendance rates
- Antenatal care coverage (4+ visits)
- Postpartum care within 48 hours
- Family planning utilization

### Child Health Indicators
- Infant mortality rate (baseline: 32 deaths per 1,000 live births)
- Under-5 mortality rate (baseline: 41 deaths per 1,000 live births)
- Immunization coverage rates
- Childhood nutrition indicators (stunting, wasting, underweight)
- Treatment coverage for childhood illnesses (diarrhea, pneumonia, malaria)

## Analysis Framework

### 1. Descriptive Analysis
County-level mapping of health outcomes and service coverage

### 2. Disparity Analysis
Calculation of equity gaps using:
- Concentration indices
- Rate ratios
- Inequality measures

### 3. Correlation Analysis
Examining relationships between:
- Socioeconomic factors and health outcomes
- Education levels and service utilization
- Geographic factors and health access

### 4. Trend Analysis
Historical comparison using multiple DHS survey rounds

### 5. Geographic Analysis
Spatial clustering identification of poor-performing areas

## Technical Requirements

### Software
- **Statistical Analysis**: R, Python, STATA, or SPSS
- **GIS Mapping**: QGIS or ArcGIS
- **Data Visualization**: Tableau, Power BI, or programming libraries (matplotlib, ggplot2, plotly)

### Key Libraries (if using Python)
```python
# Python
pandas, numpy, scipy, geopandas, matplotlib, seaborn, plotly, folium

```

## Data Access

### Registration Required
- **DHS Datasets**: Free but require registration through [DHS Program website](https://dhsprogram.com/)
- **World Bank Microdata**: [Kenya DHS 2022](https://microdata.worldbank.org/index.php/catalog/5240)

### Open Access
- **WHO Indicators**: Available through [Humanitarian Data Exchange](https://data.humdata.org/)

## Project Structure

```
kenya-maternal-child-health/
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Cleaned and transformed data
│   └── external/           # Supplementary data sources
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_descriptive_analysis.ipynb
│   ├── 03_disparity_analysis.ipynb
│   ├── 04_correlation_analysis.ipynb
│   └── 05_geographic_analysis.ipynb
├── src/
│   ├── data_processing/    # Data cleaning scripts
│   ├── analysis/          # Analysis functions
│   └── visualization/     # Plotting functions
├── outputs/
│   ├── figures/           # Generated plots and maps
│   ├── tables/            # Summary statistics tables
│   └── reports/           # Final analysis reports
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Expected Outputs

### Visualizations
- County-level choropleth maps of key indicators
- Scatter plots showing correlations between variables
- Time series plots for trend analysis
- Equity gap visualizations

### Reports
- County ranking tables for maternal and child health outcomes
- Disparity analysis summary
- Policy recommendation briefs

## Expected Impact

### Policy Impact
- Identify priority counties for targeted health interventions
- Inform resource allocation decisions for maternal and child health programs
- Support Kenya's progress toward SDG 3 (Good Health and Well-being)

### Operational Impact
- Guide deployment of community health workers
- Inform mobile health clinic routing strategies
- Support health facility strengthening priorities

## Getting Started

1. **Register for DHS data access** at [dhsprogram.com](https://dhsprogram.com/)
2. **Download datasets** from the provided sources
3. **Set up development environment** with required software
4. **Clone project structure** and organize data files
5. **Begin with exploratory data analysis**

## Contributing

This project aims to provide actionable insights for:
- Kenya's Ministry of Health
- County governments
- International development partners
- Public health researchers

## License

This project is intended for public health research and policy development. Please ensure compliance with data use agreements from DHS and other data providers.

## Contact

For questions about this analysis framework or collaboration opportunities, please refer to the contributing guidelines or open an issue in the project repository.
