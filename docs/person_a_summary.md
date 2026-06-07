# Person A Summary

## Dataset Description Findings
Train samples: 2995
Test samples: 998
Detected text column: Text
Detected ID column: Unique ID
Label columns: 12 (format: multi-column)
Unique labels: 27
Average document length: 489.13 tokens (median: 213, min: 2, max: 3961)
Average character count: 3409.48
Vocabulary size (lowercased tokens): 30086
Missing values (train/test): 30022 / 0

## Label Imbalance Findings
Most common labels: 3.b.2 - Total net official development assistance to medical research and basic health sector, 3.8.1 - Coverage of essential health services (defined as the average coverage of essential services based on tracer interventions that include reproductive, maternal, newborn and child health, infectious diseases, non-communicable diseases and service capacity and access, among the general and the most disadvantaged population), 3.4.1 - Mortality rate attributed to cardiovascular disease, cancer, diabetes or chronic respiratory disease, 3.b.3 - Proportion of health facilities that have a core set of relevant essential medicines available and affordable on a sustainable basis, 3.3.1 - Number of new HIV infections per 1,000 uninfected population, by sex, age and key populations
Rarest labels: 3.9.3 - Mortality rate attributed to unintentional poisoning, 3.4.2 - Suicide mortality rate, 3.9.1 - Mortality rate attributed to household and ambient air pollution, 3.3.4 - Hepatitis B incidence per 100,000 population, 3.6.1 - Death rate due to road traffic injuries
Imbalance ratio (max/min): 33.55

## Text Characteristic Findings
Average sentence count: 20.54
Top domain terms (cleaned): research, health, project, clinical, development, study, application, support, program, service

## Co-occurrence Findings
Strongest co-occurring label pairs: 3.8.1 - Coverage of essential health services (defined as the average coverage of essential services based on tracer interventions that include reproductive, maternal, newborn and child health, infectious diseases, non-communicable diseases and service capacity and access, among the general and the most disadvantaged population) & 3.b.3 - Proportion of health facilities that have a core set of relevant essential medicines available and affordable on a sustainable basis (175); 3.7.1 - Proportion of women of reproductive age (aged 15_49 years) who have their need for family planning satisfied with modern methods & 3.7.2 - Adolescent birth rate (aged 10_14 years; aged 15_19 years) per 1,000 women in that age group (160); 3.b.2 - Total net official development assistance to medical research and basic health sector & 3.b.3 - Proportion of health facilities that have a core set of relevant essential medicines available and affordable on a sustainable basis (145); 3.8.1 - Coverage of essential health services (defined as the average coverage of essential services based on tracer interventions that include reproductive, maternal, newborn and child health, infectious diseases, non-communicable diseases and service capacity and access, among the general and the most disadvantaged population) & 3.8.2 - Proportion of population with large household expenditures on health as a share of total household expenditure or income (141); 3.4.1 - Mortality rate attributed to cardiovascular disease, cancer, diabetes or chronic respiratory disease & 3.b.2 - Total net official development assistance to medical research and basic health sector (140)

## Preprocessing Decisions
Lowercasing, HTML tag removal, punctuation and special character filtering, whitespace normalization via token re-join.
Tokenization with regex word boundaries.
Stopword removal using NLTK English list.
Lemmatization using WordNetLemmatizer.
Preserved domain acronyms (e.g., SDG, WHO, HIV, TB, USAID) in uppercase.
Numerical-only tokens removed.