# ACStudy

This directory holds the source code needed for replicating the results that are reported in:

Talmor-Barkan Y and Bar N. et al. A multi-omic characterization reveals personalized risk factors for coronary artery disease. DOI: XXX

## Files:
- README.md - this file.
- helper_functions.py - includes python helper functions which are used in other notebooks.
- preprocess_metabolon_data.ipynb - preprocessing raw metabolon data, perform normalizations, batch correction and adjustment for storage times.
- additional_metabolon_files (folder) - includes several additional files used in some of the other notebooks.
- Matching.py - code used for matching.
- Matching ACS-PNP1.ipynb - performing the actual matching used in the paper.
- ACS-PNP table1.ipynb - computes and creates table 1 statistics.
- Data_flowchart.ipynb - detailing the data flowchart, as shown in extended data figure 1.
- ACS-PNP1_taxa_composition.ipynb - computing the taxa composition. Results are included in figure 1.
- ACS-PNP1_microbial_species_pattern.ipynb - compute the logistic regression for every microbial species. Results are included in figure 1.
- specific_microbial-metabolite_pathway.ipynb - analyzing microbial species SGB4712 and its metabolites. Results are included in figure 1. Creates extended data figure 3.
- SGB_4712 (folder) - incldues additional information on SGB4712. See https://github.com/noambar/ACStudy/blob/master/SGB_4712/README.md for details.
- serum_metabolomics_signatures.ipynb - creates the serum metabolomics signature. Results are included in figure 1. Creates figure 1. Creates extended data figure 2.
- individualized_metabolic_determinants.ipynb - the analysis of the individualized metabolic deviations. Results are included in figures 2 and 3. Creates figures 2 and 3. Creates extended data figure 4.
- metabolomics_model_of_bmi.ipynb - the analysis of the metabolomics-based model of BMI. Results are included in figure 4. Creates figure 4. Creates extended data figure 5.
- metacardis_bmi_prediction.ipynb - the analysis of the metabolomics-based model of BMI in the MetaCardis cohort. Results are included in extended data figure 6. Creates extended data figure 6.
- metacardis_SHAP.ipynb - the SHAP analysis in the metabolomics-based model of BMI in the MetaCardis cohort.

## Data:
- The raw metagenomic sequencing data per sample of our controls are available from the European Nucleotide Archive (ENA; https://www.ebi.ac.uk/ena): PRJEB11532. 
- The raw metabolomics data and phenotypes per sample of our controls are available from the European Genome-phenome Archive (EGA; https://ega-archive.org/): EGAS00001004512. 
- The raw metabolomics data and full clinical phenotypes for our cohort of individuals with ACS were deposited to the EGA: EGAS00001005342, and will be released upon publication.
- Additional data regarding SGB 4712, including the genome sequence, gene annotation, and closest references are available at https://github.com/noambar/ACStudy/tree/master/SGB_4712.
