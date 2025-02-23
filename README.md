## MycoPredict
 Predict anti-mycobacterial activity of molecules based off of SMILES strings

# Overview
MycoPredict is a target-blind machine learning model designed to predict the anti-mycobacterial activity of small molecules based on their SMILES (Simplified Molecular Input Line Entry System) representations. Inspired by the BROAD Institute's phenotypic screening approach, this model leverages publicly available datasets to identify potential anti-tubercular compounds.

This project was developed to showcase my skills in cheminformatics, machine learning, and Python programming, as well as my passion for tackling global health challenges like tuberculosis (TB) drug resistance.

# key features
- Input: SMILES strings of small molecules.
- Output: Predicted anti-mycobacterial activity (e.g., active/inactive or probability score).
- Approach: Target-blind phenotypic screening using machine learning.
- Tools: Python, RDKit, scikit-learn, pandas, and matplotlib.

# Methodology
1. Data Collection:
    Curated datasets from public sources (e.g., ChEMBL, PubChem) containing SMILES strings and corresponding anti-mycobacterial activity labels.
2. Feature Engineering:
    Molecular descriptors and fingerprints generated using RDKit.
3. Model Training:
    Built and evaluated machine learning models (e.g., Random Forest, Gradient Boosting) using scikit-learn.
4. Validation:
    Preliminary cross-validation results are available, but further external validation is ongoing.

# repository structure
- Notebooks: used to keep track of code. While this is likely not ideal, I have not had the time to re-do this.
- scripts: scripts used to perform the methodology, in numbered order.
- Results: contain the model checkpoints, metrics & params
- training_data & validation_data: contain datasets used for training and validation

# Limitations
- Model Validation: The model is still in the development phase, and external validation is ongoing.
- Dataset Size: Predictive performance may be limited by the size and diversity of the training data.
- Target-Blind Nature: While useful for phenotypic screening, the model does not provide mechanistic insights into drug-target interactions.

# Future Work
- Building in ensemble methods to combine predictions from multiple models such as gradient boosting, random forest
- Building in transfer and active learning to improve model performance
- Perform external validation using independent datasets.
- Develop a user-friendly web interface for predicting anti-mycobacterial activity.

# About me
I am a PhD student and researcher with a background in medicinal chemistry and microbiology, currently working on combating Mycobacterium tuberculosis drug resistance. This project reflects my passion for computational drug discovery and my commitment to continuous learning.

Connect with me on (LinkedIn)[https://www.linkedin.com/in/panagiotis-tsampanis-8680ba204/] or explore my other projects on (GitHub)[https://github.com/Pana-TsK].

# License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code for academic or personal purposes.

# Acknowledgments
Inspired by the BROAD Institute's work on phenotypic screening.
Built using open-source tools like RDKit and scikit-learn.
Special thanks to the ChEMBL and PubChem teams for providing publicly accessible datasets.




