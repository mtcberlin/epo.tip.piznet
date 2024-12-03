# Patent Knowledge Forum 2024 - Regional Patent Analysis with PATSTAT

This repository contains the Jupyter Notebook and associated resources for the **EPO - Patent Knowledge Forum 2024**. The focus is on demonstrating a simplified, guided workflow for patent analysis using the **Technology Intelligence Platform**.

## Objective
To empower patent professionals with minimal technical expertise to:
- Extract and analyze patent applicant and technology distributions across Germany.
- Visualize regional data dynamically at the NUTS Level 3 (Landkreise) using generative AI-supported tools.

## Key Features
- **SQL for PATSTAT**: Modular SQL queries to extract data on patent filings and grants.
- **Dynamic Mapping**: Integration of NUTS codes and CPC subclass titles for enhanced readability.
- **Visualization**: Interactive data exploration with **Pygwalker** and optional Geopandas mapping.

## Libraries Used
- `pandas`, `sqlalchemy`, `pygwalker`
- Custom modules from the EPO TIP platform (e.g., `PatstatClient`)
- `geopandas` for spatial data handling (optional)

## Getting Started
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3.	Run the Jupyter Notebook:
  ```bash
  jupyter notebook epo_pkf2024_piznet_final.ipynb
  ```
##Outputs
	•	SQL execution times for PATSTAT queries.
	•	Dynamic maps and charts of patent data visualized with Pygwalker.
	•	Insights into regional applicant and technology trends.

## Use Case
Regional rankings of technology and applicants to:
	•	Support funding decisions.
	•	Improve regional business networks and collaborations.

 ## Acknowledgments
This project is part of the EPO - Patent Knowledge Forum 2024 initiative, showcasing contributions from:
	•	Arne Krueger - Head of PIZnet.de eV & mtc.berlin
	•	Partners at PATLIB centers and patent information professionals.
