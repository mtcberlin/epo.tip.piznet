# EPO Technology Intelligence Platform
# at Patent Knowledge Forum 2024, 12/04/2024
# Regional Patent Analysis with PATSTAT under /piznet/
# Fun with Patent Classification unter /fwpc/

This repository under contains the Jupyter Notebook and associated resources for a short 10min presentaion at the **EPO - Patent Knowledge Forum 2024**. 

The focus was on demonstrating a simplified, guided workflow for patent analysis for regional applicant and technology using the newly launched **Technology Intelligence Platform**.

My name is Arne Krueger, and I was invited by the EPO at PATLIB 2024 in Ankara, to participate in a beta program prior to its launch. 

## Jupyter Notebook /piznet/epo_pkf2024_piznet_final.ipynb

### Objective
To empower and motivate patent information professionals with minimal technical expertise to:
- Extract and analyze patent applicant and technology distributions across Germany.
- Visualize regional data dynamically at the NUTS Level 3 (Landkreise) using generative AI-supported tools.

### Key Features
We are using the pre programmed SQL ORM by the EPO, but add other data for a richer experience and we want use python within this notebook to create an interactive map:
- **SQL for PATSTAT**: Modular SQL queries to extract data on patent filings and grants.
- **Dynamic Mapping**: Integration of NUTS codes and CPC subclass titles for enhanced readability.
- **Visualization**: Interactive data exploration with **Pygwalker** and optional Geopandas mapping.

### Libraries Used
- Custom modules from the EPO TIP platform (e.g., `PatstatClient`)
- `pandas`, `sqlalchemy`, `pygwalker`
- `geopandas` for spatial data handling (optional)
- system libraries for file and xml handling, a timer and other usefull extensions

### Getting Started
1. Get a Technology Intelligence Platform account:
   	at https://tip.epo.org
2. Get the Jupypter Notebook loaded into your TIP environment
    `epo_pkf2024_piznet_final.ipynb`
4. Add the optional folder as well
    `/images/` and the `/mappings/`
6. Open the Jupyter Notebook
	`epo_pkf2024_piznet_final.ipynb`
	
### Outputs
- SQL execution times for PATSTAT queries.
- `piznet/output/patent_data.csv` contains the extracted and mapped data
- Dynamic maps and charts of patent data visualized with Pygwalker. The configuration of the Charts are described in the notebook. 
- You get: Insights into regional applicant and technology trends.

### Use Case

Regional rankings of technology and applicants to:
- Support funding decisions.
- Improve regional business networks and collaborations.

### Acknowledgments
This project is part of the EPO - Patent Knowledge Forum 2024 initiative, showcasing contributions from:
- Arne Krueger - Head of PIZnet.de eV & mtc.berlin
- Carlos Perez, Head of Data Science at EPO
- Partners at PATLIB centers and patent information professionals.


## Jupyter Notebooks in /fwpc/ 

These notebooks were developed as part of learning python and enjoying the complexity and beaty of the international and cooperative patent classification system, human mankinds most extensive classification system ever invented, much bigger than any library, business, product or other classification system. 


