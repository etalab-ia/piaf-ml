README PIAF_ML
===

# PIAF ML
This project is conducted by the [Lab IA](https://www.etalab.gouv.fr/datasciences-et-intelligence-artificielle) at [Etalab](https://www.etalab.gouv.fr/).  
The aim of the Lab IA is to help the french administration to modernize its services by the use of modern AI techniques.  
Other Lab IA projects can be found at the [main GitHub repo](https://github.com/etalab-ia/). In particular, the repository for the PIAF annotator can be found [here](https://github.com/etalab/piaf)

#### -- Project Status: [Active]
## PIAF
PIAF is an Opensource project aiming at providing the community with an easily activable French Question Answering Solution. The first use case of the PIAF project will be 

The objective of this repository is to give tools for the following tasks: 
* Prepare data for the knowlgedge base
* Evaluate the performances of the PIAF stack 


### Methods Used
* Information Retrieval
* Language Modeling
* Question Answering
* Machine Learning
### Technologies 
* Python
* ElasticSearch
* Docker
* Haystack
* Transformers
## Project Description 
### Project architecture 
The PIAF solution is using the following architecture : 
[Mettre le dessin archiitecture ici]

The code for PIAF Agent and PIAF bot are hosted on the following repositories: 
* [PiafAgent](https://github.com/etalab-ia/piaf_agent)
* [PiafBot]()

The code for the Haystack librairy can be found on the Deepset.ai [repository](https://github.com/deepset-ai/haystack/)

### Treat data for the knowlgedge base
One of the goal of this repository is to generate the json files that compose the knowledge base. 

### Evaluate performances for the stack PIAF 
For now, the main use of this repo is for evaluation. The goal of the evaluation is to assess the performance of the PIAF configuration on a `test_dataset` for which the `fiches` to be retrieved in the `knowledge_base` are known. 

## Needs of this project [TODO]
- frontend developers
- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting
- etc. (be as specific as possible)
## Getting Started for development
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Install [haystack](https://github.com/deepset-ai/haystack/). There are two methods for installing Haystack : via pip or via cloning the github repository (we try to use the `master` version of haystack so installing from the github repo is preferred). 
4. Install requirements. Two options are available based on your prefered configuration: 
* Using pip:
`pip install -r requirements.txt`

* Using conda
`conda env create --name envname --file=environment.yml`
5. Make sure Docker is properly installed on your computer

### Performance evaluation
The procedure to start the evaluation script is the following:
1. Unzip the knowledge base from /data. The knowledge bases are stored in zip files called vXY where XY are two digits
2. Define a set of experiment parameters with the src/evaluation/eval_config/__init__.py
```python
parameters = {
    "k": [5], # number of results to get by ES
    "retriever_type": ["sparse"], # either dense (sbert) or sparse (bm25)
    "knowledge_base": ["./data/v11"], # knowledge base jsons
                     
    "test_dataset": ["./data/407_question-fiche_anonym.csv"] # test dataset QA,
    "weighted_precision": [True] # this is MAP, unweighted would be similar to precision ,
    "filter_level": [None] # to prefilter the searched docs by "thematique" or "sous-thematique"
}
```
3. Run:
```bash
python ./src/evaluation/retriever_25k_eval.py
```
4. If ES throws some errors, try re-running it again. Sometimes the docker image takes time to initialize.

## Project folder structure
```
/piaf-ml/
├── data
│   ├── dense_dicts
│   └── vXY # Your folder generated with Knowledge database
├── logs # Here we will put our logs when we get to it :)
├── notebooks # Notebooks with reports on experimentations
├── reports # Reports
├── results # Folder were all the results generated from evaluation scripts are stored
├── src
│   ├── data # Script related to data generation
│   │   └── notebooks # Notebooks for data generation 
│   ├── evaluation
│   │   └── eval_config # Configuration file
│   ├── models # Scripts related to training models
│   └── util # Random functions that could be accessed from multiple places
```

## How to deploy PIAF [TODO]

## Contributing Lab IA Members 
**Team Contacts :** 
* [G. Lancrenon](https://github.com/guillim)
* [R. Reynaud](https://github.com/rob192)
* [G. Santarsieri](https://github.com/giuliasantarsieri)
* [P. Soriano](https://github.com/psorianom)
## Contact
* Feel free to contact the team at piaf@data.gouv.fr with any questions or if you are interested in contributing!
