# PIAF ML
This project is conducted by the [Lab IA](https://www.etalab.gouv.fr/datasciences-et-intelligence-artificielle) at [Etalab](https://www.etalab.gouv.fr/).  
The aim of the Lab IA is to help the french administration to modernize its services by the use of modern AI techniques.  
Other Lab IA projects can be found at the [main GitHub repo](https://github.com/etalab-ia/). In particular, the repository for the PIAF annotator can be found [here](https://github.com/etalab/piaf)

#### -- Project Status: [Active]
## PIAF
PIAF is an Opensource project aiming at providing the community with an easily activable French Question Answering Solution. The first use case of the PIAF project will be 

The objective of this repository is to give tools for the following tasks: 
* **Prepare** the data for the knowlgedge base
* **Evaluate** the performances of the PIAF stack
* **Experiment** different approaches in a contained environment (before prod)


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
[Mettre le dessin architecture ici]

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
2. Set the required environment variables, see [Set environment
   variables](#set-environment-variables) below.
2. Make sure gcc, make and the Python C API header files are installed on your
   system
  - On ubuntu:
    - `sudo apt install gcc make python3-dev`
3. Install requirements. Two options are available based on your prefered configuration: 
* Using pip:
`pip install -r requirements.txt`
on Windows : `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

* Using conda
`conda env create --name envname --file=environment.yml`
5. Make sure Docker is properly installed on your computer

### Performance evaluation
The procedure to start the evaluation script is the following:
1. Prepare your knowledge base in the form of a json file formated with the squad format. More information regarding the format of the file can be found [here](https://etalab-ia.github.io/knowledge-base/piaf/howtos/Format_donnees_SQuAD.html)
2. Define a set of experiment parameters with the src/evaluation/config/retriever_reader_eval_squad.py
```python
parameters = {
    "k_retriever": [20,30],
    "k_title_retriever" : [10], # must be present, but only used when retriever_type == title_bm25
    "k_reader_per_candidate": [5],
    "k_reader_total": [3],
    "retriever_type": ["title"], # Can be bm25, sbert, dpr, title or title_bm25
    "squad_dataset": ["./data/evaluation-datasets/tiny.json"],
    "filter_level": [None],
    "preprocessing": [True],
    "boosting" : [1], #default to 1
    "split_by": ["word"],  # Can be "word", "sentence", or "passage"
    "split_length": [1000],
    "experiment_name": ["dev"]
}
```
3. Run:
```bash
python - m src.evaluation.retriever_reader.retriever_reader_eval_squad.py
```
4. If ES throws some errors, try re-running it again. Sometimes the docker image takes time to initialize.
5. Note that the results will be saved in ``results/`` in a csv form. Also, mlruns will create a record in `mlruns`

## Project folder structure

```
/piaf-ml/
├── clients # Client specific deployment code
├── logs # Here we will put our logs when we get to it :)
├── notebooks # Notebooks with reports on experimentations
├── results # Folder were all the results generated from evaluation scripts are stored
├── src
│   ├── data # Script related to data generation
│   ├── evaluation # Scripts related to pipeline performance evaluation
│   │   ├── config # Configuration files
│   │   ├── results_analysis
│   │   ├── retriever # Scripts for evaluating the retriever only
│   │   ├── retriever_reader # Scripts for evaluating the full pipeline
│   │   └── utils # Somes utils dedicated to performance evaluation
│   └── models # Scripts related to training models
└── test # Unit tests
```

## Set environment variables

Certain capabilities of this codebase (e.g., using a remote mlflow endpoint) need a set of environment variables to work properly. We use `python-dotenv` to read the contents of a `.env` file that sits at the root of the project. This file is not tracked by git for security reasons. Still, in order for everything to work properly, you need to create such a file in your local code, again, at the root of the project, such as `piaf-ml/.env`.

A template which describes the different environment variables is provided in `.env.template`. Copy it to `.env` and edit it to your needs.

#### Mlflow Specific Configutation

To be able to upload artifacts into mlflow, you need to be able to `ssh` into the designated artifact server via a `ssh` key. Also, you need a local `ssh` config that specifies an identity file for the artifact-server domain. Such as: 
```
Host your.mlflow.remotehost.adress
    User localhostusername
    IdentityFile ~/.ssh/your_private_key
```
This requirement is needed **when using `sftp`** as your artifact endpoint protocol. 

## How to deploy PIAF

### If you already published the docker images to https://hub.docker.com/

- Then go to the [datascience server](https://datascience.etalab.studio)
- *if not done yet,* do a git clone of https://github.com/deepset-ai/haystack, otherwise go to your haystack folder (ex: guillim/haystack)
- *if not done yet,* customise the docker-compose.yml fil to suit your environment variables and your configuration. Example file can be found [here](https://github.com/etalab-ia/piaf-ml/blob/master/src/util/docker-compose.yml)
Note: in this file, you specify the docker images you want for your ElasticSearch, and for PiafAgent
- run `docker-compose up` ✅

### How to publish the elasticsearch docker image
This step is the most difficult : from downloading the latest version of service-public.fr XML files, we will publish a docker image of an Elasticsearch container in which we already injected all the service-public texts.

This can be done on your laptop (preferably not on the production server as it pollutes the )

- *if not done yet,* git clone piaf-ml
- Launch the script "run-all.py" : it will download latest version of service-pubilc.fr XMLs from data.gouv.fr & store the .json into a folder in /results directory (to be verified for this location)
- *if not done yet*, git clone haystack. Now place the folder (with all the Jsons that you just created after running the run-all script) under /data/ in the haystack repo. This is the only part 
```md
CONTRIBUTING.md
Dockerfile-GPU     
MANIFEST.in        
annotation_tool    
docker-compose.yml      
haystack           
requirements.txt 
run_docker_gpu.sh
test
tutorials
Dockerfile
LICENSE
README.md
data
  v14  # here you should now see your JSONs
docs
models
rest_api           
setup.py           
```
- Now, run haystack by typing `docker-compose up`
- First, clean the old document on the index you want to update by doing a `curl -XDELETE  http://localhost:9200/document_elasticsearch` (if you forget to do this, you will add your document to the exisiting ones, making a BIG database lol)
- Connect to your haystack container by typing `docker container logs -f  haystack_haystack-api_1` (note that the container name can change, better verifying it by typing `docker container ls`)
- Then install ipython `pip install ipython`
- Then run ipython `ipyhton`
- Then run this [tutorial Turorial_inject_BM25](https://github.com/etalab-ia/piaf-ml/blob/master/src/util/Turorial_inject_BM25.py)
Verify you have the document indexed into ES going at this endpoint [ES indexes](http://localhost:9200/_cat/indices?v)
- Now exit the container 
- create the image of your new ES container by typing `docker commit 829ed24c0d1b guillim/spf_particulier:v15` but don't forget to replace `829ed24c0d1b` by the ID of the elasticsearch container you can have typing `docker container ls`
- push to docker hub: `docker push guillim/spf_particulier:v15` ✅

### How to publish the piafagent docker image
Follow README.md on the [PiafAgent repo](https://github.com/etalab-ia/piaf_agent)


## Contributing Lab IA Members 
**Team Contacts :** 
* [PA. Chevalier](https://github.com/pachevalier)
* [PE. Devineau](https://github.com/pedevineau)

**Past Members :**
* [G. Lancrenon](https://github.com/guillim)
* [R. Reynaud](https://github.com/rob192)
* [G. Santarsieri](https://github.com/giuliasantarsieri)
* [P. Soriano](https://github.com/psorianom)
* [J. Denes](https://github.com/jdenes)

## How to contribute to this project 
We love your input! We want to make contributing to this project as easy and transparent as possible : see our [contribution rules](https://github.com/etalab-ia/piaf-ml/blob/master/.github/contributing.md)

## Contact
* Feel free to contact the team at piaf@data.gouv.fr with any questions or if you are interested in contributing!
