# Deployment

This is to automate the installation of piaf-ml on the datascience sandbox

## Prerequisites

Every deployer should:
* Have SSH access to the machine
* Belong to the group `piaf-deployment` or be a sudoer
* Have [ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) installed

## Usages

2 `make` target exists:

### `make install`

With `make install`, you will install haystack for every client configured in hosts.yml 
> This is (and must remain) idempotent.

### `PIAF_CLIENT=dila make reinstall`
With `PIAF_CLIENT=dila make reinstall`, you will install haystack just for this client. 
It will not change the nginx configuration, so it should **not** be used for a first installation
> This is (and must remain) idempotent.

### `PIAF_CLIENT=dila make wipe_and_insert_data`

With `PIAF_CLIENT=dila make wipe_and_insert_data`, you will wipe every data for this client and insert it back.
The PIAF_CLIENT is one of the clients that can be found in hosts.yml under the key `deployment`  
> This is NOT idempotent, it will wipe the data and reinsert it.


## Configuration

in the `hosts.yml` file, you will configure every client in the sandbox, and some useful parameters.

* `deployments:` the array of clients, with their given configuration
  * `haystack_port`: the tcp port the haystack will be mapped to
* `haystack_commit`: specify which haystack commit will be deployed (will be deployed on **all** clients)
* `haystack_python_dependencies`: list the extra python dependencies needed
* `installation_directory`: where the files should be put (you should almost never change it on a given host)
* `ansible_python_interpreter`: needed when both python2 and python3 are on the machine, to point on the python3 executable

If your ssh user is different from your machine user, override it with `PIAF_MACHINE_USER='guillim' make install`