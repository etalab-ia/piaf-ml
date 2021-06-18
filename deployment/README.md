# Deployment

This is to automate the installation of piaf-ml on the datascience sandbox

Every deployer should:

* Have SSH access to the machine
* Belong to the group `piaf` or be a sudoer

## Usage

* Install [ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) on your machine
* With `make install`, you will install haystack for every clients configured in hosts.yml
    * This is (and must remain) idempotent, it can be relaunched any time you want.
    * If your ssh user is different from your machine user, override it with `PIAF_MACHINE_USER='guillim' make install`

## Configuration

in the `hosts.yml` file, you will configure every client in the sandbox.

## Todo

* [x] Have a folder for installing haystack instances, instead of `/home/guillim`
    * `/data/piaf-ml/{client}`
* [x] Have a group (suggestion name: `piaf-deployment`) of users allowed to deploy piaf-ml
    * [x] this group should be able to `systemctl reload nginx` --> %piaf ALL= NOPASSWD: /usr/bin/systemctl reload nginx
    * [x] the file `/etc/nginx/sites-available/piaf` should belong to the group `piaf`
* **OR**
    * Have all the users in the sudoers, but it's more dangerous
