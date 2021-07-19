This folders is for running non-regression-tests in a scheduled manner.

Usage:

1. Copy the folder to the host that will run the tests.
2. Copy a test dataset locally.
2. Copy `.env.template` to `.env` and edit it as needed.
3. Create the file `known_hosts` and fill it with the mlflow server public key
   (use ssh-keyscan or connect once manually and copy the content of
   `~/.ssh/known_hosts`)
4. Run `docker-compose up -d` to start the container that will run the non
   regression tests everyday at 4am.
