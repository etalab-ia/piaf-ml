ANSIBLE = ansible-playbook -i hosts.yml

ifndef PIAF_MACHINE_USER
	PIAF_MACHINE_USER:=$$USER
endif

install:
	@${ANSIBLE} -u ${PIAF_MACHINE_USER} install.yml

wipe_and_insert_data:
ifndef PIAF_CLIENT
	@echo "Specify the PIAF_CLIENT you want to work on"
	exit 1
endif
	@${ANSIBLE} wipe_and_insert_data.yml --extra-vars "client=${PIAF_CLIENT}"
