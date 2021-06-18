ANSIBLE = ansible-playbook -i hosts.yml

ifndef PIAF_MACHINE_USER
	PIAF_MACHINE_USER:=$$USER
endif

install:
	@${ANSIBLE} -u ${PIAF_MACHINE_USER} install.yml

#setup:
#	@${ANSIBLE} setup.yml
