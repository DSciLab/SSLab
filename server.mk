PYTHON              := python
PIP                 := pip
LAB_PORT            := 10075
DASHBOARD_PORT      := 20075
HOST_IP             := 0.0.0.0
APP_DIR             := $(shell pwd)
HOSTNAME            := $(shell hostname)
SERVER_CONF         := ./conf/m4/server_conf.m4
SERVER_CONF_OUT     := ./conf/server_conf.yml


.PHONY: clean lab dashboard init_server install


install:
	$(PIP) install dist/*.tar.gz


init_server: $(SERVER_CONF)
	m4 -D_dashboard_port_=$(DASHBOARD_PORT) -D_hostname_=$(HOSTNAME) $^ > $(SERVER_CONF_OUT)


lab: init_server
	nohup jupyter lab --port $(LAB_PORT) --notebook-dir=$(APP_DIR) &


dashboard: init_server
	nohup $(PYTHON) -m visdom.server -port $(DASHBOARD_PORT) --hostname $(HOST_IP) &


clean:
	-rm -rf ./**/.ipynb_checkpoints dist