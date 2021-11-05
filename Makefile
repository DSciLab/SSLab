MAKE        := make
PIP         := pip
PYTHON      := python

CFG_PKG     := cfg
MLUTILS_PKG := mlutils
CVUTILS_PKG := cvutils

TARGET_PATH         := /home/cwj/Experiments/SSLab
SERVER_MKDIR_CMD    := mkdir -p $(TARGET_PATH)/dist \
                                $(TARGET_PATH)/conf
SERVER_RM_DIST_CMD  := rm -rf $(TARGET_PATH)/dist/*


.PHONY: install deploy dist clean deploy_conf


# ======= build packages ======
build: clean build_cfg build_mlutils build_cvutils
	$(PYTHON) setup.py sdist


build_cfg: $(CFG_PKG)
	$(MAKE) -C $^ build


build_mlutils: $(MLUTILS_PKG)
	$(MAKE) -C $^ build


build_cvutils: $(CVUTILS_PKG)
	$(MAKE) -C $^ build
# ======= build packages end ======


collect_dist: build
	cp $(CFG_PKG)/dist/* dist
	cp $(MLUTILS_PKG)/dist/* dist
	cp $(CVUTILS_PKG)/dist/* dist


copy_conf:
	mkdir -p $(TARGET_PATH)/conf/
	cp -r ./conf/* $(TARGET_PATH)/conf/


copy_makefile:
	cp server.mk $(TARGET_PATH)/Makefile


install: collect_dist	
	$(PIP) install dist/*.tar.gz


deploy: install copy_conf copy_makefile


deploy_conf: copy_conf copy_makefile


clean:
	-rm -rf dist .eggs .tox build MANIFEST *.egg*
