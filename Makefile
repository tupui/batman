# F2PY parameters
# use "f2py -c --help-fcompiler" to get available parameters
ifeq ( $(findstring "Python 2.7.9", $(shell python --version 2>&1)), )
	F2PY   = f2py3 --verbose
else
	F2PY   = f2py --verbose
endif
F2PY_DEBUG = --debug --noopt --noarch
# F2PY_DEBUG += --debug-capi
FOPT       =
FFLAGS     = -O0
FC         = ifort -fPIC -g
FC         = gfortran -fPIC -g
#MOD_DIR = -module 
MOD_DIR = -I
#MOD_DIR = -J

###############################################################################
# process parameters

# f2py flags
F2PY_FLAGS = $(F2PY_NUM) $(F2PY_DEBUG) --opt="$(FOPT)" --f90flags="$(FFLAGS)"
SRC     = batman/input_output/tecplot.f90
MISC    = batman/input_output/io_tools.f90
MODULE  = _`basename $(SRC) .f90`

OBJ_DIR = obj
LIB_DIR = lib

# SRC = $(foreach dir,$(SRC_DIR),$(patsubst $(dir)/%.f90,%.f90, $(wildcard $(dir)/*.f90) ))
OBJ := $(MISC:.f90=.o)


all: directories $(OBJ)
	$(F2PY) $(F2PY_FLAGS) --build-dir $(OBJ_DIR) -c $(OBJ_DIR)/io_tools.o $(SRC) -m $(MODULE)
	mv $(MODULE)*.so `dirname $(SRC)`

.PHONY : directories
directories:
	@if [ ! -d $(OBJ_DIR) ] ; then mkdir -p $(OBJ_DIR) ; fi;
	@if [ ! -d $(LIB_DIR) ] ; then mkdir -p $(LIB_DIR) ; fi;

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) *.mod

%.o : %.f90
	@echo "Compiling $<" ; \
	$(FC) $(FFLAGS) -c $< -o $(OBJ_DIR)/`basename $@` $(MOD_DIR)$(OBJ_DIR)
