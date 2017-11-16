
# -----------------------------------------------------------------------------
# Compiler tools

# Fortran
FC = gfortran
FDEBUG = -g -O0
FFLAGS = $(FDEBUG) -fPIC

# Fortran to Python
# use "f2py -c --help-fcompiler" to get available parameters
F2PY = f2py
F2PY_DEBUG = --debug --noopt --noarch
F2PY_FLAGS = $(F2PY_DEBUG) --verbose --f90flags="$(FFLAGS)"

# If python 3.X.X use f2py3 instead
ifeq ( $(findstring "ython 2.", $(shell python --version 2>&1)), )
	F2PY = f2py3
endif


# -----------------------------------------------------------------------------
# Sources and Directories
SRC_DIR = batman
# SRC = $(shell find $(SRC_DIR) -iname "*.f90")
SRC     = $(SRC_DIR)/input_output/tecplot.f90
MISC    = $(SRC_DIR)/input_output/io_tools.f90
MODULE  = _$(shell basename $(SRC) .f90)

OBJ_DIR = obj
OBJ := $(MISC:.f90=.o)


# -----------------------------------------------------------------------------
# Rules

all: directories $(OBJ)
	$(F2PY) $(F2PY_FLAGS) --build-dir $(OBJ_DIR) -c $(OBJ_DIR)/io_tools.o $(SRC) -m $(MODULE)
	mv $(MODULE)*.so `dirname $(SRC)`

%.o : %.f90
	@echo "Compiling $<"
	$(FC) $(FFLAGS) -c $< -o $(OBJ_DIR)/$(@F) -I$(OBJ_DIR)


.PHONY : directories clean
directories:
	@mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) *.mod

