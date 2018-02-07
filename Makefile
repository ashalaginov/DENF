## -*- Makefile -*-
##
## User: andriis
## Time: Feb 7, 2018 10:17:10 PM
## Makefile created by Oracle Solaris Studio.
##
## This file is generated automatically.
##


#### Compiler and tool definitions shared by all build targets #####
CCC = g++
CXX = g++
BASICOPTS = -m64 -fopenmp 
CCFLAGS = $(BASICOPTS)
CXXFLAGS = $(BASICOPTS)
CCADMIN = 


# Define the target directories.
TARGETDIR_denf=GNU-amd64-Linux


all: $(TARGETDIR_denf)/denf

## Target: denf
OBJS_denf =  \
	$(TARGETDIR_denf)/main.o
USERLIBS_denf = $(SYSLIBS_denf) 
DEPLIBS_denf =    
LDLIBS_denf = $(USERLIBS_denf)


# Link or archive
$(TARGETDIR_denf)/denf: $(TARGETDIR_denf) $(OBJS_denf) $(DEPLIBS_denf)
	$(LINK.cc) $(CCFLAGS_denf) $(CPPFLAGS_denf) -o $@ $(OBJS_denf) $(LDLIBS_denf)


# Compile source files into .o files
$(TARGETDIR_denf)/main.o: $(TARGETDIR_denf) main.cpp
	$(COMPILE.cc) $(CCFLAGS_denf) $(CPPFLAGS_denf) -o $@ main.cpp



#### Clean target deletes all generated files ####
clean:
	rm -f \
		$(TARGETDIR_denf)/denf \
		$(TARGETDIR_denf)/main.o
	$(CCADMIN)
	rm -f -r $(TARGETDIR_denf)


# Create the target directory (if needed)
$(TARGETDIR_denf):
	mkdir -p $(TARGETDIR_denf)


# Enable dependency checking
.KEEP_STATE:
.KEEP_STATE_FILE:.make.state.GNU-amd64-Linux

