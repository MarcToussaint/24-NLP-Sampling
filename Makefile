BASE = ../rai

OPTIM = debug

DEPEND = Core Optim Algo KOMO Kin Gui Geo Perception

OBJS = main.o DataMetrics.o problems.o experiment.o

EIGEN = 1
OPENCV4 = 1

#LIBS += -lmcmc

include $(BASE)/_make/generic.mk
