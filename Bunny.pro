# -------------------------------------------------
# Project created by QtCreator 2012-06-21T15:24:19
# -------------------------------------------------
QT += core \
    gui \
    opengl
greaterThan(QT_MAJOR_VERSION, 4):QT += widgets
TARGET = Bunny
TEMPLATE = app
SOURCES += main.cpp \
    window.cpp \
    glwidget.cpp \
    qpointcloud.cpp \
    pointcloud.cpp \
    icpthread.cpp \
    icp.cpp
HEADERS += window.h \
    glwidget.h \
    qpointcloud.h \
    pointcloud.h \
    icpthread.h \
    icp.h
FORMS += 
CUDA_SOURCES += icp_cuda.cu
RESOURCES += bunnies.qrc

# Project dir and outputs
PROJECT_DIR = $$system(pwd)
OBJECTS_DIR = $$PROJECT_DIR/Obj
DESTDIR = ../bin

# Path to cuda SDK install
CUDA_SDK = /home/kinect/NVIDIA_GPU_Computing_SDK/C

# Path to cuda toolkit install
CUDA_DIR = /usr/local/cuda

# GPU architecture
CUDA_ARCH = sm_11

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options \
    -fno-strict-aliasing \
    -use_fast_math \
    --ptxas-options=-v

# include paths
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$CUDA_SDK/common/inc/
INCLUDEPATH += $$CUDA_SDK/../shared/inc/

# lib dirs
QMAKE_LIBDIR += $$CUDA_DIR/lib64
QMAKE_LIBDIR += $$CUDA_SDK/lib
QMAKE_LIBDIR += $$CUDA_SDK/common/lib

# libs - note than i'm using a x_86_64 machine
LIBS += -L/usr/local/cuda/lib -lcudart

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$CUDA_DIR/bin/nvcc \
    -m32 \
    -g \
    -G \
    -arch=$$CUDA_ARCH \
    -c \
    $$NVCCFLAGS \
    $$CUDA_INC \
    $$LIBS \
    ${QMAKE_FILE_NAME} \
    -o \
    ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc \
    -g \
    -G \
    -M \
    $$CUDA_INC \
    $$NVCCFLAGS \
    ${QMAKE_FILE_NAME}

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda
OTHER_FILES += icp_cuda.cu
