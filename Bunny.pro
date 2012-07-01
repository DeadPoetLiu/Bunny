#-------------------------------------------------
#
# Project created by QtCreator 2012-06-21T15:24:19
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Bunny
TEMPLATE = app


SOURCES += main.cpp\
    window.cpp \
    glwidget.cpp \
    qpointcloud.cpp \
    pointcloud.cpp \
    icpthread.cpp \
    icp.cpp

HEADERS  += \
    window.h \
    glwidget.h \
    qpointcloud.h \
    pointcloud.h \
    icpthread.h \
    icp.h

FORMS    +=

RESOURCES += \
    bunnies.qrc
