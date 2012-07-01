#include <QtGui>
#include <QtOpenGL>

#include <math.h>

#include "glwidget.h"

#include "icpthread.h"

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

GLWidget::GLWidget(QPointCloud *input, QPointCloud *test, QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    xRot = 0;
    yRot = 0;
    zRot = 0;

    sample = new QPointCloud();

    qtPurple = QColor::fromCmykF(0.39, 0.39, 0.0, 0.0);
    qtRed = QColor::fromRgb(255,0, 0);
    qtGreen = QColor::fromRgb(0, 255, 0);

    this->input = input;
    this->test = test;

    icp = new ICPThread(input, test, sample, this, this->console);

    connect(icp, SIGNAL(error_minimized()), this, SLOT(repaint()));
    connect(icp, SIGNAL(new_sample()), this, SLOT(repaint()));
}

GLWidget::~GLWidget()
{
}

QSize GLWidget::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize GLWidget::sizeHint() const
{
    return QSize(400, 400);
}

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

void GLWidget::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != xRot) {
        xRot = angle;
        emit xRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != yRot) {
        yRot = angle;
        emit yRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != zRot) {
        zRot = angle;
        emit zRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::initializeGL()
{
    glClearColor (0.0, 0.0, 0.0, 0.0);
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glRotatef(xRot / 16.0, 1.0, 0.0, 0.0);
    glRotatef(yRot / 16.0, 0.0, 1.0, 0.0);
    glRotatef(zRot / 16.0, 0.0, 0.0, 1.0);

    if(input)
        input->draw(qtRed);
    if(test)
        test->draw(qtGreen);
    if(sample)
        sample->draw(qtPurple);
}

void GLWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-0.15, 0.15, 0, 0.2, -0.2, 0.3);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        setXRotation(xRot + 8 * dy);
        setYRotation(yRot + 8 * dx);
    } else if (event->buttons() & Qt::RightButton) {
        setXRotation(xRot + 8 * dy);
        setZRotation(zRot + 8 * dx);
    }
    lastPos = event->pos();
}

void GLWidget::loadInputCloudFromPCD(QString pathToFile)
{
    input->loadDataFromPCD(pathToFile);
}

void GLWidget::loadTestCloudFromPCD(QString pathToFile)
{
    test->loadDataFromPCD(pathToFile);
}

void GLWidget::doICP()
{
    icp->start();
}

void GLWidget::setConsole(QTextEdit *console) {
    this->console = console;
    connect(icp, SIGNAL(log(QString)), this->console, SLOT(append(QString)));
}
