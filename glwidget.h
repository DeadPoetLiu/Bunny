#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QtOpenGL/QGLWidget>
#include "qpointcloud.h"

class ICPThread;

class QTextEdit;

class GLWidget : public QGLWidget
{
    Q_OBJECT

public:
    GLWidget(QPointCloud *input, QPointCloud *test, QWidget *parent = 0);
    ~GLWidget();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    void setConsole(QTextEdit *console);

public slots:
    void setXRotation(int anglcloude);
    void setYRotation(int angle);
    void setZRotation(int angle);
    void loadInputCloudFromPCD(QString pathToFile);
    void loadTestCloudFromPCD(QString pathToFile);
    void doICP(bool start);

signals:
    void xRotationChanged(int angle);
    void yRotationChanged(int angle);
    void zRotationChanged(int angle);

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);

private:
    int xRot;
    int yRot;
    int zRot;
    QPoint lastPos;
    QColor qtGreen;
    QColor qtRed;
    QColor qtPurple;
    QPointCloud *input;
    QPointCloud *test;
    QPointCloud *dummy;
    ICPThread *icp;
    QTextEdit *console;
};

#endif // GLWIDGET_H
