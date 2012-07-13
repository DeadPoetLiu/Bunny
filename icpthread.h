#ifndef ICPTHREAD_H
#define ICPTHREAD_H

#include <QThread>
#include "pointcloud.h"

class QPointCloud;

class GLWidget;

class QTextEdit;

class ICPThread : public QThread
{
    Q_OBJECT
public:
    ICPThread(QObject *parent = 0);
    ICPThread(QPointCloud *input, QPointCloud *test, QPointCloud *dummy,
              QTextEdit *console,
              QObject *parent = 0);

protected:
    void run();
    
signals:
    void error_minimized();
    void log(QString message);
    void dummy_transformed();
public slots:

private slots:
    void cleanUp();
private:
    QPointCloud *input;
    QPointCloud *test;
    QPointCloud *dummy;
    PointCloudPtr saved;
    PointCloudPtr sample;
    QTextEdit *console;
};

#endif // ICPTHREAD_H
