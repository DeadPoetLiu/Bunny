#ifndef ICPTHREAD_H
#define ICPTHREAD_H

#include <QThread>

class QPointCloud;

class GLWidget;

class QTextEdit;

class ICPThread : public QThread
{
    Q_OBJECT
public:
    ICPThread(QObject *parent = 0);
    ICPThread(QPointCloud *input, QPointCloud *test, QPointCloud *sample,
              GLWidget *widget,
              QTextEdit *console,
              QObject *parent = 0);

protected:
    void run();
    
signals:
    void error_minimized();
    void log(QString message);
    void new_sample();
public slots:

private:
    QPointCloud *input;
    QPointCloud *test;
    QPointCloud *sample;
    GLWidget *widget;
    QTextEdit *console;
};

#endif // ICPTHREAD_H
