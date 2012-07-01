#include "icpthread.h"
#include "qpointcloud.h"
#include "icp.h"
#include "glwidget.h"
#include <QTextEdit>

ICPThread::ICPThread(QObject *parent) : QThread(parent)
{

}

ICPThread::ICPThread(QPointCloud *input, QPointCloud *test, QPointCloud *sample,
                     GLWidget *widget,
                     QTextEdit *console,
                     QObject *parent) :
    QThread(parent)
{
    this->input = input;
    this->test = test;
    this->sample = sample;
    this->widget = widget;
    this->console = console;
}

void ICPThread::run()
{
    Transformation trans;
    float error, min_error = 10;
    unsigned int counter = 0;
    const unsigned int limit = 10;
    const unsigned int sample_count = 5;
    PointCloudPtr dummy = copy_point_cloud(this->test->data);
    PointCloudPtr saved = copy_point_cloud(this->test->data);
    free_point_cloud(this->sample->data);
    this->sample->data = create_sample_point_cloud(this->test->data, sample_count);
    emit new_sample();
    initial_alignment(this->input->data, dummy);
    while(true) {
        error = align(this->input->data, this->sample->data, trans);
        transform_cloud(dummy, trans.theta, trans.phi, trans.psi,
                        trans.translation);
        if(error < min_error) {
            min_error = error;
            free_point_cloud(this->test->data);
            this->test->data = copy_point_cloud(dummy);
            emit log(QString("YEAH!! Minimized: %1").arg(error, 0, 'g', 8));
            emit error_minimized();
        } else if(counter == limit){
            counter = 0;
            free_point_cloud(dummy);
            dummy = copy_point_cloud(saved);
            free_point_cloud(this->sample->data);
            this->sample->data = create_sample_point_cloud(this->test->data, sample_count);
            emit new_sample();
            initial_alignment(this->input->data, dummy);
            emit log("I think this won't getting better! Reload");
        } else {
            counter++;
        }
        emit error_minimized();
    }
}
