#include "icpthread.h"
#include "qpointcloud.h"
#include "icp.h"
#include <QTextEdit>

ICPThread::ICPThread(QObject *parent) : QThread(parent)
{

}

ICPThread::ICPThread(QPointCloud *input, QPointCloud *test, QPointCloud *dummy,
                     QTextEdit *console,
                     QObject *parent) :
    QThread(parent)
{
    this->input = input;
    this->test = test;
    this->dummy = dummy;
    this->console = console;

    connect(this, SIGNAL(terminated()), this, SLOT(cleanUp()));
}

void ICPThread::run()
{
    Transformation trans;
    float error, min_error = 10;
    unsigned int counter = 0;
    const unsigned int limit = 100;
    const unsigned int sample_count = 10;
    saved = copy_point_cloud(this->test->data);
    dummy->data = copy_point_cloud(this->test->data);
    sample = create_sample_point_cloud(this->test->data, sample_count);
    initial_alignment(this->input->data, this->test->data);
    while(true) {
        error = align(this->input->data, sample, trans);
        transform_cloud(this->dummy->data, trans.theta, trans.phi, trans.psi,
                        trans.translation);
        emit dummy_transformed();
        if(error < min_error) {
            min_error = error;
            free_point_cloud(this->test->data);
            this->test->data = copy_point_cloud(dummy->data);
            emit log(QString("YEAH!! Minimized: %1").arg(error, 0, 'g', 8));
            emit error_minimized();
            counter = 0;
        } else if(counter == limit){
            counter = 0;
            free_point_cloud(sample);
            sample = create_sample_point_cloud(saved, sample_count);
            free_point_cloud(dummy->data);
            dummy->data = copy_point_cloud(saved);
            initial_alignment(this->input->data, dummy->data);
            emit log("I think this won't get any better! Reload");
        } else {
            counter++;
        }
    }
}

void ICPThread::cleanUp()
{
    free_point_cloud(sample);
    free_point_cloud(saved);
}
