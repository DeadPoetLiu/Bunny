#include "qpointcloud.h"
#include "pointcloud.h"
#include <GL/gl.h>
#include <QFile>
#include <QTextStream>
#include <QColor>
#include <QRgb>

QPointCloud::QPointCloud(QString pathToFile, QObject *parent) :
    QObject(parent)
{
    this->loadDataFromPCD(pathToFile);
}

QPointCloud::QPointCloud(QObject *parent) : QObject(parent)
{
    this->data = 0x0;
}

void QPointCloud::loadDataFromPCD(QString pathToFile)
{
    QFile file(pathToFile);
    if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QTextStream in(&file);
    QString word;

    while(!in.atEnd()) {
        in >> word;
        if(word == "POINTS") {
            uint count;
            in >> count;
            this->data = create_point_cloud(count);
        }

        if(word == "ascii") {
            for(uint i; i < this->data->point_count; i++) {
                in >> this->data->points[i][0];
                in >> this->data->points[i][1];
                in >> this->data->points[i][2];
                this->data->center[0] += this->data->points[i][0];
                this->data->center[1] += this->data->points[i][1];
                this->data->center[2] += this->data->points[i][2];
            }
            this->data->center[0] /= this->data->point_count;
            this->data->center[1] /= this->data->point_count;
            this->data->center[2] /= this->data->point_count;
        }
    }
}

void QPointCloud::draw(QColor color)
{
    if(!data)
        return;

    QRgb rgb = color.rgb();
    glColor3f(qRed(rgb), qGreen(rgb), qBlue(rgb));

    glBegin(GL_POINTS);
    {
        for(uint i = 0; i < data->point_count; i++)
        {
            glVertex3f(data->points[i][0],
                       data->points[i][1],
                       data->points[i][2]);
        }
    }
    glEnd();
}
