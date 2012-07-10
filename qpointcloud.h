#ifndef QPOINTCLOUD_H
#define QPOINTCLOUD_H

#include <QObject>
#include "pointcloud.h"

class QColor;
class QMutex;

class QPointCloud : public QObject
{
    Q_OBJECT
public:
    explicit QPointCloud(QObject *parent = 0);
    QPointCloud(QString pathToFile, QObject *parent = 0);
    void loadDataFromPCD(QString pathToFile);

    ~QPointCloud() {
        free_point_cloud(data);
    }
    void draw(QColor color);

    PointCloudPtr data;
signals:
    
public slots:

private:

};

#endif // QPOINTCLOUD_H
