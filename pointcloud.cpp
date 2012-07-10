#include "pointcloud.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

extern "C" {

PointCloudPtr copy_point_cloud(PointCloudPtr cloud) {
    uint _i;
    PointCloudPtr copy = create_point_cloud(cloud->point_count);

    for(_i = 0; _i < copy->point_count; _i++) {
        copy->points[_i][0] = cloud->points[_i][0];
        copy->points[_i][1] = cloud->points[_i][1];
        copy->points[_i][2] = cloud->points[_i][2];
    }
    copy->center[0] = cloud->center[0];
    copy->center[1] = cloud->center[1];
    copy->center[2] = cloud->center[2];

    return copy;
}


/**
 * Allocates the needed memory for PointCloud.
 */
PointCloudPtr create_point_cloud(uint size)
{
    PointCloudPtr new_cloud = (PointCloudPtr)malloc(sizeof(PointCloud));
    new_cloud->point_count = size;
    new_cloud->points = (Points)malloc(size * sizeof(Point));
    new_cloud->center[0] = 0;
    new_cloud->center[1] = 0;
    new_cloud->center[2] = 0;
    return new_cloud;
}

PointCloudPtr create_sample_point_cloud(PointCloudPtr test, uint sample_count)
{
    PointCloudPtr sample = create_point_cloud(sample_count);
    uint _i, index;

    srand(time(NULL));

    for(_i = 0; _i < sample_count; _i++) {
        index = rand() % test->point_count;
        sample->points[_i][0] = test->points[index][0];
        sample->points[_i][1] = test->points[index][1];
        sample->points[_i][2] = test->points[index][2];
    }

    sample->center[0] = test->center[0];
    sample->center[1] = test->center[1];
    sample->center[2] = test->center[2];
    return sample;
}

/**
 * Gives the allocated memory back.
 */
void free_point_cloud(PointCloudPtr cloud)
{
    if(cloud) {
        free(cloud->points);
        free(cloud);
    }
}

/**
 * Imports data from a pcd-file.
 */
PointCloudPtr import_point_cloud_from_file(const char* filename)
{
    FILE* fh;
    int ret;
    uint points, _i;
    char buffer[80], points_read = 0;
    PointCloudPtr imported_cloud;
    Point* iter;
    float cx, cy, cz;
    float x = 0, y = 0, z = 0;

    // Open the file for reading
    fh = fopen(filename, "r");

    // Did you open the file?
    if(!fh) { // No?!
        perror("Cannot open file");
        exit(1);
    }

    // Parse file while you at the end of the file or you read all points.
    while(ret != EOF && points_read == 0) {
        ret = fscanf(fh, "%s", buffer);
        if(strcmp(buffer, "POINTS") == 0) {
            // How many points to load?
            ret = fscanf(fh, "%d", &points);
        }
        // Reached the points?
        if(strcmp(buffer, "ascii") == 0) {
            // Create PointCloud
            imported_cloud = create_point_cloud(points);

            // iterator to store the points in the cloud
            iter = imported_cloud->points;
            for(_i = 0; _i < points; _i++) {
                ret = fscanf(fh, "%f", &cx);
                ret = fscanf(fh, "%f", &cy);
                ret = fscanf(fh, "%f", &cz);

                (*iter)[0] = cx;
                (*iter)[1] = cy;
                (*iter)[2] = cz;
                x += (*iter)[0];
                y += (*iter)[1];
                z += (*iter)[2];
                // Point to the next Point in the cloud.
                iter++;
            }
            imported_cloud->center[0] = x / points;
            imported_cloud->center[1] = y / points;
            imported_cloud->center[2] = z / points;
            points_read = 1;
        }
    }

    // Close file
    fclose(fh);

    return imported_cloud;
}

void rotate_point_cloud(PointCloudPtr cloud, float theta, float phi, float psi)
{
    uint _i;
    float *x, *y, *z;
    float dx = 0, dy = 0, dz = 0;
    float inverse_center[3] = {(-1) * cloud->center[0], -1 * cloud->center[1], -1 * cloud->center[2]};
    float stored_center[3] = {cloud->center[0], cloud->center[1], cloud->center[2]};
    // translate to 0,0,0
    translate_point_cloud(cloud, inverse_center);

    for(_i = 0; _i < cloud->point_count; _i++) {
        x = &(cloud->points[_i][0]);
        y = &(cloud->points[_i][1]);
        z = &(cloud->points[_i][2]);
        dx = (cos(theta) * cos(phi) * (*x)) + ((cos(theta) * sin(phi) * sin(psi) - sin(theta) * cos(psi)) * (*y)) + ((cos(theta) * sin(phi) * cos(psi) + sin(theta) * sin(psi)) * (*z));
        dy = (sin(theta) * cos(phi) * (*x)) + ((sin(theta) * sin(phi) * sin(psi) + cos(theta) * cos(psi)) * (*y)) + ((sin(theta) * sin(phi) * sin(psi) - cos(theta) * sin(psi)) * (*z));
        dz = -sin(phi) * (*x) + cos(phi) * sin(psi) * (*y) + cos(phi) * cos(psi) * (*z);
        *x = dx;
        *y = dy;
        *z = dz;
    }

    // translate back
    translate_point_cloud(cloud, stored_center);
}

void translate_point_cloud(PointCloudPtr cloud, float translation[3])
{
    uint _i;
    float *x, *y, *z;

    for(_i = 0; _i < cloud->point_count; _i++) {
        x = &(cloud->points[_i][0]);
        y = &(cloud->points[_i][1]);
        z = &(cloud->points[_i][2]);
        *x += translation[0];
        *y += translation[1];
        *z += translation[2];
    }
    cloud->center[0] += translation[0];
    cloud->center[1] += translation[1];
    cloud->center[2] += translation[2];
}

}
