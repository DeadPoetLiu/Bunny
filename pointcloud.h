#ifndef POINTCLOUD_H
#define POINTCLOUD_H

extern "C" {

typedef unsigned int uint;

// New type Point with coords x, y, z.
typedef float Point[3];

// An array of Point called Points
typedef Point* Points;

// Type PointCloud stores Points and the number of points.
typedef struct {
    Points points;
    uint point_count;
    Point center;
} PointCloud;

// I don't like stars ;)
typedef PointCloud* PointCloudPtr;

// New type for colors
typedef unsigned char Color[3];

// AAAAND some colors
const Color RED = {255, 0, 0};
const Color GREEN = {0, 255, 0};
const Color BLUE = {0, 0, 255};
const Color YELLOW= {0, 255, 255};

PointCloudPtr create_point_cloud(uint size);
PointCloudPtr create_sample_point_cloud(PointCloudPtr test, uint sample_count);
PointCloudPtr copy_point_cloud(PointCloudPtr cloud);
void free_point_cloud(PointCloudPtr cloud);
PointCloudPtr import_point_cloud_from_file(const char* filenname);
void rotate_point_cloud(PointCloudPtr cloud, float theta, float phi, float psi);
void translate_point_cloud(PointCloudPtr cloud, float translation[3]);

}
#endif // POINTCLOUD_H
