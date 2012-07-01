#ifndef ICP_H
#define ICP_H

#include "pointcloud.h"

extern "C"
{

typedef struct _transformation{
  float theta;
  float phi;
  float psi;
  float translation[3];
} Transformation;

void initial_alignment(PointCloudPtr input, PointCloudPtr test);
void find_neighbors(PointCloudPtr input, PointCloudPtr test,
                    uint *neighbors);
float align(PointCloudPtr input, PointCloudPtr test,
           Transformation &transformation);
float get_random_angle();
void calculate_translation(PointCloudPtr input, PointCloudPtr sample,
                           uint *neighbors, float *translation);
float error_function(PointCloudPtr input, PointCloudPtr sample, uint *neighbours,
                     float alpha, float gamma, float beta, float* translation);
void transform_cloud(PointCloudPtr cloud, float theta, float phi, float psi,
                     float* translation);
void calculate_rotation(PointCloudPtr input, PointCloudPtr test);
}

#endif // ICP_H
