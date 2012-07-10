#include "icp.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

extern "C"
{

void initial_alignment(PointCloudPtr input, PointCloudPtr test)
{
    float translation[3];
    translation[0] = input->center[0] - test->center[0];
    translation[1] = input->center[1] - test->center[1];
    translation[2] = input->center[2] - test->center[2];
    translate_point_cloud(test, translation);
}

void find_neighbors(PointCloudPtr input, PointCloudPtr test, uint *neighbors)
{
    uint i, j, min_idx = 0;
    float test_x, test_y, test_z;
    float input_x, input_y, input_z;
    float dist_x, dist_y, dist_z;
    float dist = 0, min_dist = FLT_MAX;

    for(i = 0; i < test->point_count; i++)
    {
        test_x = test->points[i][0];
        test_y = test->points[i][1];
        test_z = test->points[i][2];
        min_dist = FLT_MAX;
        for(j = 0; j < input->point_count; j++)
        {
            input_x = input->points[j][0];
            input_y = input->points[j][1];
            input_z = input->points[j][2];
            dist_x = input_x - test_x;
            dist_y = input_y - test_y;
            dist_z = input_z - test_z;
            dist = sqrt(dist_x * dist_x +
                        dist_y * dist_y +
                        dist_z * dist_z);
            if(dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }
        neighbors[i] = min_idx;
    }
}

float align(PointCloudPtr input, PointCloudPtr test,
           Transformation &transformation)
{
    float error = 0, theta = 0, phi = 0, psi = 0, new_error = 0;
    float best_theta = 0, best_phi = 0, best_psi = 0;
    float translation[3] = {0, 0, 0};
    float best_translation[3] = {0, 0, 0};
    int i;
    uint *neighbors;
    PointCloudPtr sample = test;
    neighbors = (uint*)malloc(sample->point_count * sizeof(uint));
    srand(time(NULL));
    find_neighbors(input, sample, neighbors);
    error = 10;
    calculate_translation(input, sample, neighbors, translation);
    translate_point_cloud(sample, translation);
    for(i = 0; i < 100; i++) {
        theta = get_random_angle();
        phi = get_random_angle();
        psi = get_random_angle();
        new_error = error_function(input, sample, neighbors,
                                   theta, phi, psi, translation);
        if(new_error < error) {
            best_theta = theta;
            best_phi = phi;
            best_psi = psi;
            best_translation[0] = translation[0];
            best_translation[1] = translation[1];
            best_translation[2] = translation[2];
            error = new_error;
        }
    }
    transform_cloud(sample, best_theta, best_phi, best_phi, best_translation);
    find_neighbors(input, sample, neighbors);
    transformation.theta = best_theta;
    transformation.phi = best_phi;
    transformation.psi = best_psi;
    transformation.translation[0] = translation[0];
    transformation.translation[1] = translation[1];
    transformation.translation[2] = translation[2];
    return error;
}

float get_random_angle()
{
    float factor = 0;
    srand(time(NULL));
    if(rand() % 2 == 0)
        factor = -1;
    else
        factor = 1;
    return factor / (float)(rand() % 100);
}

void calculate_translation(PointCloudPtr input, PointCloudPtr sample,
                           uint *neighbors, float *translation)
{
    float x = 0, y = 0, z = 0;
    uint _i = 0;

    for(_i = 0; _i < sample->point_count; _i++) {
        x += (input->points[neighbors[_i]][0] - sample->points[_i][0]);
        y += (input->points[neighbors[_i]][1] - sample->points[_i][1]);
        z += (input->points[neighbors[_i]][2] - sample->points[_i][2]);
    }

    translation[0] = x / sample->point_count;
    translation[1] = y / sample->point_count;
    translation[2] = z / sample->point_count;
}

float error_function(PointCloudPtr input, PointCloudPtr sample, uint* neighbours,
                     float alpha, float gamma, float beta, float* translation)
{
    uint _i;
    float error = 0 , x, y, z;
    PointCloudPtr copy = copy_point_cloud(sample);

    transform_cloud(copy, alpha, beta, gamma, translation);

    for(_i = 0; _i < copy->point_count; _i++) {
        x = input->points[neighbours[_i]][0] - copy->points[_i][0];
        y = input->points[neighbours[_i]][1] - copy->points[_i][1];
        z = input->points[neighbours[_i]][2] - copy->points[_i][2];
        error += sqrt((x * x) + (y * y) + (z * z));
    }

    free_point_cloud(copy);
    return error;
}

void transform_cloud(PointCloudPtr cloud, float theta, float phi, float psi,
                     float* translation)
{
    translate_point_cloud(cloud, translation);
    rotate_point_cloud(cloud, theta, phi, psi);
}

}
