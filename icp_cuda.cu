#include "pointcloud.h"
#include <cuda_runtime.h>
#include <float.h>

extern "C" {
__global__ void find_neighbor(PointCloudPtr input, PointCloudPtr test, uint *neighbors)
{
    uint j, min_idx = 0;
    float input_x, input_y, input_z;
    float test_x = test->points[threadIdx.x][0];
    float test_y = test->points[threadIdx.x][1];
    float test_z = test->points[threadIdx.x][2];
    float dist_x, dist_y, dist_z;
    float dist = 0, min_dist = FLT_MAX;

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
    neighbors[0] = 8;
}

void find_neighbors_with_cuda(PointCloudPtr input, PointCloudPtr test, uint *neighbors)
{
    PointCloudPtr d_test, d_input;
    uint *d_neighbors;
    cudaMalloc((void**)&d_test, sizeof(Point) * test->point_count);
    cudaMalloc((void**)&d_input, sizeof(Point) * input->point_count);
    cudaMalloc((void**)&d_neighbors, sizeof(uint) * test->point_count);
    cudaMemcpy(d_test, test, test->point_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, input->point_count, cudaMemcpyHostToDevice);
    find_neighbor<<<1,test->point_count>>>(d_input, d_test, d_neighbors);
    cudaMemcpy(neighbors, d_neighbors, test->point_count, cudaMemcpyDeviceToHost);
}
}
