#include "pointcloud.h"
#include <cuda_runtime.h>
#include <float.h>
#include <cuda.h>
#include "cuPrintf.cu"

extern "C" {

__global__ void find_neighbor(float* input, uint input_count, float* test, uint test_count, uint *neighbors)
{
    uint j, min_idx = 0;
    int i = threadIdx.x + (blockIdx.x * blockDim.x);

    float input_x, input_y, input_z;
    float test_x = test[i*3];
    float test_y = test[i*3+1];
    float test_z = test[i*3+2];
    float dist_x, dist_y, dist_z;
    float dist = 0, min_dist = FLT_MAX;

    for(j = 0; j < input_count; j++)
    {
        input_x = input[j*3];
        input_y = input[j*3+1];
        input_z = input[j*3+2];
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

    if(i < test_count)
        cuPrintf("%d, %d\n", i, min_idx);
        neighbors[i] = min_idx;
}

void find_neighbors_with_cuda(PointCloudPtr input, PointCloudPtr test, uint *neighbors)
{
    float *input_points, *test_points;
    float *d_test, *d_input;
    uint *d_neighbors, i;
    int threadsPerBlock = 256;
    int blocksPerGrid = (test->point_count + threadsPerBlock - 1) / threadsPerBlock;
    input_points = (float*)malloc(sizeof(float) * input->point_count * 3);
    test_points = (float*)malloc(sizeof(float) * test->point_count * 3);
    for(i = 0; i < test->point_count; i++) {
        test_points[i*3] = test->points[i][0];
        test_points[i*3+1] = test->points[i][1];
        test_points[i*3+2] = test->points[i][2];
    }
    for(i = 0; i < input->point_count; i++) {
        input_points[i*3] = input->points[i][0];
        input_points[i*3+1] = input->points[i][1];
        input_points[i*3+2] = input->points[i][2];
    }

    cudaMalloc((void**)&d_test, 3 * sizeof(float) * test->point_count);
    cudaMalloc((void**)&d_input, 3 * sizeof(float) * input->point_count);
    cudaMalloc((void**)&d_neighbors, sizeof(uint) * test->point_count);
    cudaMemcpy(d_test, test_points, 3 * sizeof(float) * test->point_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input->points, 3 * sizeof(float) * input->point_count, cudaMemcpyHostToDevice);
    find_neighbor<<<blocksPerGrid,threadsPerBlock>>>(d_input, input->point_count, d_test, test->point_count, d_neighbors);
    cudaMemcpy(neighbors, d_neighbors, sizeof(uint) * test->point_count, cudaMemcpyDeviceToHost);
    if(d_test)
        cudaFree(d_test);
    if(d_input)
        cudaFree(d_input);
    if(d_neighbors)
        cudaFree(d_neighbors);
}
}
