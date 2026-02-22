//
// Created by hima on 2/22/26.
//

#ifndef CUDA_RAYCASTER_RAYCAST_CUH
#define CUDA_RAYCASTER_RAYCAST_CUH
__global__ void rayCastGpu(
    float* lines,
    float posX, float posY,
    float dirX, float dirY,
    float planeX, float planeY,
    int w, int* map,
    int mapWidth, int mapHeight,
    float height);
#endif //CUDA_RAYCASTER_RAYCAST_CUH