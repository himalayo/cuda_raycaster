//
// Created by hima on 2/22/26.
//

__global__ void rayCastGpu(
    float* lines,
    const float posX, const float posY,
    const float dirX, const float dirY,
    const float planeX, const float planeY,
    const int w, int* map,
    const int mapWidth, const int mapHeight,
    float height)
{
    __shared__ int result;
    if (threadIdx.x == 0) result = 2147483647;
    __syncthreads();

    int x = blockIdx.x;
    int i = threadIdx.x;

    if (i == 0) return; // thread 0 unused (steps are 1-indexed)

    float cameraX = 2.0 * (float)x / (float)w - 1.0;
    float rayDirX = dirX + planeX * cameraX;
    float rayDirY = dirY + planeY * cameraX;

    float dX = (rayDirX == 0.0f) ? 1e30 : (float)abs(1.0 / rayDirX);
    float dY = (rayDirY == 0.0f) ? 1e30 : (float)abs(1.0 / rayDirY);

    int stepX = rayDirX < 0.0f ? -1 : 1;
    int stepY = rayDirY < 0.0f ? -1 : 1;

    int mX0 = (int)posX, mY0 = (int)posY;
    float sdX0 = rayDirX < 0.0f ? (posX - (float)mX0)*dX : ((float)mX0 + 1.0f - posX)*dX;
    float sdY0 = rayDirY < 0.0f ? (posY - (float)mY0)*dY : ((float)mY0 + 1.0f - posY)*dY;

    // Closed-form O(1) computation of map cell at step i
    int xSteps, ySteps;
    if (rayDirY == 0.0) {
        xSteps = i; ySteps = 0;
    } else if (rayDirX == 0.0) {
        xSteps = 0; ySteps = i;
    } else {
        float num = (sdY0 - sdX0) + (float)(i - 1) * dY;
        float exact = num / (dX + dY);
        float fl = floor(exact);
        xSteps = (exact - fl < 1e-9) ? (int)fl : (int)fl + 1;
        xSteps = max(0, min(i, xSteps));
        ySteps = i - xSteps;
    }

    int mapX = mX0 + xSteps * stepX;
    int mapY = mY0 + ySteps * stepY;
    float sdX = sdX0 + (float)xSteps * dX;
    float sdY = sdY0 + (float)ySteps * dY;

    int hit = 2147483647;
    if (mapX < 0 || mapX >= mapWidth || mapY < 0 || mapY >= mapHeight) {
        hit = i;
    } else if (map[mapX * mapHeight + mapY] > 0) {
        hit = i;
    }

    atomicMin(&result, hit);
    __syncthreads();

    if (result-1 == i) {
        float wallDist = (sdX < sdY) ? sdX : sdY;
        if (wallDist <= 0.0f) wallDist = 0.001f;
        lines[x] = height / wallDist;
    }
}
