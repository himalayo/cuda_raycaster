#include "SDL2/SDL.h"
#include "./vendored/SDL_ttf/SDL_ttf.h"
#include <cstdio>
#include <map>

#define DEFAULT_PTSIZE  18

class RayCastingContext {
public:
    SDL_Window *window;
    SDL_Renderer *renderer;
    RayCastingContext(const int width, const int height) {
        window = SDL_CreateWindow("SDL Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, 0);
        renderer = SDL_CreateRenderer(window, -1, 0);
        load_font();
    }

    void DrawLine(const int x1, const int y1, const int x2, const int y2) const {
        SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
    }

    void DrawText(const char* string, const int x, const int y) const {
        const auto text = TTF_RenderText_Solid(font, string, white);
        if (text != nullptr)
        {
            const auto texture = SDL_CreateTextureFromSurface(renderer, text);
            const auto dst = SDL_Rect{x, y, text->w, text->h};
            SDL_FreeSurface(text);
            SDL_RenderCopy(renderer, texture, nullptr,&dst);
        }
    }

    void clear() const {
        SDL_SetRenderDrawColor(renderer, 51,51,51,SDL_ALPHA_OPAQUE);
        SDL_RenderClear(renderer);
        SDL_SetRenderDrawColor(renderer, 0,255,0,SDL_ALPHA_OPAQUE);
    }

    void show() const {
        SDL_RenderPresent(renderer);
    }

    int height() {
        SDL_GetWindowSize(window, &w, &h);
        return h;
    }

    int width() {
        SDL_GetWindowSize(window, &w, &h);
        return w;
    }
private:
    int h = 0;
    int w = 0;
    TTF_Font* font;

    void load_font()
    {
        /* Open the font */
        if (TTF_Init() < 0) {
            SDL_Log("Couldn't initialize TTF: %s\n",SDL_GetError());
            return;
        }
        font = TTF_OpenFont("sans.ttf", 18);
    }

    SDL_Color white = { 0xFF, 0xFF, 0xFF, 0 };
};

RayCastingContext* initialize(const int width, const int height) {
    if ( SDL_Init(SDL_INIT_EVERYTHING) != 0 ) {
        printf("error initializing SDL: %s\n", SDL_GetError());
        return nullptr;
    }

    auto *context = new RayCastingContext(width, height);
    return context;
}


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

float castRayCpu(const float posX, const float posY,
    const float dirX, const float dirY,
    const float planeX, const float planeY,
    const int x, const int w, int** map, const int mapWidth, const int mapHeight, float height)
{
    const float cameraX = 2.0f * static_cast<float>(x) / static_cast<float>(w) - 1;
    const float rayDirX = dirX + planeX * cameraX;
    const float rayDirY = dirY + planeY * cameraX;

    int mapX = static_cast<int>(posX);
    int mapY = static_cast<int>(posY);

    float deltaDistX = (rayDirX == 0) ? 1e20 : std::abs(1 / rayDirX);
    float deltaDistY = (rayDirY == 0) ? 1e20 : std::abs(1 / rayDirY);

    float sideDistX;
    float sideDistY;

    int stepX;
    int stepY;

    int hit = 0; //was there a wall hit?
    int side = 0; //was a NS or a EW wall hit?

    if(rayDirX < 0)
    {
        stepX = -1;
        sideDistX = (posX - static_cast<float>(mapX)) * deltaDistX;
    }
    else
    {
        stepX = 1;
        sideDistX = (static_cast<float>(mapX) + 1.0f - posX) * deltaDistX;
    }

    if(rayDirY < 0)
    {
        stepY = -1;
        sideDistY = (posY - static_cast<float>(mapY)) * deltaDistY;
    }
    else
    {
        stepY = 1;
        sideDistY = (static_cast<float>(mapY) + 1.0f - posY) * deltaDistY;
    }
    int max_ray_length = static_cast<int>(std::sqrt(std::pow(mapWidth,2)+std::pow(mapHeight, 2)))+1;

    for(int i = 0; i < max_ray_length; ++i)
    {
        //jump to next map square, either in x-direction, or in y-direction
        if(sideDistX < sideDistY)
        {
            sideDistX += deltaDistX;
            mapX += stepX;
            side = 0;
        }
        else
        {
            sideDistY += deltaDistY;
            mapY += stepY;
            side = 1;
        }

        if (mapX >= mapWidth || mapX < 0)
        {
            hit = i;
            break;
        }

        if (mapY >= mapHeight || mapY < 0)
        {
            hit = i;
            break;
        }

        //Check if ray has hit a wall
        if(map[mapX][mapY] > 0)
        {
            hit = i;
            break;
        }
    }

    if(side == 0) return height / (sideDistX - deltaDistX);
    return height / (sideDistY - deltaDistY);
}

void draw(RayCastingContext* context, const float* lines, size_t lines_len, const float* cpu_lines) {
    context->clear();
    float top_height = static_cast<float>(context->height())/4.0f;
    float bottom_height = 3.0f * static_cast<float>(context->height())/4.0f;
    for (int i = 0; i < lines_len; ++i) {
        float half_line = static_cast<float>(lines[i])/2.0f;
        float cpu_half_line = static_cast<float>(cpu_lines[i])/2.0f;
        context->DrawLine(i, static_cast<int>(top_height - half_line), i, static_cast<int>(half_line + top_height));
        context->DrawLine(i, static_cast<int>(bottom_height - cpu_half_line), i, static_cast<int>(bottom_height + cpu_half_line));
    }
    context->DrawText("Hi mom!", 10, 10);
    context->show();
}

int main()
{
    const auto context = initialize(800, 600);
    int worldWidth = 24, worldHeight = 24;
    float posX = 22, posY = 12;  //x and y start position
    float dirX = -1, dirY = 0; //initial direction vector
    float planeX = 0, planeY = 0.66; //the 2d raycaster version of camera plane

    double time = 0; //time of current frame
    double oldTime = 0; //time of previous frame
    int width = context->width();
    auto lines = static_cast<float *>(malloc(sizeof(float) * width));

    auto cpu_lines = static_cast<float *>(malloc(sizeof(float) * width));

    float* lines_gpu;
    cudaMalloc(&lines_gpu, sizeof(float) * width);
    int worldMap[24][24] =
    {
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 4, 0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 4, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    };

    const int* map[worldHeight];
    int* mapGpu;
    cudaMalloc(&mapGpu, sizeof(int)*worldWidth*worldHeight);
    for (int i = 0; i < worldHeight; i++) {
        map[i] = worldMap[i];
        cudaMemcpy(mapGpu+(i*worldWidth), map[i], worldWidth * sizeof(int), cudaMemcpyHostToDevice);
    }


    SDL_Event event;
    std::map<int, bool> keyboard;
    for (;;)
    {
        bool quit = false;
        while(SDL_PollEvent(&event))
        {
            switch (event.type)
            {
                case SDL_QUIT:
                    quit = true;
                    break;

                case SDL_KEYDOWN:
                    keyboard[event.key.keysym.sym] = true;
                    break;
                case SDL_KEYUP:
                    keyboard[event.key.keysym.sym] = false;
                    break;
                default: ;
            }
        }

        if (quit) break;
        oldTime = time;
        time = SDL_GetTicks();
        double frameTime = (time - oldTime) / 1000.0; //frameTime is the time this frame has taken, in seconds
        double moveSpeed = frameTime * 5.0; //the constant value is in squares/second
        double rotSpeed = frameTime * 3.0; //the constant value is in radians/second


        if (keyboard[SDLK_w]) {
            if(worldMap[int(posX + dirX * moveSpeed)][int(posY)] == false) posX += dirX * moveSpeed;
            if(worldMap[int(posX)][int(posY + dirY * moveSpeed)] == false) posY += dirY * moveSpeed;
        }

        if (keyboard[SDLK_s]) {
            if(worldMap[int(posX - dirX * moveSpeed)][int(posY)] == false) posX -= dirX * moveSpeed;
            if(worldMap[int(posX)][int(posY - dirY * moveSpeed)] == false) posY -= dirY * moveSpeed;
        }

        if (keyboard[SDLK_d]) {
            //both camera direction and camera plane must be rotated
            double oldDirX = dirX;
            dirX = dirX * cos(-rotSpeed) - dirY * sin(-rotSpeed);
            dirY = oldDirX * sin(-rotSpeed) + dirY * cos(-rotSpeed);
            double oldPlaneX = planeX;
            planeX = planeX * cos(-rotSpeed) - planeY * sin(-rotSpeed);
            planeY = oldPlaneX * sin(-rotSpeed) + planeY * cos(-rotSpeed);
        }

        if (keyboard[SDLK_a]) {
            //both camera direction and camera plane must be rotated
          double oldDirX = dirX;
          dirX = dirX * cos(rotSpeed) - dirY * sin(rotSpeed);
          dirY = oldDirX * sin(rotSpeed) + dirY * cos(rotSpeed);
          double oldPlaneX = planeX;
          planeX = planeX * cos(rotSpeed) - planeY * sin(rotSpeed);
          planeY = oldPlaneX * sin(rotSpeed) + planeY * cos(rotSpeed);
        }

        int height = context->height();

        int max_ray_length = (int)(sqrt(worldWidth*worldWidth + worldHeight*worldHeight)) + 1;
        rayCastGpu<<<width, max_ray_length + 1>>>(lines_gpu, posX, posY, dirX, dirY, planeX, planeY, width, mapGpu, worldWidth, worldHeight, height/2.0f);
        cudaDeviceSynchronize();
        cudaMemcpy(lines, lines_gpu, width*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < width; i++) {
            cpu_lines[i] = castRayCpu(posX, posY, dirX, dirY, planeX, planeY, i, width, (int**)map, worldWidth, worldHeight, height/2.0f);
        }
        draw(context, lines, width, cpu_lines);
    }

    return 0;
}