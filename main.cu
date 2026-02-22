#include "SDL2/SDL.h"
#include "./vendored/SDL_ttf/SDL_ttf.h"
#include <cstdio>
#include <map>
#include <format>
#include <thread>
#include <chrono>

#include "raycast.cuh"

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

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

    float deltaDistX = (rayDirX == 0) ? 1e20 : fabs(1 / rayDirX);
    float deltaDistY = (rayDirY == 0) ? 1e20 : fabs(1 / rayDirY);

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

void draw(RayCastingContext* context, const float* lines, size_t lines_len, const float* cpu_lines, double cuda_time, double cpu_time) {
    context->clear();
    float top_height = static_cast<float>(context->height())/4.0f;
    float bottom_height = 3.0f * static_cast<float>(context->height())/4.0f;
    for (int i = 0; i < lines_len; ++i) {
        float half_line = static_cast<float>(lines[i])/2.0f;
        float cpu_half_line = static_cast<float>(cpu_lines[i])/2.0f;
        context->DrawLine(i, static_cast<int>(top_height - half_line), i, static_cast<int>(half_line + top_height));
        context->DrawLine(i, static_cast<int>(bottom_height - cpu_half_line), i, static_cast<int>(bottom_height + cpu_half_line));
    }
    printf("CPU time: %f GPU Time: %f\n", cpu_time, cuda_time);
    context->show();
}

int main()
{
    const auto context = initialize(1920, 600);
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
        int threads = 32*(max_ray_length/32);
        double cuda_start = get_time();
        rayCastGpu<<<width, threads>>>(lines_gpu, posX, posY, dirX, dirY, planeX, planeY, width, mapGpu, worldWidth, worldHeight, height/2.0f);
        cudaDeviceSynchronize();
        cudaMemcpy(lines, lines_gpu, width*sizeof(float), cudaMemcpyDeviceToHost);
        double cuda_time = get_time() - cuda_start;
        double cpu_start = get_time();
        for (int i = 0; i < width; i++) {
            cpu_lines[i] = castRayCpu(posX, posY, dirX, dirY, planeX, planeY, i, width, (int**)map, worldWidth, worldHeight, height/2.0f);
        }
        double cpu_time = get_time() - cpu_start;

        draw(context, lines, width, cpu_lines, cuda_time, cpu_time);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    return 0;
}