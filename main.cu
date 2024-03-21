#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "glad/glad.h"
#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <cassert>

using namespace std;

enum class Color
{
    RED = 0,
    GREEN = 1,
    BLUE = 2,
    YELLOW = 3,
    MAGENTA = 4,
    CYAN = 5,
    MAX,
};

struct Pixel
{
    unsigned char r, g, b, a{255};

    Pixel() : r{0}, g{0}, b{0} {}
    Pixel(unsigned char r, unsigned char g, unsigned char b) : r{r}, g{g}, b{b} {}
    Pixel(unsigned char r, unsigned char g, unsigned char b, unsigned char a) : r{r}, g{g}, b{b}, a{a} {}
    Pixel(Color color)
    {
        switch (color)
        {
        case Color::RED:
            r = 255;
            g = 0;
            b = 0;
            break;
        case Color::GREEN:
            r = 0;
            g = 255;
            b = 0;
            break;
        case Color::BLUE:
            r = 0;
            g = 0;
            b = 255;
            break;
        case Color::YELLOW:
            r = 255;
            g = 255;
            b = 0;
            break;
        case Color::MAGENTA:
            r = 255;
            g = 0;
            b = 255;
            break;
        case Color::CYAN:
            r = 0;
            g = 255;
            b = 255;
            break;
        }
    }

    Pixel operator*(double val)
    {
        return Pixel(r * val, g * val, b * val, a * val);
    }
};

struct Particle
{
    double x{}, y{};
    double velX{}, velY{};
    Color color{};
};

__device__ double getForce(double distanceRatio, double attraction)
{
    const double repulsiveRadius{0.25};
    if (distanceRatio < repulsiveRadius)
    {
        return distanceRatio / repulsiveRadius - 1;
    }
    else
    {
        return attraction * (1 - distanceRatio);
    }
}

double const RADIUS{0.2};
size_t const colorCount{static_cast<size_t>(Color::MAX)};

__global__ void updatePosition(Particle *particles, size_t particleCount, double *attractionMatrix)
{
    double const FRICTION{pow(.5, 10)};

    unsigned idx{blockIdx.x * blockDim.x + threadIdx.x};

    Particle &me{particles[idx]};

    double forceX{}, forceY{};
    for (size_t i{}; i < particleCount; ++i)
    {
        if (i == idx)
            continue;
        Particle const &other{particles[i]};
        double x{other.x - me.x};
        double y{other.y - me.y};
        double dist{sqrt(x * x + y * y)};
        dist = max(dist, 0.0000000001); // Avoid division by zero

        if (dist < RADIUS)
        {
            double attraction{attractionMatrix[static_cast<int>(me.color) * colorCount + static_cast<int>(other.color)]};
            double force{getForce(dist / RADIUS, attraction)};

            forceX += (x / dist) * force;
            forceY += (y / dist) * force;
        }
    }

    me.velX += forceX;
    me.velY += forceY;
    me.velX *= FRICTION;
    me.velY *= FRICTION;

    me.x += me.velX;
    me.y += me.velY;
}

void draw(vector<Particle> const &particles, Pixel *image, int width, int height)
{
    for (size_t i{}; i < height; ++i)
    {
        for (size_t j{}; j < width; ++j)
        {
            image[i * width + j] = {0, 0, 0, 0};
        }
    }

    for (Particle const &particle : particles)
    {
        int imageX{particle.x * width};
        int imageY{particle.y * height};
        if (imageX > 0 && imageX < width && imageY > 0 && imageY < height)
            image[imageY * width + imageX] = Pixel{particle.color};
    }
}

int main(int argc, char *argv[])
{
    if (!glfwInit())
        return 1;

    GLFWwindow *window = glfwCreateWindow(1024, 1024, "Particle simulation", NULL, NULL);
    if (!window)
        return 1;

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        return 1;

    glfwSetWindowSizeCallback(window, [](GLFWwindow *win, int width, int height)
                              { glViewport(0, 0, width, height); });
    glViewport(0, 0, 1024, 1024);

    int seed{12};
    if (argc >= 2)
        seed = stoi(argv[1]);
    default_random_engine e{seed};

    // attraction matrix from x to
    uniform_real_distribution<double> uniform{-1, 1};
    size_t const colorCount{static_cast<size_t>(Color::MAX)};
    array<double, colorCount * colorCount> attractionMatrix{-0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5};
    for (size_t i{}; i < colorCount * colorCount; ++i)
        attractionMatrix[i] = uniform(e);
    double *devAttractionMatrix;
    cudaMalloc(&devAttractionMatrix, sizeof(double) * colorCount * colorCount);
    cudaMemcpy(devAttractionMatrix, attractionMatrix.data(), sizeof(double) * colorCount * colorCount, cudaMemcpyHostToDevice);

    // particles
    uniform = uniform_real_distribution<double>{0, 1};
    size_t particleCount{4 * 1024};
    if (argc >= 3)
        particleCount = stoi(argv[2]);
    assert(particleCount % 1024 == 0);
    vector<Particle> particles{particleCount};
    for (size_t i{}; i < particleCount; ++i)
    {
        particles[i].x = uniform(e);
        particles[i].y = uniform(e);
        particles[i].color = static_cast<Color>(i % colorCount);
    }
    Particle *devParticles;
    cudaMalloc(&devParticles, sizeof(Particle) * particleCount);

    cout << "Simulation: " << particleCount << " particles, " << colorCount << " colors, " << seed << " is the seed" << endl;
    cout << "Sampled attraction matrix:" << endl;
    for (size_t i{}; i < colorCount; ++i)
    {
        for (size_t j{}; j < colorCount; ++j)
        {
            cout << attractionMatrix[i * colorCount + j] << " ";
        }
        cout << endl;
    }

    int width{0}, height{0};
    Pixel *image = new Pixel[1];
    unsigned frames = 0;
    double prevTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        int prevWidth{width}, prevHeight{height};
        glfwGetFramebufferSize(window, &width, &height);
        if (width != prevWidth || height != prevHeight)
        {
            delete image;
            image = new Pixel[width * height];
        }

        cudaMemcpy(devParticles, particles.data(), sizeof(Particle) * particleCount, cudaMemcpyHostToDevice);

        dim3 blockDim{1024};
        dim3 gridDim{particleCount / 1024};
        updatePosition<<<gridDim, blockDim>>>(devParticles, particleCount, devAttractionMatrix);
        cudaDeviceSynchronize();
        cudaMemcpy(particles.data(), devParticles, sizeof(Particle) * particleCount, cudaMemcpyDeviceToHost);

        draw(particles, image, width, height);

        // actually draw
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glfwSwapBuffers(window);

        // fps
        frames++;
        double time{glfwGetTime()};
        double delta{time - prevTime};
        if (delta >= 1.)
        {
            double fps = frames / delta;
            glfwSetWindowTitle(window, ("Particle simulation, FPS: " + to_string(fps)).c_str());
            prevTime = time;
            frames = 0;
        }
    }

    // TODO: Use RAII
    glfwDestroyWindow(window);
    glfwTerminate();
}
