// Original Source: MIT License Copyright (c) 2016 Adrian Biagioli
#pragma once

#ifndef parabolic_pointer_hpp
#define parabolic_pointer_hpp

#include "util.hpp"
#include "linalg_util.hpp"
#include "solvers.hpp"
#include "GL_API.hpp"
#include "geometric.hpp"


// Parabolic motion equation, y = p0 + v0*t + 1/2at^2
inline float parabolic_curve(float p0, float v0, float a, float t) 
{ 
    return p0 + v0 * t + 0.5f * a * t * t; 
}

inline float parabolic_curve_derivative(float v0, float a, float t) 
{ 
    return v0 + a * t; 
}

inline float3 parabolic_curve(float3 p0, float3 v0, float3 a, float t)
{
    float3 ret;
    for (int x = 0; x < 3; x++) ret[x] = parabolic_curve(p0[x], v0[x], a[x], t);
    return ret;
}

inline float3 parabolic_curve_derivative(float3 v0, float3 a, float t)
{
    float3 ret;
    for (int x = 0; x < 3; x++) ret[x] = parabolic_curve_derivative(v0[x], a[x], t);
    return ret;
}

inline float3 project_onto_plane(const float3 & planeNormal, const float3 & vector)
{
    return vector - (dot(vector, planeNormal) * planeNormal);
}

inline bool linecast(float3 p1, float3 p2, float3 & hitPoint, Geometry & g)
{
    //float3 dir = normalize(p2 - p1);
    //Ray r(p1, dir);

    Ray r = between(p1, p2);

    float outT = 0.0f;
    float3 outNormal = {0, 0, 0};

    std::cout << "Test Ray: " << r << std::endl;

    // If hit
    if (intersect_ray_mesh(r, g, &outT, &outNormal))
    {
        hitPoint = r.calculate_position(outT);
        std::cout << "Out T" << outT << std::endl;
        std::cout << "Hit At: " << hitPoint << std::endl;
        std::cout << "Hit Normal: " << outNormal << std::endl;
        return true;
    }
    hitPoint = float3(0, 0, 0);
    return false;
}

// Sample a bunch of points along a parabolic curve until you hit the ground. At that point, cut off the parabola.
// p0: starting point of parabola
// v0: initial parabola velocity
// a: initial acceleration
// dist: distance between sample points
// points: number of sample points
// curve: List that will be populated by new points
inline bool compute_parabolic_curve(float3 p0, float3 v0, float3 a, float dist, int points, Geometry & g, std::vector<float3> & curve)
{
    curve.clear();
    curve.push_back(p0);

    float3 last = p0;
    float t = 0.0;

    for (int i = 0; i < points; i++)
    {
        t += dist / length(parabolic_curve_derivative(v0, a, t));
        float3 next = parabolic_curve(p0, v0, a, t);

        float3 castHit;
        bool cast = linecast(last, next, castHit, g);

        //std::cout << "Cast: " << cast << std::endl;
        
        /*
        // We successfully linecasted against the mesh, but we might not be near where the parabola and the linecast
        // so ensure there's a sufficiently small distance
        if (false)
        //if (cast && distance(castHit, next) <= 0.1)
        {
            //curve.push_back(castHit);
            //return cast;
        }
        else 
         */  
        if (cast && distance(castHit, next) <= 0.1)
        {
            std::cout << "Stopped @ I - " << i << std::endl;
            curve.push_back(castHit);
            return true;
        }
        else curve.push_back(next);
        last = next;
    }

    return false;
}

// Clamps the given velocity vector so that it can't be more than 45 degrees above the horizontal.
// This is done so that it is easier to leverage the maximum distance (at the 45 degree angle) of parabolic motion.
// Returns angle with reference to the XZ plane
float clamp_initial_velocity(float3 & velocity, float3 & velocity_normalized) 
{
    // Project the initial velocity onto the XZ plane.
    float3 velocity_fwd = project_onto_plane(float3(0, 1, 0), velocity);

    // Find the angle between the XZ plane and the velocity
    float angle = to_degrees(uangle(velocity_fwd, velocity)); 
    std::cout << "Clamped angle is: " << angle << std::endl;

    // Calculate positivity/negativity of the angle using the cross product
    // Below is "right" from controller's perspective (could also be left, but it doesn't matter for our purposes)
    float3 right = cross(float3(0, 1, 0), velocity_fwd);

    // If the cross product between forward and the velocity is in the same direction as right, then we are below the vertical
    if (dot(right, cross(velocity_fwd, velocity)) > 0)
    {
        angle *= -1.0;
    }

    // Clamp the angle if it is greater than 45 degrees
    if(angle > 45.0) 
    {
        velocity = slerp(velocity_fwd, velocity, 45.f / angle);
        velocity /= length(velocity);
        velocity_normalized = velocity;
        velocity *= length(float3(10)); // initial velocity...
        angle = 45.0;
    } 
    else
    {
        velocity_normalized = normalize(velocity);
    }

    std::cout << "Returned Angle is: " << angle << std::endl;
    return angle;
}

Geometry make_parabolic_geometry(const std::vector<float3> & points, const float3 fwd, const float uvoffset)
{
    Geometry g;
    g.vertices.resize(points.size() * 2);
    g.texCoords.resize(points.size() * 2);

    const float3 right = normalize(cross(fwd, float3(0, 1, 0)));

    const float3 thickness = float3(0.5);

    for (int x = 0; x < 11; x++)
    {
        g.vertices[2 * x] = points[x] - right * thickness;
        g.vertices[2 * x + 1] = points[x] + right * thickness;

        std::cout << "Vert 1: " <<  g.vertices[2 * x] << std::endl;
        std::cout << "Vert 2: " <<  g.vertices[2 * x + 1] << std::endl;

        float uvoffset_mod = uvoffset;
        if (x == points.size() - 1 && x > 1) 
        {
            float dist_last = length(points[x-2] - points[x-1]);
            float dist_cur = length(points[x] - points[x-1]);
            uvoffset_mod += 1 - dist_cur / dist_last;
        }

        g.texCoords[2 * x] = float2(0, x - uvoffset_mod);
        g.texCoords[2 * x + 1] = float2(1, x - uvoffset_mod);
    }

    std::vector<int> indices(2 * 3 * (g.vertices.size() - 2));

    for (int x = 0; x < g.vertices.size() / 2 - 1; x++)
    {
        int p1 = 2 * x;
        int p2 = 2 * x + 1;
        int p3 = 2 * x + 2;
        int p4 = 2 * x + 3;

        indices[12 * x + 0] = p1;
        indices[12 * x + 1] = p2;
        indices[12 * x + 2] = p3;

        indices[12 * x + 3] = p3;
        indices[12 * x + 4] = p2;
        indices[12 * x + 5] = p4;

        indices[12 * x + 6] = p3;
        indices[12 * x + 7] = p2;
        indices[12 * x + 8] = p1;

        indices[12 * x + 9] = p4;
        indices[12 * x + 10] = p2;
        indices[12 * x + 11] = p3;
    }

    std::cout << "Indices Size: " << indices.size() << std::endl;

    for (int i = 0; i < indices.size(); i+=3)
    {
         g.faces.emplace_back(indices[i + 0], indices[i + 1], indices[i + 2]);
         std::cout << g.faces.back() << std::endl;
    }

    g.compute_normals();
    g.compute_bounds();

    return g;
}

Geometry make_parabolic_pointer(float deltaTime, Geometry & g)
{
    // 1. Calculate Parabola Points
    float3 position = float3(0, 4, 0);

    float3 velocity = float3(0, 0, -1) * float3(10.0); // transform local to world 
    float3 velocity_normalized = normalize(velocity);;
    //float currentAngle = clamp_initial_velocity(velocity, velocity_normalized);
    //std::cout << "Current Angle: " << currentAngle << std::endl;

    float3 acceleration =  float3(0, 1, 0) * -9.8f;
    float pointSpacing = 1.0f;
    float pointCount = 32;

    std::vector<float3> points;

    auto r = compute_parabolic_curve(position, velocity, acceleration, pointSpacing, pointCount, g, points);

    std::cout << "Points" << std::endl;
    for (auto p : points) std::cout << p << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    float3 selectedPoint = points[points.size()-1];

    // Draw parabola (BEFORE the outside faces of the selection pad, to avoid depth issues)
    return make_parabolic_geometry(points, velocity, fmod(deltaTime, .01));
}

#endif // end parabolic_pointer_hpp
