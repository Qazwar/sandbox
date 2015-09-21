#ifndef geometry_H
#define geometry_H

#include "linear_algebra.hpp"
#include "math_util.hpp"
#include "geometric.hpp"

#include <vector>

namespace util
{

    struct Geometry
    {
        std::vector<math::float3> vertices;
        std::vector<math::float3> normals;
        std::vector<math::float4> colors;
        std::vector<math::float2> uv;
        std::vector<math::float3> tangents;
        std::vector<math::float3> bitangents;
        std::vector<math::uint3>  faces;
    };
    
    struct Model
    {
        Geometry g;
        math::Pose pose;
        math::Box<float, 3> bounds;
    };
    
    struct Mesh : public Model
    {
        Mesh()
        {
            
        }
        
        Mesh(Mesh && r)
        {
            
        }
        
        Mesh(const Mesh & r) = delete;
        
        Mesh & operator = (Mesh && r)
        {
            return *this;
        }
        
        Mesh & operator = (const Mesh & r) = delete;
        
        ~Mesh();
        
        void draw() const;
        void update() const;
    };
    
    Mesh make_mesh(Model & m)
    {
        return Mesh();
    }
    
    void compute_normals(Geometry & g)
    {
        g.normals.resize(g.vertices.size());
        
        for (auto & n : g.normals)
            n = math::float3(0,0,0);
        
        for (auto & tri : g.faces)
        {
            math::float3 n = math::cross(g.vertices[tri.y] - g.vertices[tri.x], g.vertices[tri.z] - g.vertices[tri.x]);
            g.normals[tri.x] += n;
            g.normals[tri.y] += n;
            g.normals[tri.z] += n;
        }
        
        for (auto & n : g.normals)
            n = math::normalize(n);
        
    }
    
    void compute_tangents(Geometry & g)
    {
        
    }
    
} // end namespace util

#endif // geometry_H
