/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Metal shaders used for ray tracing
*/

#include <metal_stdlib>
#include <simd/simd.h>

#import "ShaderTypes.h"

using namespace metal;

// Represents a three dimensional ray which will be intersected with the scene. The ray type
// is customized using properties of the MPSRayIntersector.
struct Ray {
    // Starting point
    packed_float3 origin;
    
    // Mask which will be bitwise AND-ed with per-triangle masks to filter out certain
    // intersections. This is used to make the light source visible to the camera but not
    // to shadow or secondary rays.
    uint mask;
    
    // Direction the ray is traveling
    packed_float3 direction;
    
    // Maximum intersection distance to accept. This is used to prevent shadow rays from
    // overshooting the light source when checking for visibility.
    float maxDistance;
    
    // The accumulated color along the ray's path so far
    packed_float3 color;
    
    packed_float3 normal;
    float pdf;
    
    int bounce;
};

// Represents an intersection between a ray and the scene, returned by the MPSRayIntersector.
// The intersection type is customized using properties of the MPSRayIntersector.
struct Intersection {
    // The distance from the ray origin to the intersection point. Negative if the ray did not
    // intersect the scene.
    float distance;
    
    // The index of the intersected primitive (triangle), if any. Undefined if the ray did not
    // intersect the scene.
    int primitiveIndex;
    
    // The barycentric coordinates of the intersection point, if any. Undefined if the ray did
    // not intersect the scene.
    float2 coordinates;
};


struct Sample
{
    float3 liColor;
    float pdf;
};


// Generates rays starting from the camera origin and traveling towards the image plane aligned
// with the camera's coordinate system.
kernel void rayKernel(uint2 tid                    [[thread_position_in_grid]],
                      // Buffers bound on the CPU. Note that 'constant' should be used for small
                      // read-only data which will be reused across threads. 'device' should be
                      // used for writable data or data which will only be used by a single thread.
                      constant Uniforms & uniforms [[buffer(0)]],
                      device Ray *rays          [[buffer(1)]],
                      device float2 *random,
                      texture2d<float, access::write> dstTex [[texture(0)]])
{
    // Since we aligned the thread count to the threadgroup size, the thread index may be out of bounds
    // of the render target size.
    if (tid.x < uniforms.width && tid.y < uniforms.height) {
        // Compute linear ray index from 2D position
        unsigned int rayIdx = tid.y * uniforms.width + tid.x;

        // Ray we will produce
        device Ray & ray = rays[rayIdx];

        // Pixel coordinates for this thread
        float2 pixel = (float2)tid;

        // Add a random offset to the pixel coordinates for antialiasing
        float2 r = random[(tid.y % 16) * 16 + (tid.x % 16)];
        pixel += r;
        
        // Map pixel coordinates to -1..1
        float2 uv = (float2)pixel / float2(uniforms.width, uniforms.height);
        uv = uv * 2.0f - 1.0f;
        
        constant Camera & camera = uniforms.camera;
        
        // Rays start at the camera position
        ray.origin = camera.position;
        
        // Map normalized pixel coordinates into camera's coordinate system
        ray.direction = normalize(uv.x * camera.right +
                                  uv.y * camera.up +
                                  camera.forward);
        // The camera emits primary rays
        ray.mask = RAY_MASK_PRIMARY;
        
        // Don't limit intersection distance
        ray.maxDistance = INFINITY;
        
        // Start with a fully white color. Each bounce will scale the color as light
        // is absorbed into surfaces.
        ray.color = float3(1.0f, 1.0f, 1.0f);
        
        ray.bounce = 0;
        
        // Clear the destination image to black
        dstTex.write(float4(0.0f, 0.0f, 0.0f, 0.0f), tid);
    }
}

// Interpolates vertex attribute of an arbitrary type across the surface of a triangle
// given the barycentric coordinates and triangle index in an intersection struct
template<typename T>
inline T interpolateVertexAttribute(device T *attributes, Intersection intersection) {
    // Barycentric coordinates sum to one
    float3 uvw;
    uvw.xy = intersection.coordinates;
    uvw.z = 1.0f - uvw.x - uvw.y;
    
    unsigned int triangleIndex = intersection.primitiveIndex;
    
    // Lookup value for each vertex
    T T0 = attributes[triangleIndex * 3 + 0];
    T T1 = attributes[triangleIndex * 3 + 1];
    T T2 = attributes[triangleIndex * 3 + 2];
    
    // Compute sum of vertex attributes weighted by barycentric coordinates
    return uvw.x * T0 + uvw.y * T1 + uvw.z * T2;
}

// Uses the inversion method to map two uniformly random numbers to a three dimensional
// unit hemisphere where the probability of a given sample is proportional to the cosine
// of the angle between the sample direction and the "up" direction (0, 1, 0)
inline float3 sampleCosineWeightedHemisphere(float2 u) {
    float phi = 2.0f * M_PI_F * u.x;
    
    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);
    
    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    
    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}


float areaLightPdf(float3 position, float3 lightSamplePos, float3 normal, float area)
{
    float3 lightDirection = lightSamplePos - position;
    float lightDistance = length(lightDirection);
    float pdf = lightDistance * lightDistance;
    pdf /= max(saturate(dot(-lightDirection, normal)), 1e-3);
    pdf /= area;
    
    return pdf;
}


// Maps two uniformly random numbers to the surface of a two-dimensional area light
// source and returns the direction to this point, the amount of light which travels
// between the intersection point and the sample point on the light source, as well
// as the distance between these two points.
inline Sample sampleAreaLight(constant AreaLight & light,
                              float2 u,
                              float3 position,
                              thread float3 & lightDirection,
                              thread float & lightDistance)
{
    Sample result;
    
    // Map to -1..1
    u = u * 2.0f - 1.0f;
    
    // Transform into light's coordinate system
    float3 samplePosition = light.position +
                            light.right * u.x +
                            light.up * u.y;
    
    // Compute vector from sample point on light source to intersection point
    lightDirection = samplePosition - position;
    
    lightDistance = length(lightDirection);
    
    float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
    
    // Normalize the light direction
    lightDirection *= inverseLightDistance;
    
    // Start with the light's color
    result.liColor = light.color;
    
    // Light falls off with the inverse square of the distance to the intersection point
    result.pdf = areaLightPdf(position, samplePosition, light.forward, length(light.right) * 2.0 * length(light.up) * 2.0);
    
    return result;
}

// Aligns a direction on the unit hemisphere such that the hemisphere's "up" direction
// (0, 1, 0) maps to the given surface normal direction
inline float3 alignHemisphereWithNormal(float3 sample, float3 normal)
{
    // Set the "up" vector to the normal
    float3 up = normal;
    
    // Find an arbitrary direction perpendicular to the normal. This will become the
    // "right" vector.
    float3 right = normalize(cross(normal, float3(0.0072f, 1.0f, 0.0034f)));
    if (length(right)  < 1e-3)
        right = simd::normalize(metal::cross(normal, float3 { 0.0072f, 0.0034f, 1.0f }));
    
    // Find a third vector perpendicular to the previous two. This will be the
    // "forward" vector.
    float3 forward = cross(right, up);
    
    // Map the direction on the unit hemisphere to the coordinate system aligned
    // with the normal.
    return sample.x * right + sample.y * up + sample.z * forward;
}

// Consumes ray/triangle intersection results to compute the shaded image
kernel void shadeKernel(uint2 tid [[thread_position_in_grid]],
                        constant Uniforms & uniforms,
                        device Ray *rays,
                        device Ray *shadowRays,
                        device Intersection *intersections,
                        device float *intersectionsShadow,
                        device float3 *vertexColors,
                        device float3 *vertexNormals,
                        device float2 *random,
                        device uint *triangleMasks,
                        texture2d<float, access::read_write> dstTex)
{
    if (tid.x < uniforms.width && tid.y < uniforms.height) {
        unsigned int rayIdx = tid.y * uniforms.width + tid.x;
        device Ray & ray = rays[rayIdx];
        device Ray & shadowRay = shadowRays[rayIdx];
        device Intersection & intersection = intersections[rayIdx];
        
        float3 color = ray.color;
        
        float intersectionDistance = intersectionsShadow[rayIdx];
        
        float3 multipleImportance = float3(0.0);
        
        if (ray.bounce > 0 && ray.maxDistance >= 0.0f && intersection.distance >= 0.0f)
        {
            uint mask = triangleMasks[intersection.primitiveIndex];
            
            if (mask != TRIANGLE_MASK_GEOMETRY)
            {
                multipleImportance = uniforms.light.color / M_PI_F;

                // Terminate the ray's path
                ray.maxDistance = -1.0f;
                
                float3 shadowRayColor = float3(0.0);
                if (shadowRay.maxDistance >= 0.0f && intersectionDistance < 0.0f)
                {
                    shadowRayColor = shadowRay.color;
                }
                
                float3 intersectionPoint = ray.origin + ray.direction * intersection.distance;
                float pdfLight = areaLightPdf(ray.origin, intersectionPoint, uniforms.light.forward,
                                              length(uniforms.light.right) * length(uniforms.light.up) * 4.0);
                float pdfScatter = dot(shadowRay.direction, ray.normal) / M_PI_F;
                
                multipleImportance = shadowRayColor / (pdfLight * shadowRay.pdf) + multipleImportance / (pdfScatter + ray.pdf);
                multipleImportance *= ray.color;
                
                dstTex.write(float4(multipleImportance, 1.0f), tid);
                
                shadowRay.maxDistance = -1.0f;
            }
        }
        
        if (ray.bounce > 0 &&
            shadowRay.maxDistance >= 0.0f && intersectionDistance < 0.0f)
        {
            if (multipleImportance.x == 0 &&
                multipleImportance.y == 0 &&
                multipleImportance.z == 0)
            {
                float3 color = shadowRay.color / shadowRay.pdf;
                
                color += dstTex.read(tid).xyz;
                
                // Write result to render target
                dstTex.write(float4(color, 1.0f), tid);
            }
        }
        
        // Intersection distance will be negative if ray missed or was disabled in a previous
        // iteration.
        if (ray.maxDistance >= 0.0f && intersection.distance >= 0.0f) {
            uint mask = triangleMasks[intersection.primitiveIndex];

            // The light source is included in the acceleration structure so we can see it in the
            // final image. However, we will compute and sample the lighting directly, so we mask
            // the light out for shadow and secondary rays.
            if (mask == TRIANGLE_MASK_GEOMETRY) {
                // Compute intersection point
                float3 intersectionPoint = ray.origin + ray.direction * intersection.distance;

                // Interpolate the vertex normal at the intersection point
                float3 surfaceNormal = interpolateVertexAttribute(vertexNormals, intersection);
                surfaceNormal = normalize(surfaceNormal);

                // Look up two uniformly random numbers for this thread
                float2 r = random[(tid.x % 16) * 16 + (tid.y % 16) + 256 * ray.bounce];

#if D_EMIT_SHADOW_RAY
                float3 lightDirection;
                Sample sampleLi;
                float lightDistance;
                
                // Compute the direction to, color, and distance to a random point on the light
                // source
                sampleLi = sampleAreaLight(uniforms.light, r, intersectionPoint, lightDirection, lightDistance);
                
                // Scale the light color by the cosine of the angle between the light direction and
                // surface normal
                sampleLi.liColor *= saturate(dot(surfaceNormal, lightDirection)) / M_PI_F;
#endif

                // Interpolate the vertex color at the intersection point
                color *= interpolateVertexAttribute(vertexColors, intersection);
                
                // Compute the shadow ray. The shadow ray will check if the sample position on the
                // light source is actually visible from the intersection point we are shading.
                // If it is, the lighting contribution we just computed will be added to the
                // output image.
                
                // Add a small offset to the intersection point to avoid intersecting the same
                // triangle again.
                
#if D_EMIT_SHADOW_RAY
                shadowRay.origin = intersectionPoint + surfaceNormal * 1e-3f;
                
                // Travel towards the light source
                shadowRay.direction = lightDirection;
                
                // Avoid intersecting the light source itself
                shadowRay.mask = RAY_MASK_SHADOW;
                
                // Don't overshoot the light source
                shadowRay.maxDistance = lightDistance - 1e-3f;
                
                shadowRay.normal = surfaceNormal;
                
                // Multiply the color and lighting amount at the intersection point to get the final
                // color, and pass it along with the shadow ray so that it can be added to the
                // output image if needed.
                shadowRay.color = sampleLi.liColor * color;
                shadowRay.pdf = sampleLi.pdf;
#endif
                
                // Next we choose a random direction to continue the path of the ray. This will
                // cause light to bounce between surfaces. Normally we would apply a fair bit of math
                // to compute the fraction of reflected by the current intersection point to the
                // previous point from the next point. However, by choosing a random direction with
                // probability proportional to the cosine (dot product) of the angle between the
                // sample direction and surface normal, the math entirely cancels out except for
                // multiplying by the interpolated vertex color. This sampling strategy also reduces
                // the amount of noise in the output image.
                float3 sampleDirection = sampleCosineWeightedHemisphere(r);
                sampleDirection = alignHemisphereWithNormal(sampleDirection, surfaceNormal);

                ray.origin = intersectionPoint + surfaceNormal * 1e-3f;
                ray.direction = sampleDirection;
                ray.color = color;
                ray.bounce = ray.bounce + 1;
                ray.pdf = max(metal::dot(sampleDirection, surfaceNormal), 1e-3) / M_PI_F;
                ray.normal = surfaceNormal;
                
#if D_EMIT_SHADOW_RAY
                ray.mask = RAY_MASK_SECONDARY;
#endif
            }
            else {
                // In this case, a ray coming from the camera hit the light source directly, so
                // we'll write the light color into the output image.
                float3 color = uniforms.light.color;
                
#if !D_EMIT_SHADOW_RAY
                color *= ray.color;
#endif
                dstTex.write(float4(color, 1.0f), tid);
                
                // Terminate the ray's path
                ray.maxDistance = -1.0f;
                shadowRay.maxDistance = -1.0f;
            }
        }
        else {
            // The ray missed the scene, so terminate the ray's path
            ray.maxDistance = -1.0f;
            shadowRay.maxDistance = -1.0f;
        }
    }
}

// Accumulates the current frame's image with a running average of all previous frames to
// reduce noise over time.
kernel void accumulateKernel(uint2 tid [[thread_position_in_grid]],
                             constant Uniforms & uniforms,
                             texture2d<float> renderTex,
                             texture2d<float, access::read_write> accumTex)
{
    if (tid.x < uniforms.width && tid.y < uniforms.height) {
        float3 color = renderTex.read(tid).xyz;

        // Compute the average of all frames including the current frame
        if (uniforms.frameIndex > 0) {
            float3 prevColor = accumTex.read(tid).xyz;
            prevColor *= uniforms.frameIndex;
            
            color += prevColor;
            color /= (uniforms.frameIndex + 1);
        }
        
        accumTex.write(float4(color, 1.0f), tid);
    }
}

// Screen filling quad in normalized device coordinates
constant float2 quadVertices[] = {
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

struct CopyVertexOut {
    float4 position [[position]];
    float2 uv;
};

// Simple vertex shader which passes through NDC quad positions
vertex CopyVertexOut copyVertex(unsigned short vid [[vertex_id]]) {
    float2 position = quadVertices[vid];
    
    CopyVertexOut out;
    
    out.position = float4(position, 0, 1);
    out.uv = position * 0.5f + 0.5f;
    
    return out;
}

// Simple fragment shader which copies a texture and applies a simple tonemapping function
fragment float4 copyFragment(CopyVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float3 color = tex.sample(sam, in.uv).xyz;
    
    // Apply a very simple tonemapping function to reduce the dynamic range of the
    // input image into a range which can be displayed on screen.
    color = color / (1.0f + color);
    
    return float4(color, 1.0f);
}
