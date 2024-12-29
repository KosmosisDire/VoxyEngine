#version 460

#include "raycast.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Match C# structures
struct RaycastHit {
    vec3 position;
    vec3 cellPosition;
    vec3 normal;
    vec3 perVoxNormal;
    vec3 color;
    bool valid;
    uint materialId;
    float distance;
    int blockIndex;
    int voxelIndex;
};

struct RaycastResult {
    vec3 accumulatedColor;
    float opacity;
    int recordedHits;
    int totalSteps;
    bool hasHit;
    int hitCount;
    RaycastHit hits[1]; 
};

struct RaycastInput {
    vec3 origin;
    vec3 direction;
    float maxDistance;
};

// Buffer bindings
layout(std430, binding = 9) buffer RayBuffer {
    RaycastInput rays[];
};

layout(std430, binding = 10) buffer ResultBuffer {
    RaycastResult results[];
};

uniform int rayCount;

// Helper function to convert TraceResult to RaycastHit
RaycastHit convertHit(TraceResult trace) {
    RaycastHit result;
    result.position = trace.position;
    result.normal = trace.normal;
    result.perVoxNormal = trace.perVoxNormal;
    result.color = vec3(0.0); // Placeholder for color, as TraceResult does not include color directly
    result.materialId = trace.materialId;
    result.distance = trace.dist;
    result.cellPosition = vec3(trace.cell);
    result.valid = trace.hit;
    return result;
}

void main() {
    uint rayIndex = gl_GlobalInvocationID.x;
    if (rayIndex >= rayCount) return;

    // Get ray data
    RaycastInput ray = rays[rayIndex];

    // Check basic ray validity
    if (length(ray.direction) < 0.02) {
        return;
    }

    // Perform single-ray trace
    TraceResult trace = traceSingleRay(Ray(ray.origin, normalize(ray.direction), ray.maxDistance), 512);

    // Convert trace result to raycast result
    RaycastResult result;
    result.accumulatedColor = vec3(0.0); // Placeholder, as traceSingleRay does not accumulate color
    result.opacity = trace.hit ? 1.0 : 0.0;
    result.recordedHits = trace.hit ? 1 : 0;
    result.totalSteps = 0; // No direct step tracking available in traceSingleRay
    result.hasHit = trace.hit;
    result.hitCount = trace.hit ? 1 : 0;

    // Populate hits array
    for (int i = 0; i < 1; i++) {
        if (i == 0 && trace.hit) {
            result.hits[i] = convertHit(trace);
        } else {
            // Initialize empty hit
            result.hits[i].valid = false;
        }
    }

    // Store result
    results[rayIndex] = result;
}
