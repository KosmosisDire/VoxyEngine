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
    int numHits;
    int totalSteps;
    bool hasHit;
    int hitCount;
    RaycastHit hits[MAX_RECORDED_HITS]; 
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

// Helper function to convert RaymarchHit to RaycastHit
RaycastHit convertHit(RaymarchHit hit) {
    RaycastHit result;
    result.position = hit.position;
    result.normal = hit.normal;
    result.perVoxNormal = hit.perVoxNormal;
    result.color = hit.color;
    result.materialId = hit.materialId;
    result.distance = hit.dist;
    result.cellPosition = vec3(hit.cell);
    result.blockIndex = hit.blockIndex;
    result.voxelIndex = hit.voxelIndex;
    result.valid = hit.valid;
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
    
    // Perform raymarch with new accumulator-based system
    TransparencyAccumulator acc = raymarchMultiLevel(
        ray.origin, 
        normalize(ray.direction), 
        512,
        ray.maxDistance
    );

    // Convert accumulator to result structure
    RaycastResult result;
    result.accumulatedColor = acc.color;
    result.opacity = acc.opacity;
    result.numHits = acc.numHits;
    result.totalSteps = acc.steps;
    result.hasHit = acc.hasHit;
    result.hitCount = acc.hitCount;
    
    // Convert hits
    for (int i = 0; i < MAX_RECORDED_HITS; i++) {
        if (i < acc.numHits) {
            result.hits[i] = convertHit(acc.hits[i]);
        } else {
            // Initialize empty hit
            result.hits[i].valid = false;
        }
    }
    
    // Store result
    results[rayIndex] = result;
}