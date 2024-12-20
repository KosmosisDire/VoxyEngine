#include "common.glsl"
#include "lygia/generative/snoise.glsl"

uniform vec3 sunDir;
uniform vec3 sunColor;

#define MAX_RECORDED_HITS 2
#define TRANSPARENT_ACC_RATE 16

// Sky color calculation
vec3 getSkyColor(vec3 dir) {
    float sunDot = max(0.0, dot(dir, sunDir));
    vec3 sun = (pow(sunDot, 48.0) + pow(sunDot, 4.0) * 0.2) * sunColor;
    
    float sunHeight = dot(vec3(0,1,0), sunDir);
    vec3 skyColor = mix(
        mix(sunColor, vec3(0.6, 0.7, 0.9), sunHeight),
        vec3(0.2, 0.3, 0.7),
        dir.y
    );
    
    return sun + skyColor;
}

struct RaymarchHit {
    vec3 position;      // Hit position
    vec3 normal;       // Surface normal at hit
    vec3 perVoxNormal; // Per-voxel normal
    vec3 color;        // Material color at hit
    uint materialId;    // Material ID
    float dist;        // Distance from ray origin
    bool valid;        // Whether this hit is valid
    ivec3 cell;        // Grid cell of the hit
    int blockIndex;    // Block index for the hit
    int voxelIndex;    // Voxel index for the hit
};

// Add MaterialCache struct to store cached properties
struct MaterialCache {
    vec3 color;           // Cached material color
    float transparency;   // Cached transparency value
    uint materialId;      // ID of cached material
    bool valid;          // Whether cache entry is valid
};

struct TransparencyAccumulator {
    // Existing fields
    vec3 color;
    vec3 normal;
    float opacity;     
    RaymarchHit hits[MAX_RECORDED_HITS];
    int numHits;       
    uint lastMaterialId;
    int steps;         
    bool hasHit;       
    int hitCount;      
    
    // New caching fields
    MaterialCache currentMaterial;  // Currently active material cache
};

TransparencyAccumulator initAccumulator() {
    TransparencyAccumulator acc;
    // Initialize existing fields
    acc.color = vec3(0.0);
    acc.normal = vec3(0.0);
    acc.opacity = 0.0;
    acc.numHits = 0;
    acc.lastMaterialId = 0u;
    acc.steps = 0;
    acc.hasHit = false;
    acc.hitCount = 0;
    
    // Initialize material cache
    acc.currentMaterial.valid = false;
    acc.currentMaterial.materialId = 0u;
    
    // Initialize hits array
    for (int i = 0; i < MAX_RECORDED_HITS; i++) {
        acc.hits[i].valid = false;
    }
    return acc;
}


vec3 sampleGradient(Material material, float t) {
    // Ensure t is in 0-1 range
    t = clamp(t, 0.0, 1.0);

    // Find the two control points to interpolate between
    float prevPoint = -1.0;
    float nextPoint = -1.0;
    vec3 prevColor = vec3(0.0);
    vec3 nextColor = vec3(0.0);

    // First find the valid control points that bound our t value
    for (int i = 0; i < 4; i++) {
        float controlPoint = material.colors[i].w;
        if (controlPoint < 0.0) break; // No more valid control points

        if (controlPoint <= t && (prevPoint < 0.0 || controlPoint > prevPoint)) {
            prevPoint = controlPoint;
            prevColor = material.colors[i].xyz;
        }

        if (controlPoint >= t && (nextPoint < 0.0 || controlPoint < nextPoint)) {
            nextPoint = controlPoint;
            nextColor = material.colors[i].xyz;
        }
    }

    // Handle edge cases
    if (prevPoint < 0.0) return nextColor;  // Before first control point
    if (nextPoint < 0.0) return prevColor;  // After last control point
    if (abs(nextPoint - prevPoint) < EPSILON) return prevColor; // Same control point

    // Calculate interpolation factor
    float factor = (t - prevPoint) / (nextPoint - prevPoint);

    // Smooth interpolation
    factor = smoothstep(0.0, 1.0, factor);

    // Interpolate between colors
    return mix(prevColor, nextColor, factor);
}

vec3 getMaterialColor(uint materialId, ivec3 voxelCell) {
    Material material = materials[materialId];

    // Start with base gradient position
    float gradientPos = 0.5;
    
    // Apply noise if enabled
    if (material.noiseStrength > 0.0) {
        // Calculate noise and normalize to 0-1 range
        float noise = snoise(vec3(voxelCell) * material.noiseSize);
        noise = noise * 0.5 + 0.5; // Convert from -1,1 to 0,1 range
        
        // Apply noise strength as a blend between base gradient position and noise
        gradientPos = mix(gradientPos, noise, material.noiseStrength * 30);
    }
    
    // Sample the gradient using our normalized position
    return sampleGradient(material, gradientPos);
}

// Helper function to update material cache
void updateMaterialCache(inout TransparencyAccumulator acc, uint materialId, ivec3 voxelCell) {
    // Only update cache if material changes
    if (!acc.currentMaterial.valid || acc.currentMaterial.materialId != materialId) {
        Material material = materials[materialId];
        acc.currentMaterial.color = getMaterialColor(materialId, voxelCell);
        acc.currentMaterial.transparency = material.transparency;
        acc.currentMaterial.materialId = materialId;
        acc.currentMaterial.valid = true;
    }
}

// Modified compositeOver function
void compositeOver(inout TransparencyAccumulator acc, uint materialId, vec3 newNormal, 
                  vec3 perVoxNormal, float dist, vec3 pos, ivec3 cell, 
                  int blockIndex, int voxelIndex)
{
    // Update material cache if needed
    updateMaterialCache(acc, materialId, cell);
    
    // Use cached values
    vec3 newColor = acc.currentMaterial.color;

    // Record the hit if it's a material transition
    if (acc.numHits < MAX_RECORDED_HITS && (acc.numHits == 0 || acc.currentMaterial.transparency < EPSILON))
    {
        // Create the new hit record
        RaymarchHit hit;
        hit.position = pos;
        hit.normal = newNormal;
        hit.perVoxNormal = perVoxNormal;
        hit.color = newColor;
        hit.materialId = materialId;
        hit.dist = dist;
        hit.valid = true;
        hit.cell = cell;
        hit.blockIndex = blockIndex;
        hit.voxelIndex = voxelIndex;
        
        // Add the hit to the next available slot
        acc.hits[acc.numHits] = hit;
        acc.numHits++;
        acc.lastMaterialId = materialId;
    }
    
    // Accumulate color using cached values
    if (acc.hitCount % TRANSPARENT_ACC_RATE == 0 || acc.currentMaterial.transparency < EPSILON)
    {
        float alpha = 1.0 - acc.currentMaterial.transparency;
        float remainingAlpha = 1.0 - acc.opacity;
        float blendFactor = alpha * remainingAlpha;
        
        acc.color += newColor * blendFactor;
        acc.opacity += blendFactor;
        acc.hasHit = true;

        if (acc.hitCount == 0)
        {
            acc.normal = newNormal;
        }
        else if (acc.currentMaterial.transparency < EPSILON || acc.lastMaterialId != materialId)
        {
            Material material = materials[acc.currentMaterial.materialId];
            acc.normal += newNormal * material.specularStrength;
        }

        if (acc.currentMaterial.transparency < EPSILON)
        {
            acc.normal = normalize(acc.normal);
        }
    }



    
    
    acc.hitCount++;
}



TransparencyAccumulator raymarchMultiLevel(vec3 origin, vec3 direction, int maxSteps, float maxDistance = 64.0) {
    vec3 trueOrigin = origin;
    vec3 invDir = 1.0 / max(abs(direction), vec3(EPSILON));
    invDir *= sign(direction);

    float tMin, tMax;
    TransparencyAccumulator acc = initAccumulator();
    
    if (!intersectBox(origin, invDir, vec3(0), vec3(GRID_SIZE), tMin, tMax)) {
        acc.color = getSkyColor(direction);
        return acc;
    }

    if (tMin > 0) {
        origin += (tMin + EPSILON) * direction;
    }

    vec3 normal;
    
    // Start at chunk level
    DDAState chunkDDA = initDDA(origin, direction, invDir);
    while (acc.steps < maxSteps && isInBounds(chunkDDA.cell, ivec3(GRID_SIZE))) {
        int chunkIndex = getChunkIndex(chunkDDA.cell);
        Bitmask chunkMask = getChunkBitmask(chunkIndex);

        if (!isEmptyMask(chunkMask)) {
            vec3 chunkUVs = getDDAUVs(chunkDDA, origin, direction);
            
            // Block traversal
            vec3 blockOrigin = chunkUVs * float(BLOCK_SIZE);
            blockOrigin = clamp(blockOrigin, vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
            DDAState blockDDA = initDDA(blockOrigin, direction, invDir);

            while (acc.steps < maxSteps && isInBounds(blockDDA.cell, ivec3(BLOCK_SIZE))) {
                int blockLocalIndex = getLocalIndex(blockDDA.cell);

                if (getBit(chunkMask, blockLocalIndex)) {
                    int blockIndex = globalIndex(chunkIndex, blockLocalIndex);
                    Bitmask blockMask = getBlockBitmask(blockIndex);

                    if (!isEmptyMask(blockMask)) {
                        vec3 blockUVs = getDDAUVs(blockDDA, blockOrigin, direction);
                        
                        // Voxel traversal
                        vec3 voxelOrigin = blockUVs * float(BLOCK_SIZE);
                        voxelOrigin = clamp(voxelOrigin, vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
                        DDAState voxelDDA = initDDA(voxelOrigin, direction, invDir);

                        while (acc.steps < maxSteps && isInBounds(voxelDDA.cell, ivec3(BLOCK_SIZE))) {
                            int voxelLocalIndex = getLocalIndex(voxelDDA.cell);

                            if (getBit(blockMask, voxelLocalIndex)) {
                                vec3 voxelUVs = getDDAUVs(voxelDDA, voxelOrigin, direction);
                                ivec3 absoluteCell = chunkDDA.cell * CHUNK_SIZE + 
                                                   blockDDA.cell * BLOCK_SIZE + 
                                                   voxelDDA.cell;
                                vec3 hitPos = vec3(absoluteCell + voxelUVs) / float(CHUNK_SIZE);
                                float dist = distance(trueOrigin, hitPos);

                                if (dist > maxDistance) {
                                    return acc;
                                }

                                vec3 voxelNormal = getPerVoxelNormal(absoluteCell, voxelLocalIndex, 
                                                                    blockIndex, blockMask);
                                vec3 finalNormal = normal;
                                if (voxelNormal != vec3(0)) {
                                    finalNormal = voxelNormal;
                                }

                                int voxelGlobalIndex = globalIndex(blockIndex, voxelLocalIndex);
                                uint materialId = getMaterial(voxelGlobalIndex);

                                // Use optimized compositeOver with material caching
                                compositeOver(acc, materialId, finalNormal, voxelNormal, 
                                           dist, hitPos, absoluteCell, 
                                           blockIndex, voxelLocalIndex);

                                // Early exit if fully opaque
                                if (acc.opacity > 0.99) {
                                    return acc;
                                }

                                // Continue if transparent (using cached transparency)
                                if (acc.currentMaterial.transparency > EPSILON) {
                                    normal = stepDDA(voxelDDA);
                                    acc.steps++;
                                    continue;
                                }

                                return acc;
                            }

                            normal = stepDDA(voxelDDA);
                            acc.steps++;
                        }
                    }
                }

                normal = stepDDA(blockDDA);
                acc.steps++;
            }
        }

        normal = stepDDA(chunkDDA);
        acc.steps++;
    }

    // Blend with sky for partial transparency or complete misses
    if (acc.hasHit) {
        vec3 skyColor = getSkyColor(direction);
        acc.color = acc.color + (1.0 - acc.opacity) * skyColor;
    } else {
        acc.color = getSkyColor(direction);
    }
    
    return acc;
}