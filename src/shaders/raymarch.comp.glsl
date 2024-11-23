#version 460

#include "common.glsl"
#include "lygia/generative/snoise.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;
layout(rgba32f, binding = 0) uniform image2D outputImage;

// Uniforms
uniform mat4 viewMatrix;
uniform mat4 projMatrix;
uniform vec3 cameraPos;
uniform vec3 sunDir;
uniform vec3 sunColor;
uniform float time;

// Enhanced result struct for raymarch hits
struct RaymarchResult {
    bool hit;
    vec3 position;
    vec3 normal;
    ivec3 cell;
    uint materialId;
    float distance;
    bool isHomogeneous;
    int blockIndex;
    int voxelIndex;
};

// Calculate ray direction from pixel coordinates
vec3 getRayDir(ivec2 pixel, ivec2 screenSize) {
    vec2 uv = (vec2(pixel) + 0.5) / vec2(screenSize);
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc, 1.0, 1.0);
    vec4 viewPos = inverse(projMatrix) * clipPos;
    viewPos.w = 0.0;
    vec3 worldDir = (inverse(viewMatrix) * viewPos).xyz;
    return normalize(worldDir);
}

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

// Simple single-level DDA raymarching for shadows/AO
RaymarchResult raymarchSingle(vec3 origin, vec3 direction, int maxSteps) {
    vec3 invDir = 1.0 / max(abs(direction), vec3(EPSILON));
    invDir *= sign(direction);

    float tMin, tMax;
    if (!intersectBox(origin, invDir, vec3(0), vec3(GRID_SIZE, GRID_SIZE, GRID_SIZE), tMin, tMax)) {
        return RaymarchResult(false, vec3(0), vec3(0), ivec3(0), 0, 0, false, -1, -1);
    }

    if (tMin > 0) {
        origin += (tMin + EPSILON) * direction;
    }

    int steps = 0;
    ivec3 normal;

    vec3 voxelEnter = origin * CHUNK_SIZE;
    DDAState voxelDDA = initDDA(voxelEnter, direction, invDir);

    while (steps < maxSteps && isInBounds(voxelDDA.cell, ivec3(GRID_SIZE * CHUNK_SIZE))) {
        int chunkIndex = getChunkIndex(ivec3(voxelDDA.cell / (BLOCK_SIZE * BLOCK_SIZE)));
        int blockLocalIndex = getLocalIndex(ivec3((voxelDDA.cell / BLOCK_SIZE) % BLOCK_SIZE));
        
        uvec4 chunkMask = chunkMasks[chunkIndex];
        if (!isEmptyMask(chunkMask) && getBit(chunkMask, blockLocalIndex)) {
            int blockIndex = globalIndex(chunkIndex, blockLocalIndex);
            int voxelLocalIndex = getLocalIndex(voxelDDA.cell % BLOCK_SIZE);
            
            if (getBit(blockMasks[blockIndex], voxelLocalIndex)) {
                vec3 hitPos = vec3(voxelDDA.cell) / float(CHUNK_SIZE);
                uint materialId = getBlockMaterialIndex(blockIndex);
                
                return RaymarchResult(
                    true, hitPos, vec3(normal), voxelDDA.cell,
                    materialId, distance(origin, hitPos),
                    false,
                    blockIndex, voxelLocalIndex
                );
            }
        }

        normal = stepDDA(voxelDDA);
        steps++;
    }

    return RaymarchResult(false, vec3(0), vec3(0), ivec3(0), 0, 0, false, -1, -1);
}

// Multi-level raymarching with hierarchical material handling
RaymarchResult raymarchMultiLevel(vec3 origin, vec3 direction, int maxSteps) {
    vec3 invDir = 1.0 / max(abs(direction), vec3(EPSILON));
    invDir *= sign(direction);

    float tMin, tMax;
    if (!intersectBox(origin, invDir, vec3(0), vec3(GRID_SIZE, GRID_SIZE, GRID_SIZE), tMin, tMax)) {
        return RaymarchResult(false, vec3(0), vec3(0), ivec3(0), 0, 0, false, -1, -1);
    }

    if (tMin > 0) {
        origin += (tMin + EPSILON) * direction;
    }

    int steps = 0;
    ivec3 normal;

    // Start at chunk level
    DDAState chunkDDA = initDDA(origin, direction, invDir);
    while (steps < maxSteps && isInBounds(chunkDDA.cell, ivec3(GRID_SIZE))) {
        int chunkIndex = getChunkIndex(chunkDDA.cell);
        uvec4 chunkMask = chunkMasks[chunkIndex];

        if (!isEmptyMask(chunkMask)) {
            // Get entry point into chunk space
            vec3 chunkUVs = getDDAUVs(chunkDDA, origin, direction);
            vec3 blockOrigin = chunkUVs * float(BLOCK_SIZE);
            blockOrigin = clamp(blockOrigin, vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
            
            // Initialize block DDA
            DDAState blockDDA = initDDA(blockOrigin, direction, invDir);

            while (steps < maxSteps && isInBounds(blockDDA.cell, ivec3(BLOCK_SIZE))) {
                int blockLocalIndex = getLocalIndex(blockDDA.cell);

                if (getBit(chunkMask, blockLocalIndex))
                {
                    int blockIndex = globalIndex(chunkIndex, blockLocalIndex);
                    uvec4 blockMask = blockMasks[blockIndex];

                    if (!isEmptyMask(blockMask)) {
                        // Cache homogeneous block material if possible
                        uint materialId = getBlockMaterialIndex(blockIndex);
                        bool isHomogeneous = isBlockHomogenous(blockIndex);

                        // Get entry point into block space
                        vec3 blockUVs = getDDAUVs(blockDDA, blockOrigin, direction);
                        vec3 voxelOrigin = blockUVs * float(BLOCK_SIZE);
                        voxelOrigin = clamp(voxelOrigin, vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
                        
                        // end traversal at higher level
                        if (false)
                        {
                            ivec3 absoluteCell = chunkDDA.cell * CHUNK_SIZE + blockDDA.cell * BLOCK_SIZE;
                            vec3 hitPos = vec3(absoluteCell + voxelOrigin) / float(CHUNK_SIZE);
                            return RaymarchResult(
                                true,                    // hit
                                hitPos,                  // position
                                vec3(normal),            // normal
                                absoluteCell,            // cell
                                materialId,           // materialId
                                distance(origin, hitPos),// distance
                                isHomogeneous,           // isHomogeneous
                                blockIndex,              // blockIndex
                                0                        // voxelIndex
                            );
                        }
                        
                        // Initialize voxel DDA
                        DDAState voxelDDA = initDDA(voxelOrigin, direction, invDir);

                        while (steps < maxSteps && isInBounds(voxelDDA.cell, ivec3(BLOCK_SIZE))) {
                            int voxelLocalIndex = getLocalIndex(voxelDDA.cell);

                            if (getBit(blockMask, voxelLocalIndex)) {
                                vec3 voxelUVs = getDDAUVs(voxelDDA, voxelOrigin, direction);
                                ivec3 absoluteCell = chunkDDA.cell * CHUNK_SIZE + 
                                                   blockDDA.cell * BLOCK_SIZE + 
                                                   voxelDDA.cell;
                                vec3 hitPos = vec3(absoluteCell + voxelUVs) / float(CHUNK_SIZE);

                                if (!isHomogeneous && steps < maxSteps * 0.5)
                                {
                                    materialId = getVoxelMaterial(blockIndex, voxelLocalIndex);
                                }

                                return RaymarchResult(
                                    true,                    // hit
                                    hitPos,                  // position
                                    vec3(normal),            // normal
                                    absoluteCell,            // cell
                                    materialId,           // materialId
                                    distance(origin, hitPos),// distance
                                    isHomogeneous,           // isHomogeneous
                                    blockIndex,              // blockIndex
                                    voxelLocalIndex          // voxelIndex
                                );
                            }

                            normal = stepDDA(voxelDDA);
                            steps++;
                        }
                    }
                }

                normal = stepDDA(blockDDA);
                steps++;
            }
        }

        normal = stepDDA(chunkDDA);
        steps++;
    }

    return RaymarchResult(false, vec3(0), vec3(0), ivec3(0), 0, 0, false, -1, -1);
}

// Shadow ray check
float castShadowRay(vec3 origin, vec3 normal) {
    origin += normal * EPSILON;
    RaymarchResult shadowResult = raymarchMultiLevel(origin, sunDir, 256);
    
    if (shadowResult.hit) {
        return 0.0;
    }
    
    return 1.0;
}

// Calculate ambient occlusion
float calculateAO(vec3 position, vec3 normal, int sampleCount, vec2 pixelCoord) {
    float ao = 0.0;
    uint seed = uint(pixelCoord.x * 123745.0 + pixelCoord.y * 677890.0 + time * 111111.0);
    
    for (int i = 0; i < sampleCount; i++) {
        // Generate hemisphere sample
        vec3 sampleDir = vec3(
            randomFloat(seed) * 2.0 - 1.0,
            randomFloat(seed) * 2.0 - 1.0,
            randomFloat(seed) * 2.0 - 1.0
        );
        sampleDir = normalize(sampleDir * sign(dot(sampleDir, normal)));
        
        RaymarchResult aoResult = raymarchMultiLevel(position + normal * EPSILON, sampleDir, 6);
        if (aoResult.hit) {
            float dist = aoResult.distance;
            ao += 1.0 / (1.0 + dist * dist * 0.1);
        }
    }
    
    return 1.0 - (ao / float(sampleCount));
}

// Material shading function
vec3 calculateShading(RaymarchResult hit) {
    if (hit.materialId == EMPTY_VALUE) {
        return vec3(1, 0, 0);
    }

    Material material = materials[hit.materialId];
    vec3 baseColor = material.color;

    // if (hit.isHomogeneous)
    // {
    //     return vec3(0);
    // }
    // else
    // {
    //     return baseColor;
    // }
    
    // Add material-specific noise if enabled
    if (material.noiseStrength > 0.0) {
        float noise = snoise(vec3(hit.cell) * material.noiseSize) * material.noiseStrength;
        baseColor = clamp(baseColor + vec3(noise), 0.0, 1.0);
    }
    
    // Calculate lighting components
    float sun = max(0.0, dot(hit.normal, sunDir));

    if (sun > 0) 
    {
        sun *= castShadowRay(hit.position, hit.normal);
    }
    
    float ao = 0.4;//calculateAO(hit.position, hit.normal, 1, vec2(pixel));

    // Ambient lighting
    vec3 ambient = baseColor * 0.2 * ao;
    
    // Diffuse lighting
    vec3 diffuse = baseColor * sunColor * sun;
    
    // Emissive
    vec3 emissive = baseColor * material.emission;
    
    // Combine components
    return ambient + diffuse + emissive;
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 screenSize = imageSize(outputImage);
    
    if (any(greaterThanEqual(pixel, screenSize))) {
        return;
    }
    
    // Calculate ray direction
    vec3 rayDir = getRayDir(pixel, screenSize);
    
    // Perform primary ray march
    RaymarchResult result = raymarchMultiLevel(cameraPos, rayDir, 512);
    
    if (result.hit)
    {
        // Calculate final color
        vec3 color = calculateShading(result);
        
        // Apply fog
        float fogAmount = 1.0 - exp(-result.distance * 0.0005);
        color = mix(color, getSkyColor(rayDir), fogAmount);
        
        imageStore(outputImage, pixel, vec4(color, 1.0));
    } else {
        // Sky color for missed rays
        imageStore(outputImage, pixel, vec4(getSkyColor(rayDir), 1.0));
    }
}