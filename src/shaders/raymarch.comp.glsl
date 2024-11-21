#version 460

#include "common.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(rgba32f, binding = 0) uniform image2D outputImage;

uniform mat4 viewMatrix;
uniform mat4 projMatrix;
uniform vec3 cameraPos;
uniform vec3 sunDir;
uniform vec3 sunColor;

struct DDAState {
    ivec3 cell;
    ivec3 raySign;
    vec3 deltaDist;
    vec3 sideDist;
};

struct StepCounts {
    int chunkSteps;
    int outerSteps;
    int voxelSteps;
    int totalSteps;
    int reinitializations;
};
 
DDAState initDDA(vec3 pos, vec3 rayDir, vec3 invDir) {
    DDAState state;
    
    state.raySign = ivec3(sign(rayDir));
    state.cell = ivec3(floor(pos));
    state.deltaDist = invDir;
    state.sideDist = ((state.cell - pos) + 0.5 + state.raySign * 0.5) * state.deltaDist;

    return state;
}

vec3 stepMask(vec3 sideDist) {
    // Yoinked from https://www.shadertoy.com/view/l33XWf
    bvec3 move;
    bvec3 pon=lessThan(sideDist.xyz,sideDist.yzx);

    move.x=pon.x && !pon.z;
    move.y=pon.y && !pon.x;
    move.z=!(move.x||move.y);

    return vec3(move);
}

ivec3 stepDDA(inout DDAState state)
{
    ivec3 mask = ivec3(stepMask(state.sideDist));
    ivec3 normalNeg = mask * state.raySign;
    state.cell += normalNeg;
    state.sideDist += mask * state.raySign * state.deltaDist;
    return -normalNeg;
}

void stepDDANoNormal(inout DDAState state)
{
    ivec3 mask = ivec3(stepMask(state.sideDist));
    ivec3 normalNeg = mask * state.raySign;
    state.cell += normalNeg;
    state.sideDist += mask * state.raySign * state.deltaDist;
}

bool isInBounds(ivec3 cell, ivec3 boundsMax) {
    return all(greaterThanEqual(cell, ivec3(0))) && all(lessThan(cell, boundsMax));
}

vec3 getDDAUVs(DDAState state, vec3 rayPos, vec3 rayDir)
{
    vec3 mini = ((vec3(state.cell)-rayPos) + 0.5 - 0.5 * vec3(state.raySign)) * state.deltaDist;
    float d = max (mini.x, max (mini.y, mini.z));
    vec3 intersect = rayPos + rayDir * d;
    vec3 uv3d = intersect - vec3(state.cell);

    if (state.cell == floor(rayPos)) // Handle edge case where camera origin is inside of block
        uv3d = rayPos - state.cell;

    return uv3d;
}

uint raymarchMultilevel(vec3 rayPos, vec3 rayDir, int maxSteps, out StepCounts steps, out ivec3 normal, out vec3 hitPos)
{
    // simple single level dda traversal
    steps.chunkSteps = 0;
    steps.outerSteps = 0;
    steps.voxelSteps = 0;
    steps.totalSteps = 0;
    steps.reinitializations = 0;

    vec3 invDir = 1.0 / rayDir;

    // intersect with the chunk grid
    float tMin, tMax;
    if (intersectBox(rayPos, invDir, vec3(0), vec3(GRID_SIZE), tMin, tMax))
    {
        rayPos += max(tMin - EPSILON, 0.0) * rayDir;
    }

    DDAState chunkDDA = initDDA(rayPos, rayDir, invDir);
    steps.reinitializations++;
    while (steps.totalSteps < maxSteps)
    {
        bool insideGrid = all(lessThanEqual(chunkDDA.cell, ivec3(GRID_SIZE - EPSILON))) && all(greaterThanEqual(chunkDDA.cell, ivec3(EPSILON)));

        uint chunkIndex = getChunkIndex(chunkDDA.cell);
        uint chunkMaskLow = chunkMasksLow[chunkIndex];
        uint chunkMaskHigh = chunkMasksHigh[chunkIndex];

        if (insideGrid && (chunkMaskLow != 0 || chunkMaskHigh != 0))
        {
            uint chunkOffset = chunkIndex * SPLIT_SIZE * SPLIT_SIZE * SPLIT_SIZE;

            vec3 chunkUVs = getDDAUVs(chunkDDA, rayPos, rayDir);
            vec3 blockEnter = clamp(chunkUVs * SPLIT_SIZE, vec3(EPSILON), vec3(SPLIT_SIZE - EPSILON));
            DDAState blockDDA = initDDA(blockEnter, rayDir, invDir);
            steps.reinitializations++;
            while (steps.totalSteps < maxSteps && all(lessThanEqual(blockDDA.cell, ivec3(SPLIT_SIZE - EPSILON))) && all(greaterThanEqual(blockDDA.cell, ivec3(EPSILON))))
            {
                int blockIndexLocal = getSplitIndexLocal(blockDDA.cell);

                if (getBit(blockIndexLocal, chunkMaskLow, chunkMaskHigh))
                {
                    uint blockIndex = chunkOffset + blockIndexLocal;
                    uint blockMaskLow = voxelMasksLow[blockIndex];
                    uint blockMaskHigh = voxelMasksHigh[blockIndex];
                    uint blockDataStartIndex = blockIndices[blockIndex];

                    if (blockDataStartIndex != INVALID_INDEX)
                    {
                        // just return the first voxel for debugging
                        // hitPos = (chunkDDA.cell * CHUNK_SIZE + blockDDA.cell * SPLIT_SIZE + vec3(EPSILON)) * CHUNK_SIZE_INV;
                        // return blockDataStartIndex;

                        vec3 blockUVs = getDDAUVs(blockDDA, blockEnter, rayDir);
                        vec3 voxelEnter = clamp(blockUVs * SPLIT_SIZE, vec3(EPSILON), vec3(SPLIT_SIZE - EPSILON));
                        DDAState voxelDDA = initDDA(voxelEnter, rayDir, invDir);
                        steps.reinitializations++;
                        int innerSteps = 0;
                        while(innerSteps <= 12 && all(lessThanEqual(voxelDDA.cell, ivec3(SPLIT_SIZE - EPSILON))) && all(greaterThanEqual(voxelDDA.cell, ivec3(EPSILON))))
                        {
                            int voxelIndexLocal = getSplitIndexLocal(voxelDDA.cell);
                            if (getBit(voxelIndexLocal, blockMaskLow, blockMaskHigh))
                            {
                                vec3 voxelUVs = getDDAUVs(voxelDDA, voxelEnter, rayDir);
                                hitPos = (chunkDDA.cell * CHUNK_SIZE + blockDDA.cell * SPLIT_SIZE + voxelDDA.cell + voxelUVs) * CHUNK_SIZE_INV;
                                return blockDataStartIndex + voxelIndexLocal;
                            }
                            
                            normal = stepDDA(voxelDDA);
                            steps.voxelSteps++;
                            steps.totalSteps++;
                            innerSteps++;
                        }
                    }
                }

                // Step the DDA
                normal = stepDDA(blockDDA);
                steps.outerSteps++;
                steps.totalSteps++;
            }
        }

        // Step the DDA
        normal = stepDDA(chunkDDA);
        steps.chunkSteps++;
        steps.totalSteps++;
    }

    return INVALID_INDEX;
}

// Modify the single-level raymarch to use the new data structure
uint raymarchSinglelevel(vec3 rayPos, vec3 rayDir, int maxSteps, out int steps)
{
    steps = 0;
    vec3 invDir = 1.0 / rayDir;

    vec3 voxelEnter = rayPos * SPLIT_SIZE * SPLIT_SIZE;
    DDAState voxelDDA = initDDA(voxelEnter, rayDir, invDir);
    while(steps < maxSteps && all(lessThanEqual(voxelDDA.cell, ivec3(GRID_SIZE * SPLIT_SIZE * SPLIT_SIZE - EPSILON))) && all(greaterThanEqual(voxelDDA.cell, ivec3(EPSILON))))
    {
        int chunkIndex = getChunkIndex(ivec3(voxelDDA.cell * CHUNK_SIZE_INV));
        int blockLocalIndex = getSplitIndexLocal(ivec3((voxelDDA.cell / SPLIT_SIZE) % SPLIT_SIZE));
        int blockIndexOffset = chunkIndex * SPLIT_SIZE * SPLIT_SIZE * SPLIT_SIZE + blockLocalIndex;

        // Check if block exists
        bool blockExists = getBit(blockLocalIndex, 
                                chunkMasksLow[chunkIndex], 
                                chunkMasksHigh[chunkIndex]);

        if (blockExists) {
            // Get physical block index and check voxel
            uint physicalBlockIndex = blockIndices[blockIndexOffset];
            uint voxelMaskLow = voxelMasksLow[blockIndexOffset];
            uint voxelMaskHigh = voxelMasksHigh[blockIndexOffset];
            int voxelLocalIndex = getSplitIndexLocal(voxelDDA.cell % SPLIT_SIZE);

            if (getBit(voxelLocalIndex, voxelMaskLow, voxelMaskHigh) && 
                all(lessThanEqual(voxelDDA.cell, ivec3(GRID_SIZE * SPLIT_SIZE * SPLIT_SIZE - EPSILON))) && 
                all(greaterThanEqual(voxelDDA.cell, ivec3(EPSILON))))
            {
                return uint(physicalBlockIndex + voxelLocalIndex);
            }
        }
        
        stepDDANoNormal(voxelDDA);
        steps++;
    }

    return INVALID_INDEX;
}

// rayarch the scene at the block level before stepping down the voxel level only once for the first hit
bool raymarchDoubleLevel(vec3 rayPos, vec3 rayDir, int maxSteps, out int steps)
{
    steps = 0;
    vec3 invDir = 1.0 / rayDir;

    vec3 blockEnter = rayPos * SPLIT_SIZE;
    DDAState blockDDA = initDDA(blockEnter, rayDir, invDir);
    while(steps < maxSteps && all(lessThanEqual(blockDDA.cell, ivec3(GRID_SIZE * SPLIT_SIZE - EPSILON))) && all(greaterThanEqual(blockDDA.cell, ivec3(EPSILON))))
    {
        int chunkIndex = getChunkIndex(ivec3(blockDDA.cell / SPLIT_SIZE));
        int blockIndexLocal = getSplitIndexLocal(ivec3(blockDDA.cell % SPLIT_SIZE));

        // Check if block exists
        bool blockExists = getBit(blockIndexLocal, 
                                chunkMasksLow[chunkIndex], 
                                chunkMasksHigh[chunkIndex]);
        
        if (blockExists)
        {
            int chunkOffset = chunkIndex * SPLIT_SIZE * SPLIT_SIZE * SPLIT_SIZE;
            int blockIndex = chunkOffset + blockIndexLocal;
            uint blockMaskLow = voxelMasksLow[blockIndex];
            uint blockMaskHigh = voxelMasksHigh[blockIndex];

            if (blockMaskLow != 0 || blockMaskHigh != 0)
            {
                vec3 blockUVs = getDDAUVs(blockDDA, blockEnter, rayDir);
                vec3 voxelEnter = clamp(blockUVs * SPLIT_SIZE, vec3(EPSILON), vec3(SPLIT_SIZE - EPSILON));
                DDAState voxelDDA = initDDA(voxelEnter, rayDir, invDir);
                int innerSteps = 0;
                while(innerSteps <= 12 && all(lessThanEqual(voxelDDA.cell, ivec3(SPLIT_SIZE - EPSILON))) && all(greaterThanEqual(voxelDDA.cell, ivec3(EPSILON))))
                {
                    int voxelIndexLocal = getSplitIndexLocal(voxelDDA.cell);
                    if (getBit(voxelIndexLocal, blockMaskLow, blockMaskHigh))
                    {
                        return true;
                    }
                    
                    stepDDANoNormal(voxelDDA);
                    innerSteps++;
                }
            }
        }

        stepDDANoNormal(blockDDA);
        steps++;
    }

    return false;
}

vec3 calculateRayOrigin(ivec2 pixelCoord, ivec2 imageSize) {
    vec3 origin = -vec3(viewMatrix[3]) * mat3(viewMatrix);
    return origin;
}

vec3 calculateRayDirection(vec2 fragCoord) {
    vec2 ndc = (fragCoord / vec2(imageSize(outputImage))) * 2.0 - 1.0;
    vec4 clipSpacePos = vec4(ndc, 1.0, 1.0);
    vec4 viewSpacePos = inverse(projMatrix) * clipSpacePos;
    viewSpacePos.w = 0.0;
    vec3 worldSpaceDir = (inverse(viewMatrix) * viewSpacePos).xyz;
    return normalize(worldSpaceDir);
}

vec3 background(vec3 d)
{
    const float sun_intensity = 1.0;
    vec3 sun = (pow(max(0.0, dot(d, sunDir)), 48.0) + pow(max(0.0, dot(d, sunDir)), 4.0) * 0.25) * sun_intensity * vec3(1.0, 0.85, 0.5);
    // vec3 sky = mix(vec3(0.6, 0.65, 0.8), vec3(0.15, 0.25, 0.65), d.y) * 1.15;
    float sunHeight = dot(vec3(0,1,0), sunDir);
    vec3 sky = mix(mix(sunColor, vec3(0.6, 0.65, 0.8), sunHeight), vec3(0.15, 0.25, 0.65), d.y) * 1.15;
    return sun + sky;
}

vec3 generateHemisphereSample(vec3 normal, int i, int n, inout uint seed) {
    const float PHI = (1.0 + sqrt(5.0)) * 0.5;
    
    // Base distribution with slight randomization
    float i_n = (float(i) + randomFloat(seed) * 0.8) / float(n);
    float phi = 2.0 * 3.14159265359 * (i_n * PHI);
    
    // Cosine weighted distribution for height
    float cosTheta = sqrt(1.0 - i_n);
    float sinTheta = sqrt(i_n);
    
    vec3 direction = vec3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );
    
    // Create rotation basis
    vec3 up = abs(normal.y) < 0.999 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    
    // Transform to world space
    return normalize(
        tangent * direction.x +
        bitangent * direction.y +
        normal * direction.z
    );
}

float smoothFalloff(float dist) {
    // Scale distance for desired AO radius
    dist *= 0.5;
    
    // Smooth falloff curve
    float near = exp(-dist * dist);
    float far = 1.0 / (1.0 + dist * dist);
    
    return mix(near, far, smoothstep(0.0, 1.0, dist));
}

float calculateAO(vec3 hitPos, vec3 normal, ivec2 pixelCoord, ivec2 imageSize) {
    float aoFactor = 0.0;
    const int NUM_SAMPLES = 16;
    int aoSteps;
    
    uint seed = uint(pixelCoord.x * 1973 + pixelCoord.y * 9277) | 1u;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        vec3 aoDir = generateHemisphereSample(normal, i, NUM_SAMPLES, seed);
        vec3 aoOrigin = hitPos + normal * EPSILON;

        bool hit = raymarchSinglelevel(aoOrigin + aoDir * 0.1, aoDir, 10, aoSteps) != INVALID_INDEX;
        
        if (hit) {
            // Use actual distance for falloff calculation
            float dist = float(aoSteps) * 0.00000001;
            float occlusion = smoothFalloff(dist) * 2;
            
            // Weight based on angle between normal and sample direction
            // float weight = max(dot(aoDir, normal), 0.0);
            aoFactor += occlusion;
        }
    }
    
    float ao = 1.0 - (aoFactor / float(NUM_SAMPLES));
    return ao;
}


float directLight(vec3 origin, vec3 lightDir)
{
    StepCounts shadowSteps;
    float light = 1.0;
    vec3 shadowNormal;
    vec3 shadowHitPos;
    uint shadowIndex = raymarchMultilevel(origin, lightDir, 256, shadowSteps, shadowNormal, shadowHitPos);

    if (shadowIndex != INVALID_INDEX)
    {
        light = 0.5;
    }

    return light;
}

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imageSize = imageSize(outputImage);
    
    if (any(greaterThanEqual(pixelCoord, imageSize)))
        return;

    vec3 origin = calculateRayOrigin(pixelCoord, imageSize); 
    vec3 direction = calculateRayDirection(vec2(pixelCoord));

    StepCounts steps;
    ivec3 normal;
    vec3 hitPos;

    uint voxelIndex = raymarchMultilevel(origin, direction, MAX_STEPS, steps, normal, hitPos);
    
    if (voxelIndex != INVALID_INDEX)
    {
        float sqrDepth = dot(hitPos - cameraPos, hitPos - cameraPos);
        vec3 voxelColor = unpackColor(voxelData[voxelIndex]);
        float sun = max(0.0, dot(normal, sunDir));

        // imageStore(outputImage, pixelCoord, vec4(vec3(sun), 1));

        // Shadow calculation
        if (sun > 0.0)
        {
            vec3 shadowOrigin = hitPos + normal * EPSILON;
            vec3 shadowDir = sunDir;
            sun *= directLight(shadowOrigin, shadowDir);
        }
        
        // Ambient occlusion
        float aoFactor = 1.0;
        aoFactor = calculateAO(hitPos, normal, pixelCoord, imageSize);

        // fog
        float fog = 1.0 - exp(-sqrDepth * 0.00005);

        vec3 ambient = voxelColor * 0.6 * aoFactor;
        vec3 direct = voxelColor * sunColor * sun;
        vec3 color = ambient + direct;

        color = mix(color, background(direction), fog);
        
        imageStore(outputImage, pixelCoord, vec4(vec3(color), 1));
    }
    else 
    {
        imageStore(outputImage, pixelCoord, vec4(background(direction), 1));
    }
}