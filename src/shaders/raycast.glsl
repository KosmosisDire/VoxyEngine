#version 460

#include "common.glsl"
#include "lygia/generative/snoise.glsl"
#include "atmosphere.glsl"

struct Ray {
    vec3 origin;
    vec3 direction;
    float maxDist;
};

struct ShadingResult
{
    vec3 color;
    float shadowFactor;
    float aoFactor;
};

struct TraceResult {
    bool hit;
    float dist;
    vec3 position;
    vec3 normal;
    vec3 perVoxNormal;
    uint materialId;
    ivec3 cell;
    vec3 voxelUV;
    float shadowFactor;
    float aoFactor;
};

struct ShadowResult {
    bool completelyBlocked;
    float transmission;
};

// Used to store final per-pixel results for subsequent post-processing.
struct PixelResult {
    vec3 opaqueColor;
    vec3 transparentColor;
    vec3 reflectionColor;
    float shadowFactor;
    float aoFactor;
    TraceResult primary;
    TraceResult reflection;
    TraceResult transmission;
};


vec3 sampleGradient(Material material, float t)
{
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

vec3 getMaterialColor(uint materialId, ivec3 voxelCell, vec3 voxelUVs) {
    Material material = materials[materialId];

    // Start with base gradient position
    float gradientPos = 0.5;
    
    // Apply noise if enabled
    if (material.noiseStrength > 0.0) {
        // Calculate noise and normalize to 0-1 range
        float noise = (snoise(vec3(voxelCell) * material.noiseSize / 20.0) * 0.5 + 0.5) * 0.5;
        noise += (snoise(vec3(voxelCell) * material.noiseSize) * 0.5 + 0.5) * 0.5;
        
        // Apply noise strength as a blend between base gradient position and noise
        gradientPos = mix(gradientPos, noise, material.noiseStrength * 30);
    }
    
    // Sample the gradient using our normalized position
    return sampleGradient(material, gradientPos);
}

Bitmask bitmaskAnd(Bitmask a, Bitmask b) {
    return Bitmask(a.mask1 & b.mask1, a.mask2 & b.mask2, a.mask3 & b.mask3, a.mask4 & b.mask4);
}

Bitmask getDirectionMask(vec3 dir, ivec3 pos)
{
    int dirIndex  = 0;
    if (dir.x < 0) dirIndex |= 1;
    if (dir.y < 0) dirIndex |= 2;
    if (dir.z < 0) dirIndex |= 4;

    int cellIndex = getLocalIndex(pos);

    int bitmaskIndex = (cellIndex * 8 + dirIndex) * 16;

    Bitmask mask = Bitmask(
        uvec4(directionBitmasks[bitmaskIndex], directionBitmasks[bitmaskIndex + 1], directionBitmasks[bitmaskIndex + 2], directionBitmasks[bitmaskIndex + 3]),
        uvec4(directionBitmasks[bitmaskIndex + 4], directionBitmasks[bitmaskIndex + 5], directionBitmasks[bitmaskIndex + 6], directionBitmasks[bitmaskIndex + 7]),
        uvec4(directionBitmasks[bitmaskIndex + 8], directionBitmasks[bitmaskIndex + 9], directionBitmasks[bitmaskIndex + 10], directionBitmasks[bitmaskIndex + 11]),
        uvec4(directionBitmasks[bitmaskIndex + 12], directionBitmasks[bitmaskIndex + 13], directionBitmasks[bitmaskIndex + 14], directionBitmasks[bitmaskIndex + 15])
    );

    return mask;
}

bool hasVoxelsInDirection(Bitmask chunkMask, vec3 dir, ivec3 pos)
{
    // Get the directional bitmask
    Bitmask dirMask = getDirectionMask(dir, pos);
    
    // AND with the chunk occupancy mask
    Bitmask result = bitmaskAnd(chunkMask, dirMask);
    
    return !isEmptyMask(result);
}

bool inBounds(ivec3 c) {
    return all(greaterThanEqual(c,ivec3(EPSILON))) && all(lessThan(c,ivec3(GRID_SIZE)));
}

vec3[4] getFaceVertices(ivec3 voxelPos, vec3 normal) {
    vec3[4] vertices;
    vec3 up, right;
    
    // Calculate perpendicular vectors for the face
    if(abs(normal.x) > 0.99) {
        up = vec3(0,1,0);
        right = vec3(0,0,1);
    } else if(abs(normal.y) > 0.99) {
        up = vec3(0,0,1);
        right = vec3(1,0,0);
    } else {
        up = vec3(0,1,0);
        right = vec3(1,0,0);
    }
    
    // Calculate face vertices
    vec3 basePos = vec3(voxelPos) + normal * 0.5;
    vertices[0] = basePos - right * 0.5 - up * 0.5; // bottom left
    vertices[1] = basePos + right * 0.5 - up * 0.5; // bottom right
    vertices[2] = basePos + right * 0.5 + up * 0.5; // top right
    vertices[3] = basePos - right * 0.5 + up * 0.5; // top left
    
    return vertices;
}

float cornerOcclusion(ivec3 pos, ivec3 dir1, ivec3 dir2, vec3 uv, int radius)
{
    float totalOcclusion = 0.0;
    float totalWeight = 0.0;

    for (int r1 = 1; r1 <= radius; r1++)
    {
        for (int r2 = 1; r2 <= radius; r2++)
        {
            // Sample occupancy at the three possible blocks in this corner wedge
            float occ1 = sampleOpaqueOccupancy(pos + dir1 * r1);
            float occ2 = sampleOpaqueOccupancy(pos + dir2 * r2);
            float occ3 = sampleOpaqueOccupancy(pos + dir1 * r1 + dir2 * r2);

            // A simple distance-based falloff for corners that are further out:
            // (You can also use e.g. 1.0/(r1*r1 + r2*r2) or whatever shape you prefer)
            float dist = length(vec2(r1, r2));
            float distWeight = 1.0 / (1.0 + dist * dist);

            //--------------------------------
            // UV-based corner weighting
            //--------------------------------
            // For dir1, pick whichever UV component is relevant;  same for dir2.
            // E.g. if dir1.x != 0, use uv.x; if dir1.y != 0, use uv.y, etc.
            // Then we shift by sign(dir1.x) to decide if we want uv or (1-uv).
            float w1 = 1.0;
            float w2 = 1.0;

            // Figure out which coordinate to read from uv for dir1
            if (abs(dir1.x) > 0.5) {
                w1 = (dir1.x > 0.0) ? uv.x : (1.0 - uv.x);
            } 
            else if (abs(dir1.y) > 0.5) {
                w1 = (dir1.y > 0.0) ? uv.y : (1.0 - uv.y);
            } 
            else if (abs(dir1.z) > 0.5) {
                w1 = (dir1.z > 0.0) ? uv.z : (1.0 - uv.z);
            }

            // Figure out which coordinate to read from uv for dir2
            if (abs(dir2.x) > 0.5) {
                w2 = (dir2.x > 0.0) ? uv.x : (1.0 - uv.x);
            }
            else if (abs(dir2.y) > 0.5) {
                w2 = (dir2.y > 0.0) ? uv.y : (1.0 - uv.y);
            }
            else if (abs(dir2.z) > 0.5) {
                w2 = (dir2.z > 0.0) ? uv.z : (1.0 - uv.z);
            }

            // Apply a smooth step so that as UV approaches the corner, the weight goes up
            w1 = smoothstep(0.0, 1.0, w1);
            w2 = smoothstep(0.0, 1.0, w2);

            // Final corner weighting for this sample
            float cornerWeight = distWeight * (w1 * w2);

            //--------------------------------
            // Combine occupancy for these 3 samples
            //--------------------------------
            // Weighted sum: you might prefer max(), or average them differently,
            // depending on how "sharp" you want the corner occlusion to be.
            float occludedCount = (occ1 + occ2 + occ3 * 0.5);

            // Accumulate
            totalOcclusion += occludedCount * cornerWeight;
            totalWeight     += cornerWeight;
        }
    }

    // Normalize and invert to get actual AO contribution
    // (the “*0.33” factor is just a tweak so it’s not too dark)
    if (totalWeight > 0.0)
    {
        return 1.0 - (totalOcclusion / totalWeight) * 0.33;
    }
    else
    {
        return 1.0;
    }
}

float calculateAO(ivec3 voxelPos, vec3 normal, vec3 uv, int radius)
{
    // Round normal to figure out which face we’re on
    ivec3 faceDir = ivec3(round(normal));

    // If normal is diagonal-ish, pick whichever axis is largest in magnitude
    if (abs(faceDir.x) + abs(faceDir.y) + abs(faceDir.z) != 1)
    {
        if (abs(normal.x) > abs(normal.y) && abs(normal.x) > abs(normal.z))
            faceDir = ivec3(sign(normal.x), 0, 0);
        else if (abs(normal.y) > abs(normal.x) && abs(normal.y) > abs(normal.z))
            faceDir = ivec3(0, sign(normal.y), 0);
        else
            faceDir = ivec3(0, 0, sign(normal.z));
    }

    // Pick two side directions perpendicular to faceDir
    ivec3 side1, side2;
    if (abs(faceDir.x) == 1)
    {
        side1 = ivec3(0, 1, 0);
        side2 = ivec3(0, 0, 1);
    }
    else if (abs(faceDir.y) == 1)
    {
        side1 = ivec3(1, 0, 0);
        side2 = ivec3(0, 0, 1);
    }
    else
    {
        side1 = ivec3(1, 0, 0);
        side2 = ivec3(0, 1, 0);
    }

    // We sample occlusion in each of the 4 corners around this face
    float aoSum = 0.0;
    ivec3 basePos = voxelPos + faceDir;  // Just beyond face

    aoSum += cornerOcclusion(basePos,  side1,  side2, uv, radius);
    aoSum += cornerOcclusion(basePos,  side1, -side2, uv, radius);
    aoSum += cornerOcclusion(basePos, -side1,  side2, uv, radius);
    aoSum += cornerOcclusion(basePos, -side1, -side2, uv, radius);

    // Average contributions from all four corners
    float ao = aoSum * 0.25;

    // Clamp and return
    return clamp(ao, 0.0, 1.0);
}

TraceResult traceSingleRay(Ray ray, int maxSteps) {
    vec3 startPos = ray.origin; 

    TraceResult res;
    vec3 hitPos = ray.origin + ray.direction * ray.maxDist;
    res.hit = false;
    res.dist = ray.maxDist;
    res.position = hitPos;

    vec3 invDir = 1.0 / max(abs(ray.direction), vec3(EPSILON));
    invDir *= sign(ray.direction);

    float tMin, tMax;
    if (!intersectBox(ray.origin, invDir, vec3(0), vec3(GRID_SIZE), tMin, tMax))
    {
        return res;
    }

    if (tMin > 0) {
        ray.origin += (tMin + EPSILON) * ray.direction;
    }

    int steps=0; 
    vec3 normal;

    if (tMin > 0) {
        normal = -sign(ray.direction);
    } else {
        normal = sign(ray.direction);
    }

    DDAState chunkDDA = initDDA(ray.origin, ray.direction, invDir);
    while (steps < maxSteps && isInBounds(chunkDDA.cell, ivec3(GRID_SIZE)))
    {
        int cIdx = getChunkIndex(chunkDDA.cell);
        Bitmask cMask = getChunkBitmask(cIdx);

        if(!isEmptyMask(cMask))
        {
            // Block DDA
            vec3 chunkUV = getDDAUVs(chunkDDA, ray.origin, ray.direction);
            vec3 blockOrig = clamp(chunkUV * float(BLOCK_SIZE), vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
            DDAState blockDDA = initDDA(blockOrig, ray.direction, invDir);

            while(steps < maxSteps && isInBounds(blockDDA.cell, ivec3(BLOCK_SIZE)) && hasVoxelsInDirection(cMask, ray.direction, blockDDA.cell))
            {    
                int bIdx = getLocalIndex(blockDDA.cell);

                if(getBit(cMask, bIdx))
                {
                    // Voxel DDA
                    int gbIdx = globalIndex(cIdx, bIdx);
                    Bitmask bMask = getBlockBitmask(gbIdx);

                    if(!isEmptyMask(bMask))
                    {
                        vec3 bUV = getDDAUVs(blockDDA, blockOrig, ray.direction);
                        vec3 voxOrig = clamp(bUV * float(BLOCK_SIZE), vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
                        DDAState voxDDA = initDDA(voxOrig, ray.direction, invDir);

                        if (hasVoxelsInDirection(bMask, ray.direction, voxDDA.cell))
                        while(steps < maxSteps && isInBounds(voxDDA.cell, ivec3(BLOCK_SIZE)))
                        {
                            int vIdx = getLocalIndex(voxDDA.cell);

                            if(getBit(bMask, vIdx))
                            {
                                vec3 voxUV = getDDAUVs(voxDDA, voxOrig, ray.direction);
                                ivec3 absCell = chunkDDA.cell * CHUNK_SIZE + blockDDA.cell * BLOCK_SIZE + voxDDA.cell;
                                vec3 hitPos = vec3(absCell + voxUV) / float(CHUNK_SIZE);
                                float dist = distance(startPos, hitPos);

                                if(dist > ray.maxDist) {
                                    return res;
                                }

                                int vgIdx = globalIndex(gbIdx, vIdx);
                                vec3 perVoxNormal = getPerVoxelNormal(absCell, vIdx, gbIdx, bMask);
                                uint seed = absCell.x * 17 + absCell.y * 192 + absCell.z * 172;
                                uint mId = getMaterial(vgIdx);
                                Material material = materials[mId];
                                perVoxNormal = normalize(perVoxNormal + vec3(randomFloat(seed), randomFloat(seed), randomFloat(seed)) * material.roughness);

                                res.hit = true;
                                res.dist = dist;
                                res.position = hitPos;
                                res.normal = normal;
                                res.perVoxNormal = perVoxNormal;
                                res.materialId = mId;
                                res.cell = absCell;
                                res.voxelUV = voxUV;

                                return res;
                            }

                            normal = stepDDA(voxDDA);
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

    return res;
}

TraceResult traceSingleRayIgnoreTransparent(Ray ray, int maxSteps) {
    vec3 startPos = ray.origin; 

    TraceResult res;
    vec3 hitPos = ray.origin + ray.direction * ray.maxDist;
    res.hit = false;
    res.dist = ray.maxDist;
    res.position = hitPos;

    vec3 invDir = 1.0 / max(abs(ray.direction), vec3(EPSILON));
    invDir *= sign(ray.direction);

    float tMin, tMax;
    if (!intersectBox(ray.origin, invDir, vec3(0), vec3(GRID_SIZE), tMin, tMax))
    {
        return res;
    }

    if (tMin > 0) {
        ray.origin += (tMin + EPSILON) * ray.direction;
    }

    int steps = 0; 
    vec3 normal;

    if (tMin > 0) {
        normal = -sign(ray.direction);
    } else {
        normal = sign(ray.direction);
    }

    DDAState chunkDDA = initDDA(ray.origin, ray.direction, invDir);
    while (steps < maxSteps && isInBounds(chunkDDA.cell, ivec3(GRID_SIZE)))
    {
        int cIdx = getChunkIndex(chunkDDA.cell);
        Bitmask cMask = getChunkBitmask(cIdx);

        if(!isEmptyMask(cMask))
        {
            // Block DDA
            vec3 chunkUV = getDDAUVs(chunkDDA, ray.origin, ray.direction);
            vec3 blockOrig = clamp(chunkUV * float(BLOCK_SIZE), vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
            DDAState blockDDA = initDDA(blockOrig, ray.direction, invDir);

            while(steps < maxSteps && isInBounds(blockDDA.cell, ivec3(BLOCK_SIZE)) && hasVoxelsInDirection(cMask, ray.direction, blockDDA.cell))
            {    
                int bIdx = getLocalIndex(blockDDA.cell);

                if(getBit(cMask, bIdx))
                {
                    // Voxel DDA
                    int gbIdx = globalIndex(cIdx, bIdx);
                    Bitmask bMask = getBlockBitmask(gbIdx);

                    if(!isEmptyMask(bMask))
                    {
                        vec3 bUV = getDDAUVs(blockDDA, blockOrig, ray.direction);
                        vec3 voxOrig = clamp(bUV * float(BLOCK_SIZE), vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
                        DDAState voxDDA = initDDA(voxOrig, ray.direction, invDir);

                        if (hasVoxelsInDirection(bMask, ray.direction, voxDDA.cell))
                        while(steps < maxSteps && isInBounds(voxDDA.cell, ivec3(BLOCK_SIZE)))
                        {
                            int vIdx = getLocalIndex(voxDDA.cell);

                            if(getBit(bMask, vIdx))
                            {
                                int vgIdx = globalIndex(gbIdx, vIdx);
                                uint mId = getMaterial(vgIdx);
                                
                                // Check if material is transparent - if so, continue traversal
                                Material material = materials[mId];
                                if (material.transparency > EPSILON) {
                                    normal = stepDDA(voxDDA);
                                    steps++;
                                    continue;
                                }

                                vec3 voxUV = getDDAUVs(voxDDA, voxOrig, ray.direction);
                                ivec3 absCell = chunkDDA.cell * CHUNK_SIZE + blockDDA.cell * BLOCK_SIZE + voxDDA.cell;
                                vec3 hitPos = vec3(absCell + voxUV) / float(CHUNK_SIZE);
                                float dist = distance(startPos, hitPos);

                                if(dist > ray.maxDist) {
                                    return res;
                                }

                                vec3 perVoxNormal = getPerVoxelNormal(absCell, vIdx, gbIdx, bMask);
                                uint seed = absCell.x * 17 + absCell.y * 192 + absCell.z * 172;
                                perVoxNormal = normalize(perVoxNormal + vec3(randomFloat(seed), randomFloat(seed), randomFloat(seed)) * material.roughness);

                                res.hit = true;
                                res.dist = dist;
                                res.position = hitPos;
                                res.normal = normal;
                                res.perVoxNormal = perVoxNormal;
                                res.materialId = mId;
                                res.cell = absCell;
                                res.voxelUV = voxUV;

                                return res;
                            }

                            normal = stepDDA(voxDDA);
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

    return res;
}

ShadowResult traceShadowRay(Ray ray, int maxSteps) {
    vec3 startPos = ray.origin;
    
    // Initialize shadow result
    ShadowResult res;
    res.completelyBlocked = false;
    res.transmission = 1.0;

    vec3 invDir = 1.0 / max(abs(ray.direction), vec3(EPSILON));
    invDir *= sign(ray.direction);

    float tMin, tMax;
    if (!intersectBox(ray.origin, invDir, vec3(0), vec3(GRID_SIZE), tMin, tMax)) {
        return res;
    }

    if (tMin > 0) {
        ray.origin += (tMin + EPSILON) * ray.direction;
    }

    int steps = 0;
    vec3 normal;

    // Calculate initial normal based on entry point
    if (tMin > 0) {
        normal = -sign(ray.direction);
    } else {
        normal = sign(ray.direction);
    }

    // Chunk level traversal
    DDAState chunkDDA = initDDA(ray.origin, ray.direction, invDir);
    while (steps < maxSteps && isInBounds(chunkDDA.cell, ivec3(GRID_SIZE))) {
        int chunkIndex = getChunkIndex(chunkDDA.cell);
        Bitmask chunkMask = getChunkBitmask(chunkIndex);

        if (!isEmptyMask(chunkMask)) {
            // Block traversal
            vec3 chunkUVs = getDDAUVs(chunkDDA, ray.origin, ray.direction);
            vec3 blockOrigin = clamp(chunkUVs * float(BLOCK_SIZE), vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
            DDAState blockDDA = initDDA(blockOrigin, ray.direction, invDir);

            while (steps < maxSteps && isInBounds(blockDDA.cell, ivec3(BLOCK_SIZE)) && 
                   hasVoxelsInDirection(chunkMask, ray.direction, blockDDA.cell)) {
                int blockLocalIndex = getLocalIndex(blockDDA.cell);

                if (getBit(chunkMask, blockLocalIndex)) {
                    int blockIndex = globalIndex(chunkIndex, blockLocalIndex);
                    Bitmask blockMask = getBlockBitmask(blockIndex);

                    if (!isEmptyMask(blockMask)) {
                        vec3 blockUVs = getDDAUVs(blockDDA, blockOrigin, ray.direction);
                        vec3 voxelOrigin = clamp(blockUVs * float(BLOCK_SIZE), vec3(EPSILON), vec3(BLOCK_SIZE - EPSILON));
                        DDAState voxelDDA = initDDA(voxelOrigin, ray.direction, invDir);

                        while (steps < maxSteps && isInBounds(voxelDDA.cell, ivec3(BLOCK_SIZE)) && 
                               hasVoxelsInDirection(blockMask, ray.direction, voxelDDA.cell)) {
                            int voxelLocalIndex = getLocalIndex(voxelDDA.cell);

                            if (getBit(blockMask, voxelLocalIndex)) {
                                vec3 voxelUVs = getDDAUVs(voxelDDA, voxelOrigin, ray.direction);
                                ivec3 absoluteCell = chunkDDA.cell * CHUNK_SIZE + 
                                                   blockDDA.cell * BLOCK_SIZE + 
                                                   voxelDDA.cell;
                                vec3 hitPos = vec3(absoluteCell + voxelUVs) / float(CHUNK_SIZE);
                                float dist = distance(startPos, hitPos);

                                if (dist > ray.maxDist) {
                                    return res;
                                }

                                // Get material and check transparency
                                int voxelGlobalIndex = globalIndex(blockIndex, voxelLocalIndex);
                                uint materialId = getMaterial(voxelGlobalIndex);
                                Material material = materials[materialId];
                                float alpha = 1.0 - material.transparency;

                                // Handle opaque case - early exit
                                if (alpha > 0.99) {
                                    res.completelyBlocked = true;
                                    res.transmission = 0.0;
                                    return res;
                                }

                                // Handle transparent case
                                res.transmission *= material.transparency;
                                if (res.transmission < 0.001) {
                                    return res;
                                }
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

    return res;
}

ShadingResult shadingFromTraceResult(TraceResult primary, Material mat, vec3 normal, vec3 perVoxNormal, vec3 nextRayOrigin, float maxDist)
{
    ShadingResult shading;
    shading.color = vec3(0);
    shading.shadowFactor = 1.0;
    shading.aoFactor = 1.0;


    vec3 baseColor = getMaterialColor(primary.materialId, primary.cell, vec3(0.5));

    float ndl = max(dot(perVoxNormal, sunDir), mat.transparency);
    ShadowResult sRes = traceShadowRay(Ray(nextRayOrigin, sunDir, maxDist), 512);

    shading.color = baseColor;
    shading.shadowFactor = (sRes.transmission) * ndl;
    shading.aoFactor = calculateAO(primary.cell, normal, primary.voxelUV, 2);

    return shading;
}


PixelResult tracePixel(vec3 camOrigin, vec3 camDirection, int maxSteps, float maxDist) 
{
    PixelResult px;
    px.opaqueColor = vec3(0);
    px.transparentColor = vec3(0);
    px.reflectionColor = vec3(0);
    px.shadowFactor = 1.0;
    px.aoFactor = 1.0;
    px.transmission.shadowFactor = 1.0;
    px.transmission.aoFactor = 1.0;
    px.reflection.shadowFactor = 1.0;
    px.reflection.aoFactor = 1.0;
    px.primary.shadowFactor = 1.0;
    px.primary.aoFactor = 1.0;
    px.transmission.perVoxNormal = vec3(0);
    px.reflection.perVoxNormal = vec3(0);
    px.primary.perVoxNormal = vec3(0);
    px.transmission.normal = vec3(0);
    px.reflection.normal = vec3(0);
    px.primary.normal = vec3(0);

    camDirection = normalize(camDirection);
    Ray primaryRay = Ray(camOrigin, camDirection, maxDist);
    TraceResult primary = traceSingleRay(primaryRay, maxSteps);
    primary.shadowFactor = 1.0;
    primary.aoFactor = 1.0;


    if(!primary.hit)
    {
        px.primary = primary;
        return px;
    }

    vec3 normal = primary.normal;
    vec3 perVoxNormal = primary.perVoxNormal;
    vec3 nextRayOrigin = primary.position + 0.001 * normal;
    Material mat = materials[primary.materialId];

    // Shading
    ShadingResult shading = shadingFromTraceResult(primary, mat, normal, perVoxNormal, nextRayOrigin, maxDist);

    primary.shadowFactor = shading.shadowFactor;
    primary.aoFactor = shading.aoFactor;

    px.opaqueColor = shading.color;
    px.shadowFactor = shading.shadowFactor;
    px.aoFactor = shading.aoFactor;

    // Reflection
    if(mat.reflectivity > EPSILON)
    {
        Ray refl;
        refl.origin = nextRayOrigin;
        refl.direction = reflect(primaryRay.direction, primary.perVoxNormal);
        refl.maxDist = maxDist;

        TraceResult reflTrace = traceSingleRayIgnoreTransparent(refl, maxSteps);

        if(reflTrace.hit)
        {
            vec3 reflNormal = reflTrace.normal;
            vec3 reflPerVoxNormal = reflTrace.perVoxNormal;
            vec3 reflNextRayOrigin = reflTrace.position + 0.001 * reflNormal;
            Material reflMat = materials[reflTrace.materialId];

            ShadingResult reflShading = shadingFromTraceResult(reflTrace, reflMat, reflNormal, reflPerVoxNormal, reflNextRayOrigin, maxDist);

            reflTrace.shadowFactor = reflShading.shadowFactor;
            reflTrace.aoFactor = reflShading.aoFactor;

            px.reflectionColor = reflShading.color;
            px.shadowFactor *= reflShading.shadowFactor;
            px.aoFactor *= reflShading.aoFactor;
        }
        else
        {
            px.reflectionColor = getSkyColor(refl.direction);
            reflTrace.shadowFactor = 1.0;
            reflTrace.aoFactor = 1.0;
        }

        px.reflection = reflTrace;
    }

    px.shadowFactor = max(px.shadowFactor, mat.transparency);

    // Refraction / transparency pass
    if(mat.transparency > EPSILON) {
        Ray refr;
        refr.origin = nextRayOrigin;
        refr.direction = primaryRay.direction;
        refr.maxDist = maxDist;

        TraceResult refrTrace = traceSingleRayIgnoreTransparent(refr, maxSteps);
        if(refrTrace.hit)
        {
            // Shading
            vec3 refrNormal = refrTrace.normal;
            vec3 refrPerVoxNormal = refrTrace.perVoxNormal;
            vec3 refrNextRayOrigin = refrTrace.position + 0.001 * refrNormal;
            Material refrMat = materials[refrTrace.materialId];

            ShadingResult refrShading = shadingFromTraceResult(refrTrace, refrMat, refrNormal, refrPerVoxNormal, refrNextRayOrigin, maxDist);

            refrTrace.shadowFactor = refrShading.shadowFactor;
            refrTrace.aoFactor = refrShading.aoFactor;

            px.transparentColor = refrShading.color;
            px.shadowFactor *= refrShading.shadowFactor;
            px.aoFactor *= refrShading.aoFactor;
        }
        else
        {
            px.transparentColor = getSkyColor(refr.direction);
            refrTrace.shadowFactor = 1.0;
            refrTrace.aoFactor = 1.0;
        }

        px.transmission = refrTrace;
    }

    px.shadowFactor = clamp(px.shadowFactor, 0.0, 1.0);
    px.primary = primary;
    
    return px;
}
