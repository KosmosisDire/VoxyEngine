#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Input buffers
layout(std430, binding = 0) readonly buffer PositionsBuffer {
    vec4 positions[];
};

layout(std430, binding = 1) readonly buffer MaterialsBuffer {
    int materials[];
};

layout(std430, binding = 2) buffer LightLevelsBuffer {
    vec4 lightLevels[];
};

layout(std430, binding = 3) readonly buffer ColorsBuffer {
    vec4 colors[];
};

// Uniforms
layout(location = 0) uniform vec3 lightDirection;
layout(location = 1) uniform vec4 lightColor;
layout(location = 2) uniform int chunkSize;
layout(location = 3) uniform float voxelSize;
layout(location = 4) uniform uint currentTime;
layout(location = 5) uniform vec3 cameraPosition;

const ivec3 neighborOffsets[6] = ivec3[6](
    ivec3(-1, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, -1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, -1),
    ivec3(0, 0, 1)
);


float distSquared( vec3 A, vec3 B )
{
    vec3 C = A - B;
    return dot( C, C );
}

float max3(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float min3(vec3 v) {
    return min(min(v.x, v.y), v.z);
}

vec3 scaleMax(vec3 v) {
    float m =  max3(v);
    return v / m;
}

vec3 scaleMin(vec3 v) {
    float m =  min3(v);
    return v / m;
}

int getIndexFromPosition(ivec3 pos) {
    if (pos.x < 0 || pos.x >= chunkSize ||
        pos.y < 0 || pos.y >= chunkSize ||
        pos.z < 0 || pos.z >= chunkSize) {
        return -1;
    }
    return pos.x + (pos.y << int(log2(float(chunkSize)))) + 
           (pos.z << int(log2(float(chunkSize * chunkSize))));
}

vec3 getPositionFromIndex(int index) {
    int x = index % chunkSize;
    int y = (index / chunkSize) % chunkSize;
    int z = index / (chunkSize * chunkSize);
    return vec3(x, y, z) * voxelSize;
}

float random(float seed) {
    // Hash function based on a modified version of Wang hash
    seed = fract(seed * .1031);
    seed *= seed + 33.33;
    seed *= seed + seed;
    return fract(seed);
}

// Alternative version using vec2 seed for more entropy
float random(vec2 seed) {
    return fract(sin(dot(seed.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

vec3 getRandomDirection(float seed) {
    float theta = random(seed) * 2.0 * 3.14159265;
    float phi = acos(2.0 * random(seed + 37) - 1.0);
    
    float sinPhi = sin(phi);
    return vec3(
        cos(theta) * sinPhi,
        sin(theta) * sinPhi,
        cos(phi)
    );
}

bool intersectBox(vec3 origin, vec3 invDir, vec3 boxMin, vec3 boxMax, out float tMin) {
    vec3 t0 = (boxMin - origin) * invDir;
    vec3 t1 = (boxMax - origin) * invDir;
    
    vec3 tMinVec = min(t0, t1);
    vec3 tMaxVec = max(t0, t1);
    
    tMin = max(max(tMinVec.x, tMinVec.y), tMinVec.z);
    float tMax = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
    
    return tMin <= tMax && tMax > 0.0;
}

struct RayHit {
    int index;
    vec3 point;
    vec3 normal;
};

RayHit rayOctreeIntersection(vec3 origin, vec3 dir, float maxDist) {
    vec3 invDir = 1.0 / dir;
    
    RayHit hit;
    hit.index = -1;
    hit.normal = vec3(0);
    
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // Start with root node
    
    while (stackPtr > 0) {
        int current = stack[--stackPtr];
        
        vec4 positionSize = positions[current];
        vec3 position = positionSize.xyz;
        float size = abs(positionSize.w);
        bool isLeaf = positionSize.w < 0.0;
        
        vec3 nodeMin = position;
        vec3 nodeMax = position + vec3(size);
        
        float tMin;
        if (!intersectBox(origin, invDir, nodeMin, nodeMax, tMin) || tMin > maxDist) {
            continue;
        }
        
        if (isLeaf) 
        {
            int dataIndex = getIndexFromPosition(ivec3(position / voxelSize));
            int material = materials[dataIndex];

            if (material == 0) continue;
            
            hit.index = current;
            hit.point = origin + dir * tMin;
            
            const float bias = 0.0001;
            if (abs(hit.point.x - nodeMin.x) < bias)
                hit.normal = vec3(-1.0, 0.0, 0.0);
            else if (abs(hit.point.x - nodeMax.x) < bias)
                hit.normal = vec3(1.0, 0.0, 0.0);
            else if (abs(hit.point.y - nodeMin.y) < bias)
                hit.normal = vec3(0.0, -1.0, 0.0);
            else if (abs(hit.point.y - nodeMax.y) < bias)
                hit.normal = vec3(0.0, 1.0, 0.0);
            else if (abs(hit.point.z - nodeMin.z) < bias)
                hit.normal = vec3(0.0, 0.0, -1.0);
            else if (abs(hit.point.z - nodeMax.z) < bias)
                hit.normal = vec3(0.0, 0.0, 1.0);
                
            return hit;
        } else {
            int firstChildIndex = current * 8 + 1;
            int childOrder = (dir.x > 0.0 ? 1 : 0) | (dir.y > 0.0 ? 2 : 0) | (dir.z > 0.0 ? 4 : 0);
            
            for (int i = 0; i < 8; ++i) {
                stack[stackPtr++] = firstChildIndex + (childOrder ^ i);
            }
        }
    }
    
    return hit;
}

void main() {
    uint index = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * chunkSize + gl_GlobalInvocationID.z * chunkSize * chunkSize;

    // skip solid voxels
    if (materials[index] != 0) return;

    vec3 position = gl_GlobalInvocationID * voxelSize;

    vec3 castPosition = position + vec3(voxelSize * 0.5);
    
    // Early shadow test
    int towardsLightIndex = getIndexFromPosition(ivec3((castPosition / voxelSize) + lightDirection * 2));
    if (towardsLightIndex >= 0 && towardsLightIndex < materials.length() && materials[towardsLightIndex] != 0) {
        lightLevels[index] *= 0.95;
        return;
    }

    bool nextToSolid = false;
    
    for (int i = 0; i < 6; i++) {
        int neighborIndex = getIndexFromPosition(ivec3((position / voxelSize) + neighborOffsets[i]));
        if (neighborIndex >= 0 && neighborIndex < materials.length() && materials[neighborIndex] != 0) {
            nextToSolid = true;
            break;
        }
    }
    
    if (!nextToSolid) return;

    // Primary ray cast
    RayHit hit = rayOctreeIntersection(castPosition, lightDirection, 1000.0);
    
    if (hit.index == -1) {
        // Cast ray in opposite direction for bounce lighting
        hit = rayOctreeIntersection(castPosition, -lightDirection, 10.0);
        
        if (hit.index != -1) {
            // lightLevels[index] = vec4(scaleMax(lightLevels[index].xyz), 1.0);
            lightLevels[index].w = 1.0;
            
            // Skip indirect lighting if too far from camera
            float camDistance = distSquared(cameraPosition, hit.point);
            if ((index * 11) % 2 != 0 && camDistance > 20.0 * 20.0) return;
            if ((index * 17) % 4 != 0 && camDistance > 40.0 * 40.0) return;
            if ((index * 37) % 8 != 0 && camDistance > 80.0 * 80.0) return;
            if ((index * 72) % 16 != 0 && camDistance > 160.0 * 160.0) return;
            if (camDistance > 320.0 * 320.0) return;

            
            vec4 hitIndexPosition = positions[hit.index];
            int hitDataIndex = getIndexFromPosition(ivec3(hitIndexPosition.xyz / voxelSize));
            vec4 blockColor = colors[hitDataIndex];
            vec4 reflectColor = vec4(blockColor.xyz * lightColor.xyz, 0.0);
            
            vec3 reflectDirection = reflect(-lightDirection, hit.normal);
            
            // Indirect lighting calculation
            const int samples = 1;
            for (int i = 0; i < samples; i++)
            {
                uint seed = uint(currentTime + index * 31 + i * 17);

                vec3 scatterDirection = normalize(reflectDirection * 0.25 + getRandomDirection(seed));
                
                vec3 scatterCast = hit.point + hit.normal * 0.01;
                RayHit bounceHit = rayOctreeIntersection(scatterCast, scatterDirection, 50.0);
                
                if (bounceHit.index != -1) {
                    ivec3 adjPosition = ivec3((bounceHit.point / voxelSize) + bounceHit.normal * voxelSize * 0.5);
                    int adjDataIndex = getIndexFromPosition(adjPosition);

                    float sqrDist = (50.0*50.0) / (distSquared(scatterCast, bounceHit.point) * (25.0 * 25.0) + (50.0*50.0));
                    
                    if (adjDataIndex >= 0 && adjDataIndex < lightLevels.length()) {
                        lightLevels[adjDataIndex].xyz = lightLevels[adjDataIndex].xyz * 0.9 + (reflectColor.xyz * 0.1 * sqrDist);
                        
                        // Spread to neighbors
                        for (int j = 0; j < 6; j++) {
                            int neighborIndex = getIndexFromPosition(adjPosition + neighborOffsets[j]);
                            if (neighborIndex >= 0 && neighborIndex < lightLevels.length()) {
                                lightLevels[neighborIndex].xyz = lightLevels[neighborIndex].xyz * 0.9 + (reflectColor.xyz * 0.1 * sqrDist);
                            }
                        }
                    }
                }
            }
        }
    } else {
        lightLevels[index] *= 0.9;
    }
}