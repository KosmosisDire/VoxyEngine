#version 460 core

#define MAX_OCTREE_NODES 3000000
#define MAX_STEPS 128
#define EPSILON 0.001

layout(location = 0) out vec4 FragColor;   // Color output
layout(location = 1) out vec4 FragNormal;  // Normal output
layout(location = 2) out vec4 FragPosition;  // Normal output

layout(std430, binding = 0) buffer nodeDataBuffer
{
    vec4 posAndSize[MAX_OCTREE_NODES]; // xyz: position, w: size
};
layout(std430, binding = 1) buffer nodeInfoBuffer
{
    vec4 nodeData[MAX_OCTREE_NODES]; // x: material, y: isLeaf, z: lightLevel, w: unused
};

uniform sampler2D ssaoTexture;

uniform float time;
uniform vec2 mouse;
uniform vec2 resolution;
uniform mat4 view;
uniform mat4 projection;
uniform float near;
uniform float far;
uniform int octreeNodeCount;
uniform int maxTreeDepth;


in vec3 FragPos;
in vec2 TexCoords;

const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;
const float HALF_PI = 1.57079632679;

const vec3 sun_dir = normalize(vec3(0.7, 0.5, 0.5));

// Function declarations
float sq(float x) { return x * x; }
float sq(vec2 x) { return dot(x, x); }
float sqi(float x) { return 1.0 - sq(1.0 - x); }
float sqi(vec2 x) { return 1.0 - sq(1.0 - x); }

float fastSqrt(float x) {
    return x * inversesqrt(x);
}

float fastSin(float x) 
{
    x = mod(x + PI, TWO_PI) - PI; // restrict x to [-PI, PI]
    float s = x * (1.0 - abs(x) * 0.3183098861837907);
    return s * (1.0 - abs(s) * 0.22308510060189463);
}

float fastCos(float x) 
{
    return fastSin(x + HALF_PI);
}

float fastAsin(float x) {
    return fastSin(x) * (1.5707963267948966 - abs(x) * (1.5707963267948966 * abs(x)));
}

float fastAcos(float x) {
    return HALF_PI - fastAsin(x);
}

vec2 FastSinCosV5(float x) //Calculate sin/cos together to save instructions
{
    //Base on FastSinV5
	vec2 zeroTo2PI = mod(vec2(x, x) + vec2(0, HALF_PI), vec2(TWO_PI, TWO_PI)); //move to range 0-2pi
    vec4 core = vec4(zeroTo2PI.xxyy) * (1.0f / HALF_PI)  + vec4(-1.0f, -3.0f, -1.0f, -3.0f);
    vec4 result = clamp(-core * core + vec4(1.0f, 1.0f, 1.0f, 1.0f), 0, 1);
    return vec2(result.xz - result.yw);
}


vec3 background(vec3 d)
{
    const float sun_intensity = 1.0;
    vec3 sun = (pow(max(0.0, dot(d, sun_dir)), 48.0) + pow(max(0.0, dot(d, sun_dir)), 4.0) * 0.25) * sun_intensity * vec3(1.0, 0.85, 0.5);
    vec3 sky = mix(vec3(0.6, 0.65, 0.8), vec3(0.15, 0.25, 0.65), d.y) * 1.15;
    return sun + sky;
}

bool intersectBox(vec3 origin, vec3 invDir, vec3 boxMin, vec3 boxMax, out float tMin, out float tMax)
{
    vec3 t0 = (boxMin - origin) * invDir;
    vec3 t1 = (boxMax - origin) * invDir;
    
    tMin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    tMax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    
    return tMin <= tMax && tMax > 0.0;
}

int findNodeIndex(vec3 position)
{
    int currentIndex = 0;
    vec4 rootNode = posAndSize[0];
    float size = rootNode.w;

    if (position.x < rootNode.x || position.x < rootNode.y || position.z < rootNode.z ||
        position.x >= rootNode.x + size || position.y >= rootNode.y + size || position.z >= rootNode.z + size)
    {
        return -1;
    }

    while (currentIndex < octreeNodeCount)
    {
        if (nodeData[currentIndex].y == 1)
            return currentIndex;

        float childSize = size * 0.5f;

        int childOffset = ((position.x >= rootNode.x + childSize ? 1 : 0) |
                            (position.y >= rootNode.y + childSize ? 2 : 0) |
                            (position.z >= rootNode.z + childSize ? 4 : 0));

        currentIndex = currentIndex * 8 + 1 + childOffset;

        if ((childOffset & 1) != 0) rootNode.x += childSize;
        if ((childOffset & 2) != 0) rootNode.y += childSize;
        if ((childOffset & 4) != 0) rootNode.z += childSize;

        size = childSize;
    }

    return -1;
}

// int findNodeIndex(vec3 position)
// {
//     const int maxDepth = 6;
//     const float rootSize = 64.0;

//     // Check if the position is within the octree bounds
//     if (position.x < 0 || position.y < 0 || position.z < 0 ||
//         position.x >= rootSize || position.y >= rootSize || position.z >= rootSize)
//     {
//         return -1;
//     }

//     // Normalize positions to [0, 1) range
//     float fx = position.x / rootSize;
//     float fy = position.y / rootSize;
//     float fz = position.z / rootSize;

//     // Convert to integer coordinates
//     int x = int(fx * (1 << maxDepth));
//     int y = int(fy * (1 << maxDepth));
//     int z = int(fz * (1 << maxDepth));

//     // Interleave bits of x, y, and z
//     int index = 0;
//     for (int i = 0; i < maxDepth; ++i) {
//         index |= ((x & (1 << i)) << (2 * i)) |
//                  ((y & (1 << i)) << (2 * i + 1)) |
//                  ((z & (1 << i)) << (2 * i + 2));
//     }

//     // Calculate the final index
//     return (1 << (3 * maxDepth) - 1) / 7 + index;
// }


int rayOctreeIntersection(vec3 origin, vec3 dir, float maxDist, out vec3 hitPoint, out vec3 normal)
{
    vec3 invDir = 1.0 / dir;
    const float bias = 0.001;
    
    int stack[64]; 
    int stackPtr = 0;
    stack[stackPtr++] = 0; // Start with root node
    
    while (stackPtr > 0)
    {
        int current = stack[--stackPtr];

        vec4 nodePosAndSize = posAndSize[current];
        vec3 nodeMin = nodePosAndSize.xyz;
        float nodeSize = nodePosAndSize.w;
        vec3 nodeMax = nodeMin + vec3(nodeSize);

        float tMin, tMax;
        if (!intersectBox(origin, invDir, nodeMin, nodeMax, tMin, tMax) || tMin > maxDist)
        {
            continue;
        }

        // Avoid repeated access to global memory for material info
        vec4 data = nodeData[current];
        int material = int(data.x);
        int isLeaf = int(data.y);

        if (material != 0)  // Hit a leaf node
        {
            if (material == 0) return -1;

            // Calculate the hit point only once upon intersection
            hitPoint = origin + dir * tMin;

            if (abs(hitPoint.x - nodeMin.x) < bias) {
                normal = vec3(-1.0, 0.0, 0.0);  // Hit on the -X face
            } else if (abs(hitPoint.x - nodeMax.x) < bias) {
                normal = vec3(1.0, 0.0, 0.0);   // Hit on the +X face
            } else if (abs(hitPoint.y - nodeMin.y) < bias) {
                normal = vec3(0.0, -1.0, 0.0);  // Hit on the -Y face
            } else if (abs(hitPoint.y - nodeMax.y) < bias) {
                normal = vec3(0.0, 1.0, 0.0);   // Hit on the +Y face
            } else if (abs(hitPoint.z - nodeMin.z) < bias) {
                normal = vec3(0.0, 0.0, -1.0);  // Hit on the -Z face
            } else if (abs(hitPoint.z - nodeMax.z) < bias) {
                normal = vec3(0.0, 0.0, 1.0);   // Hit on the +Z face
            }

            return current;
        }
        else  // Continue traversing the octree
        {
            int firstChildIndex = current * 8 + 1;

            // Precompute child order in a more efficient manner without sign() calls
            int childOrder = (dir.x > 0.0 ? 1 : 0) | (dir.y > 0.0 ? 2 : 0) | (dir.z > 0.0 ? 4 : 0);

            for (int i = 0; i < 8; ++i) {
                stack[stackPtr++] = firstChildIndex + (childOrder ^ i);  // Efficient traversal
            }
        }
    }
    
    return -1;
}

vec3 moduloF(vec3 x, float y)
{
    return x - y * floor(x / y);
}

bool ray(vec3 p, vec3 d, out vec3 color, out vec3 normal, out float depth, out vec3 position)
{
    vec3 ambient_color = vec3(0.5, 0.5, 0.7);
    const vec3 sun_color = vec3(0.7, 0.5, 0.5);
    const float view_distance = 200.0;

    vec3 hitPoint;
    vec3 fog_color = background(d);
    int hit = rayOctreeIntersection(p, d, view_distance, hitPoint, normal);

    if (hit != -1)
    {
        vec4 transform = posAndSize[hit];
        vec3 nodePosition = transform.xyz;
        vec3 nodeCenter = nodePosition + vec3(0.5);
        vec4 data = nodeData[hit];
        float lightLevel = data.z;
        vec3 hitPositionLocal = moduloF((hitPoint - normal * 0.0001) - nodePosition, 1.0) - vec3(0.5);

        // float updown = 0.0;
        // vec3 up = nodePosition + vec3(0, 1, 0);
        // int upIndex = findNodeIndex(up + vec3(0,0.1,0));

        // if (upIndex != -1)
        // {
        //     vec4 neighborData = nodeData[upIndex];
        //     float neighborLight = neighborData.z;
        //     float dist = up.y - hitPoint.y;
        //     updown += neighborLight * max(0.0, 1.0 - dist);
        // }

        // vec3 down = nodePosition;
        // int downIndex = findNodeIndex(nodeCenter - vec3(0, 0.1, 0));

        // if (downIndex != -1)
        // {
        //     vec4 neighborData = nodeData[downIndex];
        //     float neighborLight = neighborData.z;
        //     float dist = hitPoint.y - down.y;
        //     updown += neighborLight * max(0.0, 1.0 - dist);
        // }

        // float leftright = 0.0;
        // vec3 left = nodePosition + vec3(0, 0, 1);
        // int leftIndex = findNodeIndex(left + vec3(0,0,0.1));

        // if (leftIndex != -1)
        // {
        //     vec4 neighborData = nodeData[leftIndex];
        //     float neighborLight = neighborData.z;
        //     float dist = left.z - hitPoint.z;
        //     leftright += neighborLight * max(0.0, 1.0 - dist);
        // }

        // vec3 right = nodePosition;
        // int rightIndex = findNodeIndex(nodeCenter - vec3(0, 0, 0.1));

        // if (rightIndex != -1)
        // {
        //     vec4 neighborData = nodeData[rightIndex];
        //     float neighborLight = neighborData.z;
        //     float dist = hitPoint.z - right.z;
        //     leftright += neighborLight * max(0.0, 1.0 - dist);
        // }

        // float frontback = 0.0;
        // vec3 front = nodePosition + vec3(1, 0, 0);
        // int frontIndex = findNodeIndex(front + vec3(0.1,0,0));

        // if (frontIndex != -1)
        // {
        //     vec4 neighborData = nodeData[frontIndex];
        //     float neighborLight = neighborData.z;
        //     float dist = front.x - hitPoint.x;
        //     frontback += neighborLight * max(0.0, 1.0 - dist);
        // }

        // vec3 back = nodePosition;
        // int backIndex = findNodeIndex(nodeCenter - vec3(0.1, 0, 0));

        // if (backIndex != -1)
        // {
        //     vec4 neighborData = nodeData[backIndex];
        //     float neighborLight = neighborData.z;
        //     float dist = hitPoint.x - back.x;
        //     frontback += neighborLight * max(0.0, 1.0 - dist);
        // }

        // lightLevel = (lightLevel + updown + leftright + frontback) * 0.25;

        fog_color *= lightLevel;
        ambient_color *= lightLevel;

        // Calculate the distance from the origin to the hitPoint for depth
        position = (view * vec4(hitPoint, 1.0)).xyz;
        float dist = length(hitPoint - p);
        depth = dist;

        float fog_factor = min(1.0, dist * dist / (view_distance * view_distance));
        float sun_factor = max(0.0, dot(normal, sun_dir));

        // Check for ground (special case)  
        if (abs(hitPoint.y - 18.0) < 0.1) {
            vec3 c = 1.9 * ambient_color + sun_factor * sun_color;
            color = mix(c, fog_color, fog_factor * 0.6 + 0.4);
            return true;
        }

        // Shadow calculation (optional)
        if (sun_factor > 0.1) 
        {
            vec3 shadowHitPoint, shadowNormal;
            int shadowHit = rayOctreeIntersection(hitPoint + normal * 0.001, normalize(sun_dir), 50, shadowHitPoint, shadowNormal);

            if (shadowHit != -1)
            {
                sun_factor = 0;
            }
        }

        float ambient_factor = 0.5;
        vec3 texel = vec3(0.5);
        vec3 diffuse = texel;
        vec3 c = diffuse * (ambient_factor * ambient_color + sun_factor * sun_color);

        color = mix(c, fog_color, fog_factor);
        return true;
    }

    position = p + d * view_distance;
    color = fog_color;
    depth = far;
    return false;
}

vec3 calculateRayOrigin() {
    return -vec3(view[3]) * mat3(view);
}

vec3 calculateRayDirection(vec2 fragCoord) {
    vec2 ndc = (fragCoord / resolution) * 2.0 - 1.0;
    vec4 clipSpacePos = vec4(ndc, 1.0, 1.0);
    vec4 viewSpacePos = inverse(projection) * clipSpacePos;
    viewSpacePos.w = 0.0;
    vec3 worldSpaceDir = (inverse(view) * viewSpacePos).xyz;
    return normalize(worldSpaceDir);
}

void main()
{
    vec2 fragCoord = TexCoords * resolution;
    
    vec3 origin = calculateRayOrigin();
    vec3 direction = calculateRayDirection(fragCoord);

    // Calculate color and depth value
    float depth;
    vec3 normal;
    vec3 color;
    vec3 position;
    ray(origin, direction, color, normal, depth, position);
    
    // Normalize the depth value between 0 and 1
    float normalizedDepth = (depth - near) / (far - near);
    
    // Write to depthTexture
    gl_FragDepth = normalizedDepth;

    // Apply simple tone mapping for color
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2)); // Gamma correction

    float ssaoFactor = texture(ssaoTexture, TexCoords).r;
    color *= ssaoFactor; // Apply SSAO factor
    
    FragColor = vec4(color, 1.0);
    FragNormal = vec4(normal, 1.0);
    FragPosition = vec4(position, 1.0);
}
