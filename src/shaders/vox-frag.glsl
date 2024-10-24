#version 460 core

layout(location = 0) out vec4 FragColor;   // Color output
layout(location = 1) out vec4 FragNormal;  // Normal output
layout(location = 2) out vec4 FragPosition;  // Normal output

layout(std430, binding = 0) buffer positionsBuffer { vec4 positions[]; }; // w is size
layout(std430, binding = 1) buffer materialsBuffer { int materials[]; };
layout(std430, binding = 2) buffer lightLevelsBuffer { vec4 lightLevels[]; };
layout(std430, binding = 3) buffer colorsBuffer { vec4 colors[]; };

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
uniform vec3 sunDir;
uniform vec3 sunColor;
uniform float voxelSize;
uniform int chunkSize;

in vec3 FragPos;
in vec2 TexCoords;

const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;
const float HALF_PI = 1.57079632679;

vec3 background(vec3 d)
{
    const float sun_intensity = 1.0;
    vec3 sun = (pow(max(0.0, dot(d, sunDir)), 48.0) + pow(max(0.0, dot(d, sunDir)), 4.0) * 0.25) * sun_intensity * vec3(1.0, 0.85, 0.5);
    // vec3 sky = mix(vec3(0.6, 0.65, 0.8), vec3(0.15, 0.25, 0.65), d.y) * 1.15;
    float sunHeight = dot(vec3(0,1,0), sunDir);
    vec3 sky = mix(mix(sunColor, vec3(0.6, 0.65, 0.8), sunHeight), vec3(0.15, 0.25, 0.65), d.y) * 1.15;
    return sun + sky;
}

float min3(vec3 x)
{
    return min(min(x.x, x.y), x.z);
}

float max3(vec3 x)
{
    return max(max(x.x, x.y), x.z);
}

bool intersectBox(vec3 origin, vec3 invDir, vec3 boxMin, vec3 boxMax, out float tMin)
{
    vec3 t0 = (boxMin - origin) * invDir;
    vec3 t1 = (boxMax - origin) * invDir;
    
    tMin = max3(min(t0, t1));
    float tMax = min3(max(t0, t1));
    
    return tMin <= tMax && tMax > 0.0;
}

int getIndexFromPosition(vec3 position)
{
    if (min3(position) < 0 || max3(position) >= chunkSize) return -1;

    int result =  int(position.x) + int(position.y) * int(chunkSize) + int(position.z) * int(chunkSize) * int(chunkSize);
    return result;
}

int rayOctreeIntersection(vec3 origin, vec3 dir, float maxDist, out vec3 hitPoint, out vec3 normal, out int steps)
{
    vec3 invDir = 1.0 / dir;
    const float bias = 0.001;
    
    int stack[64]; 
    int stackPtr = 0;
    stack[stackPtr++] = 0; // Start with root node
    
    while (stackPtr > 0)
    {
        int current = stack[--stackPtr];

        vec4 positionSize = positions[current];
        vec3 position = positionSize.xyz;
        float size = positionSize.w;
        bool leaf = size < 0.0;
        size = abs(size);
        vec3 nodeMin = position;
        vec3 nodeMax = position + vec3(size);

        float tMin;
        if (!intersectBox(origin, invDir, nodeMin, nodeMax, tMin) || tMin > maxDist)
        {
            continue;
        }

        steps++;

        if (leaf)  // Hit a leaf node
        {
            int dataIndex = getIndexFromPosition(position / voxelSize);
            int material = materials[dataIndex];

            if (material == 0) continue;  // Empty voxel

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
        else
        {
            int firstChildIndex = current * 8 + 1;

            int childOrder = (dir.x > 0.0 ? 1 : 0) | (dir.y > 0.0 ? 2 : 0) | (dir.z > 0.0 ? 4 : 0);

            for (int i = 0; i < 8; ++i) {
                stack[stackPtr++] = firstChildIndex + (childOrder ^ i);  // Efficient traversal
            }
        }
    }
    
    return -1;
}


bool ray(vec3 p, vec3 d, out vec3 color)
{
    vec3 ambient_color = vec3(0.5, 0.5, 0.7);
    const float view_distance = 500.0;
    vec3 fog_color = background(d);

    vec3 hitPoint;
    vec3 normal;
    int steps;
    int hit = rayOctreeIntersection(p, d, view_distance, hitPoint, normal, steps);

    if (hit != -1)
    {
        vec3 voxelPos = positions[hit].xyz;
        int dataIndex = getIndexFromPosition(voxelPos / voxelSize);

        float dist = length(hitPoint - p);
        float fog_factor = min(1.0, dist * dist / (view_distance * view_distance));

        vec3 diffuse = colors[dataIndex].xyz;

        // light level per face
        // find adjacent air voxel
        int adjIndex = getIndexFromPosition((hitPoint / voxelSize) + normal * voxelSize * 0.5);
        vec4 lightLevel = lightLevels[adjIndex];
        if (adjIndex == -1) lightLevel = vec4(1.0);

        float sun = max(0.0, dot(normal, sunDir));

        if (sun > 0.0)
        {
            // calculate shadow
            vec3 shadowOrigin = hitPoint + normal * 0.001;
            vec3 shadowDir = sunDir;
            vec3 shadowHitPoint;
            vec3 shadowNormal;
            int shadowSteps;
            int shadowHit = rayOctreeIntersection(shadowOrigin, shadowDir, view_distance, shadowHitPoint, shadowNormal, shadowSteps);

            if (shadowHit != -1)
            {
                float shadowDist = length(shadowHitPoint - shadowOrigin);
                if (shadowDist < length(shadowHitPoint - hitPoint))
                {
                    sun = 0.0;
                }
            }

        }

        vec3 c = diffuse * (ambient_color + sun) * lightLevel.w * sunColor + lightLevel.xyz;

        color = mix(c, fog_color, fog_factor * (length(lightLevel.xyz) + lightLevel.w));
        // mix steps 
        // color = mix(vec3(color), vec3(float(steps) / 50.0), 0.5);
        return true;
    }

    color = fog_color;
    color = vec3(steps) / 50.0;
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

    vec3 color;
    ray(origin, direction, color);
    FragColor = vec4(color, 1.0);
}
