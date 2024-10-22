#version 460 core

layout(location = 0) out vec4 FragColor;   // Color output
layout(location = 1) out vec4 FragNormal;  // Normal output
layout(location = 2) out vec4 FragPosition;  // Normal output

layout(std430, binding = 0) buffer positionsBuffer { vec4 positions[]; }; // w is size
layout(std430, binding = 1) buffer materialsBuffer { int materials[]; };
layout(std430, binding = 2) buffer lightLevelsBuffer { float lightLevels[]; };
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
uniform vec3 sun_dir;
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
    vec3 sun = (pow(max(0.0, dot(d, sun_dir)), 48.0) + pow(max(0.0, dot(d, sun_dir)), 4.0) * 0.25) * sun_intensity * vec3(1.0, 0.85, 0.5);
    vec3 sky = mix(vec3(0.6, 0.65, 0.8), vec3(0.15, 0.25, 0.65), d.y) * 1.15;
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
    position /= voxelSize;
    int result =  int(position.x + position.y * chunkSize + position.z * chunkSize * chunkSize);

    if (result < 0 || result >= chunkSize*chunkSize*chunkSize)
    {
        return -1;
    }

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

        int dataIndex = getIndexFromPosition(position);
        int material = materials[dataIndex];

        if (leaf)  // Hit a leaf node
        {
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

            // Precompute child order in a more efficient manner without sign() calls
            int childOrder = (dir.x > 0.0 ? 1 : 0) | (dir.y > 0.0 ? 2 : 0) | (dir.z > 0.0 ? 4 : 0);

            for (int i = 0; i < 8; ++i) {
                stack[stackPtr++] = firstChildIndex + (childOrder ^ i);  // Efficient traversal
            }
        }
    }
    
    return -1;
}


bool ray(vec3 p, vec3 d, out vec3 color, out vec3 normal, out float depth, out vec3 position)
{
    vec3 ambient_color = vec3(0.5, 0.5, 0.7);
    const vec3 sun_color = vec3(0.7, 0.5, 0.5);
    const float view_distance = 500.0;
    vec3 fog_color = background(d);

    vec3 hitPoint;
    int steps;
    int hit = rayOctreeIntersection(p, d, view_distance, hitPoint, normal, steps);

    if (hit != -1)
    {
        // vec3 voxelPosition = positions[hit].xyz;
        // int dataIndex = getIndexFromPosition(voxelPosition);

        // block infront of hit
        // int faceAdjacentIndex = getIndexFromPosition(floor(hitPoint + (normal * voxelSize * 0.1)));
        // if (faceAdjacentIndex == -1) faceAdjacentIndex = hit;
        // float lightLevel = lightLevels[faceAdjacentIndex];

        // Calculate the distance from the origin to the hitPoint for depth
        // position = (view * vec4(hitPoint, 1.0)).xyz;
        float dist = length(hitPoint - p);
        // depth = dist;

        float fog_factor = min(1.0, dist * dist / (view_distance * view_distance));
        // fog_factor *= lightLevel;
        // float sun_factor = max(0.0, dot(normal, sun_dir));
        float sun_factor = 1.0;

        // Shadow calculation (optional)
        // if (sun_factor > 0) 
        // {
        //     vec3 shadowHitPoint, shadowNormal;
        //     int stepsShadow;
        //     int shadowHit = rayOctreeIntersection(hitPoint + normal * 0.001, normalize(sun_dir), 50, shadowHitPoint, shadowNormal, stepsShadow);

        //     if (shadowHit != -1)
        //     {
        //         sun_factor = 0;
        //     }
        // }
 
        // float ambient_factor = 0.5 * lightLevel;
        float ambient_factor = 0.5;
        vec3 texel = vec3(0.5);
        vec3 diffuse = texel;
        vec3 c = diffuse * (ambient_factor * ambient_color + sun_factor * sun_color);

        color = mix(c, fog_color, fog_factor);
        // mix steps 
        color = mix(vec3(color), vec3(float(steps) / 10.0), 0.1);
        return true;
    }

    // position = p + d * view_distance;
    color = vec3(steps) / 10.0;
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
    // float ssaoFactor = texture(ssaoTexture, TexCoords).r;
    // color *= ssaoFactor; // Apply SSAO factor
    
    FragColor = vec4(color, 1.0);
    FragNormal = vec4(normal, 1.0);
    FragPosition = vec4(position, 1.0);
}
