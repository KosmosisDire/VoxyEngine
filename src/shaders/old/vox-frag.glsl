#version 460 core

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragNormal;
layout(location = 2) out vec4 FragPosition;



// Required uniforms

uniform float time;
uniform vec2 resolution;
uniform mat4 view;
uniform mat4 projection;
uniform float near;
uniform float far;
uniform vec3 sunDir;
uniform vec3 sunColor;

in vec2 TexCoords;

#include "chunk_common.glsl"
#include "trace_common.glsl"

vec3 background(vec3 d)
{
    const float sun_intensity = 1;
    vec3 sun = (pow(max(0.0, dot(d, sunDir)), 200.0) + pow(max(0.0, dot(d, sunDir)), 4.0) * 0.25) * 1 * vec3(1.0, 0.85, 0.5);
    // vec3 sky = mix(vec3(0.6, 0.65, 0.8), vec3(0.15, 0.25, 0.65), d.y) * 1.15;
    float sunHeight = dot(vec3(0,1,0), sunDir);
    vec3 sky = mix(mix(sunColor, vec3(0.6, 0.65, 0.8), sunHeight) * (sun_intensity), vec3(0.15, 0.25, 0.65), d.y * (sun_intensity)) * 1.15;
    return sun + sky;
}

const ivec3 neighborOffsets[6] = ivec3[6](
    ivec3(-1, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, -1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, -1),
    ivec3(0, 0, 1)
);


bool ray(vec3 p, vec3 d, out vec3 color)
{
    vec3 ambient_color = vec3(0.5, 0.5, 0.7);
    const float view_distance = 2000.0;
    vec3 fog_color = background(d);

    vec3 hitPoint;
    vec3 normal;
    RayHit hit = rayOctreeIntersection(p, d, view_distance);

    if (hit.chunkIndex != -1)
    {
        ivec3 voxelPos = unpackNodeData(hit.nodeIndex).position;
        int dataIndex = getDataIndex(voxelPos, hit.chunkIndex);
        VoxelMaterial material = unpackMaterial(dataIndex);

        float dist = length(hit.point - p);
        float fog_factor = min(1.0, dist * dist / (view_distance * view_distance));

        vec3 diffuse = material.color;

        // get light level at the adjacent block
        vec3 lightPosition = hit.point + hit.normal * 0.1;
        int adjIndex = getVoxelAt(lightPosition);
        
        float sun = max(0.0, dot(hit.normal, sunDir));
 
        const float shadowDist = 500.0;
        if (sun > 0.0 && dist < shadowDist)
        {
            // Shadow ray with chunk transitions
            vec3 shadowOrigin = hit.point + hit.normal * 0.001;
            RayHit shadowHit = rayOctreeIntersection(shadowOrigin, sunDir, shadowDist / 2.0);
            
            if (shadowHit.chunkIndex != -1)
            {
                // set sun towards 0 the closer we are to the shadow so it fades out
                sun = sun * (dist / shadowDist);
            }
        }

        vec4 lightLevel = adjIndex != -1 ? lightLevels[adjIndex] : vec4(0);

        // vec3 c = diffuse * (ambient_color + sun) * lightLevel.w * sunColor + lightLevel.xyz * 0.1;
        vec3 c = diffuse * (ambient_color + sun * sunColor);
        color = mix(c, fog_color, fog_factor * (length(lightLevel.xyz) + lightLevel.w));
        
        return true;
    }

    color = fog_color;
    color = vec3(hit.steps) / 100.0;
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
    // debug node data
    vec2 fragCoord = TexCoords * resolution;

    // FragColor = vec4(globalNodes[int(fragCoord.x)].size / 1.0,0,0, 1);
    // return;
    
    vec3 origin = calculateRayOrigin(); 
    vec3 direction = calculateRayDirection(fragCoord);

    vec3 color;
    ray(origin, direction, color);
    FragColor = vec4(color, 1.0);
}