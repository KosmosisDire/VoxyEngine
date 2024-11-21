#version 460
#extension GL_EXT_gpu_shader4 : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Required uniforms
uniform vec3 lightDirection;
uniform vec4 lightColor;
uniform uint currentTime;
uniform vec3 cameraPosition;
uniform int currentChunk;

#include "chunk_common.glsl"
#include "trace_common.glsl"

// Random number generation functions moved to chunk_common.glsl

vec3 getRandomDirection(uint seed) {
    vec3 rand = vec3(
        random(seed),
        random(seed ^ 0x1234567),
        random(seed ^ 0x89ABCDEF)
    );
    float theta = rand.x * 2.0 * 3.14159265;
    float phi = acos(2.0 * rand.y - 1.0);
    float r = pow(rand.z, 1.0/3.0);
    return vec3(
        r * sin(phi) * cos(theta),
        r * sin(phi) * sin(theta),
        r * cos(phi)
    );
}

const ivec3 neighborOffsets[6] = ivec3[6](
    ivec3(-1, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, -1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, -1),
    ivec3(0, 0, 1)
);


void main()
{
    int index = getDataIndex(ivec3(gl_GlobalInvocationID), currentChunk);
    VoxelMaterial material = unpackMaterial(index);
    if (material.isSolid) return;

    // Get chunk data using new structure
    ChunkData chunk = chunks[currentChunk]; 
    vec3 worldPos = vec3(chunk.position) + vec3(gl_GlobalInvocationID) * voxelSize;
    
    vec3 castPosition = worldPos + voxelSize * 0.5;
    
    // // Early shadow test
    // int towardsLightIndex = getVoxelAt(ivec3(castPosition + lightDirection * 2));
    // if (towardsLightIndex >= 0 && towardsLightIndex < materials.length() && materials[towardsLightIndex] != 0) {
    //     lightLevels[index] *= 0.95;
    //     return;
    // }

    // bool nextToSolid = false;
    
    // for (int i = 0; i < 6; i++) {
    //     int neighborIndex = getVoxelAt(ivec3(worldPos + neighborOffsets[i] * voxelSize));
    //     if (neighborIndex >= 0 && neighborIndex < materials.length() && materials[neighborIndex] != 0) {
    //         nextToSolid = true;
    //         break;
    //     }
    // }
    
    // if (!nextToSolid) return;




    RayHit hit = rayOctreeIntersection(castPosition, lightDirection, 1000.0);
    
    if (hit.chunkIndex == -1)
    {
        lightLevels[index].w = 1.0;
        // Cast ray in opposite direction for bounce lighting
        // hit = rayOctreeIntersection(castPosition, -lightDirection, 10.0);
        
        // if (hit.chunkIndex != -1) {
        //     // lightLevels[index] = vec4(scaleMax(lightLevels[index].xyz), 1.0);
        //     lightLevels[index].w = 1.0;
            
        //     // Skip indirect lighting if too far from camera
        //     float camDistance = distSquared(cameraPosition, hit.point);
        //     if ((index * 11) % 2 != 0 && camDistance > 20.0 * 20.0) return;
        //     if ((index * 17) % 4 != 0 && camDistance > 40.0 * 40.0) return;
        //     if ((index * 37) % 8 != 0 && camDistance > 80.0 * 80.0) return;
        //     if ((index * 72) % 16 != 0 && camDistance > 160.0 * 160.0) return;
        //     if (camDistance > 320.0 * 320.0) return;

            
        //     NodeData node = unpackNodeData(hit.nodeIndex);
        //     int dataIndex = getDataIndex(node.position, hit.chunkIndex);
        //     VoxelMaterial material = unpackMaterial(dataIndex);
        //     vec4 reflectColor = vec4(material.color * lightColor.xyz, 0.0);
            
        //     vec3 reflectDirection = reflect(-lightDirection, hit.normal);
            
        //     // Indirect lighting calculation
        //     const int samples = 1;
        //     for (int i = 0; i < samples; i++)
        //     {
        //         uint seed = uint(currentTime + index * 31 + i * 17);

        //         vec3 scatterDirection = normalize(reflectDirection * 0.25 + getRandomDirection(seed));
                
        //         vec3 scatterCast = hit.point + hit.normal * 0.01;
        //         RayHit bounceHit = rayOctreeIntersection(scatterCast, scatterDirection, 50.0);
                
        //         if (bounceHit.chunkIndex != -1) {
        //             ivec3 adjPosition = ivec3((bounceHit.point / voxelSize) + bounceHit.normal * voxelSize * 0.5);
        //             int adjDataIndex = getVoxelAt(adjPosition);

        //             float sqrDist = (50.0*50.0) / (distSquared(scatterCast, bounceHit.point) * (25.0 * 25.0) + (50.0*50.0));
                    
        //             if (adjDataIndex >= 0 && adjDataIndex < lightLevels.length()) {
        //                 lightLevels[adjDataIndex].xyz = lightLevels[adjDataIndex].xyz * 0.9 + (reflectColor.xyz * 0.1 * sqrDist);
                        
        //                 // Spread to neighbors
        //                 for (int j = 0; j < 6; j++) {
        //                     int neighborIndex = getVoxelAt(adjPosition + neighborOffsets[j]);
        //                     if (neighborIndex >= 0 && neighborIndex < lightLevels.length()) {
        //                         lightLevels[neighborIndex].xyz = lightLevels[neighborIndex].xyz * 0.9 + (reflectColor.xyz * 0.1 * sqrDist);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
    } else {
        lightLevels[index] *= 0.9;
    }
}