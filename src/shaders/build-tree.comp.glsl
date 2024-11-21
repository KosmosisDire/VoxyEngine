#version 460

#include "common.glsl"
#include "lygia/generative/snoise.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

// Uniforms for chunk generation
uniform int chunkIndex;
uniform int chunkCount;
const float noiseScale = 0.03;

bool genSolid(vec3 worldPos, int chunkIndex)
{
    // return true;
    // height map with 3 levels of noise
    float h = snoise(vec2(worldPos.x, worldPos.z) * noiseScale);
    h += snoise(vec2(worldPos.x, worldPos.z) * noiseScale * 2) * 0.5;
    h += snoise(vec2(worldPos.x, worldPos.z) * noiseScale * 4) * 0.25;
    h = h * 0.5 + 0.5;
    h *= 7;
    h += 30;

    // generate caves
    float cave = snoise(worldPos * noiseScale * 10);
    cave = pow(cave, 10.0);

    return worldPos.y < h && cave > 0;
}

// Generate a color based on position and fractal values
void getSolidData(vec3 worldPos, out bool solid, out vec3 color, out bool emissive)
{
    solid = genSolid(worldPos, chunkIndex);
    if (solid) {
        emissive = false;
        uint seedx = uint(worldPos.x);
        uint seedy = uint(worldPos.y);
        uint seedz = uint(worldPos.z);
        color = vec3(randomFloat(seedx) * 0.5 + 0.5, randomFloat(seedy) * 0.5 + 0.5, randomFloat(seedz) * 0.5 + 0.5);
        // bool solidAbove = genSolid(worldPos + vec3(0.0, 4 * noiseScale, 0.0), chunkIndex);
        // bool solidAboveFar = genSolid(worldPos + vec3(0.0, 32 * noiseScale, 0.0), chunkIndex);
        // if (!solidAbove && !solidAboveFar) {
        //     color = vec3(0.31, 0.46, 0.39) + snoise(worldPos * 100) * 0.05;
        //     emissive = false;
        // }
        // else if (!solidAbove) {
        //     color = vec3(0.31, 0.27, 0.21) + snoise(worldPos * 50) * 0.05;
        //     emissive = false;
        // }
        // else {
        //     color = vec3(0.7, 0.7, 0.8) + snoise(worldPos * 3) * 0.04;
        //     emissive = true;
        // }
    }
}


void main()
{
    // Get our position in the outer grid
    ivec3 outerPos = ivec3(gl_WorkGroupID) % SPLIT_SIZE;
    // Get our position in the voxel grid
    ivec3 voxelPos = ivec3(gl_LocalInvocationID) % SPLIT_SIZE;

    for (int i = chunkIndex; i < chunkIndex + chunkCount; i++)
    {
        if (i >= GRID_SIZE * GRID_SIZE * GRID_SIZE) break;

        // Calculate world position more precisely
        vec3 chunkOrigin = vec3(getChunkGridPos(i));
        vec3 blockOffset = vec3(outerPos) / float(SPLIT_SIZE);
        vec3 voxelOffset = vec3(voxelPos) / float(SPLIT_SIZE) / float(SPLIT_SIZE);
        vec3 worldPos = chunkOrigin + blockOffset + voxelOffset;
        
        // Generate terrain
        bool solid;
        vec3 color;
        bool emissive;
        getSolidData(worldPos, solid, color, emissive);
        
        if (solid)
        {
            int blockIndex = getSplitIndexLocal(outerPos);
            int voxelIndex = getSplitIndexLocal(voxelPos);
            setVoxel(i, blockIndex, voxelIndex, color, emissive); // Normalize to voxel grid space
        }
    }
    
    
}
