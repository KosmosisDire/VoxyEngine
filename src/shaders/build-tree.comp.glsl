#version 460

#include "common.glsl"
#include "lygia/generative/snoise.glsl"

layout(local_size_x = 5, local_size_y = 5, local_size_z = 5) in;

// Uniforms for chunk generation
uniform int chunkIndex;
uniform int chunkCount;
const float noiseScale = 0.03;

bool genSolid(vec3 worldPos)
{
    // height map with 3 levels of noise
    float h = snoise(vec2(worldPos.x, worldPos.z) * noiseScale) * 2;
    h += snoise(vec2(worldPos.x, worldPos.z) * noiseScale * 2) * 0.5;
    h += snoise(vec2(worldPos.x, worldPos.z) * noiseScale * 4) * 0.25;
    h = h * 0.5 + 0.5;
    h *= 4;
    h += 30;

    // generate caves
    float cave = snoise(worldPos * noiseScale);
    return worldPos.y < h && cave > 0;
}

uint getMaterialForPosition(vec3 worldPos, bool solidAbove) 
{
    // Add some variation in materials based on depth and noise
    float depth = 30.0 - worldPos.y;
    float materialNoise = snoise(worldPos * noiseScale * 5);
    
    if (!solidAbove) {
        // Surface material (grass)
        return 1;
    }
    else if (depth < 5.0 || materialNoise > 0.7) {
        // Near surface or random pockets (dirt)
        return 2;
    }
    else if (materialNoise < -0.7) {
        // Random ore deposits
        return 4;
    }
    else {
        // Default stone
        return 3;
    }
}

void main()
{
    // Get our position in the block grid
    ivec3 blockPos = ivec3(gl_WorkGroupID) % BLOCK_SIZE;
    // Get our position in the voxel grid
    ivec3 voxelPos = ivec3(gl_LocalInvocationID) % BLOCK_SIZE;

    for (int chunk = chunkIndex; chunk < chunkIndex + chunkCount; chunk++)
    {
        if (chunk >= GRID_SIZE * GRID_SIZE * GRID_SIZE) break;

        ivec3 chunkPos = getChunkPos(chunk);
        vec3 worldPos = composePosition(chunkPos, blockPos, voxelPos);
        
        // Generate terrain
        bool solid = genSolid(worldPos);
        
        if (solid)
        {
            int blockLocalIndex = getLocalIndex(blockPos);
            int voxelLocalIndex = getLocalIndex(voxelPos);
            int blockIndex = globalIndex(chunk, blockLocalIndex);

            // Set the occupancy bits
            atomicSetChunkBit(chunk, blockLocalIndex);
            atomicSetBlockBit(blockIndex, voxelLocalIndex);

            // Check conditions for material selection
            bool solidAbove = genSolid(worldPos + vec3(0, VOXEL_SIZE * 6, 0));
            uint materialId = getMaterialForPosition(worldPos, solidAbove);

            // Set the material using our new system
            setVoxelMaterial(blockIndex, voxelLocalIndex, materialId);
            memoryBarrier();
        }
    }
}