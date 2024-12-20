#version 460

#include "common.glsl"
#include "lygia/generative/snoise.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Uniforms for chunk generation
uniform int chunkIndex;
uniform int chunkCount;
const float noiseScale = 0.3;

bool genSolid(vec3 worldPos)
{
    // height map with 3 levels of noise
    float h = snoise(vec2(worldPos.x, worldPos.z) * noiseScale * 0.1) * 2;
    h += snoise(vec2(worldPos.x, worldPos.z) * noiseScale * 0.5) * 0.5;
    h += snoise(vec2(worldPos.x, worldPos.z) * noiseScale * 0.9) * 0.25;
    h = h * 0.5 + 0.5;
    h *= 1;
    h += 6;

    // generate caves
    float cave = snoise(worldPos * noiseScale);
    return worldPos.y < h && cave > 0;
}

uint getMaterialForPosition(vec3 worldPos, bool solidAbove) 
{
    // Add some variation in materials based on depth and noise
    float depth = 30.0 - worldPos.y;
    float materialNoise = snoise(worldPos * noiseScale * 5) * 0.7 + snoise(worldPos * noiseScale * 10) * 0.2 + snoise(worldPos * noiseScale * 20) * 0.1;

    // add tiny random noise to material noise
    materialNoise += snoise(worldPos * noiseScale * 100) * 0.1;
    
    if (!solidAbove) {
        // Surface material (grass)
        return materialNoise > 0.35 ? 9 : 2;
    }
    else if (depth < 5.0 || materialNoise > 0.55) {
        // Near surface or random pockets (dirt)
        return 1;
    }
    else if (materialNoise < -0.55) {
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
        // vec3 worldBlockPos = composePosition(chunkPos, blockPos, ivec3(voxelPos.x >= 5 ? 5 : 0, voxelPos.y >= 5 ? 5 : 0, voxelPos.z >= 5 ? 5 : 0));
        // vec3 worldBlockPos = composePosition(chunkPos, blockPos, ivec3(0));
        vec3 worldPos = composePosition(chunkPos, blockPos, voxelPos);

        // round to nearest 5 voxels to make the terrain more blocky
        vec3 worldBlockPos = vec3(
            floor(worldPos.x / (VOXEL_SIZE * 4.0)) * (VOXEL_SIZE * 4.0),
            floor(worldPos.y / (VOXEL_SIZE * 4.0)) * (VOXEL_SIZE * 4.0),
            floor(worldPos.z / (VOXEL_SIZE * 4.0)) * (VOXEL_SIZE * 4.0)
        );
        
        // Generate terrain
        bool solid = genSolid(worldBlockPos);
        
        if (solid)
        {
            int blockLocalIndex = getLocalIndex(blockPos);
            int voxelLocalIndex = getLocalIndex(voxelPos);
            int blockIndex = globalIndex(chunk, blockLocalIndex);

            // Set the occupancy bits
            atomicSetChunkBit(chunk, blockLocalIndex);
            atomicSetBlockBit(blockIndex, voxelLocalIndex);

            // Check conditions for material selection
            bool solidAbove = genSolid(worldBlockPos + vec3(0, VOXEL_SIZE * 4, 0));
            uint materialId = getMaterialForPosition(worldPos, solidAbove);

            // Set the material
            int voxelIndex = globalIndex(blockIndex, voxelLocalIndex);
            setMaterial(voxelIndex, materialId);
        }
    }
}