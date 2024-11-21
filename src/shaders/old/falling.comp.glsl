#version 460

layout(local_size_x = 512) in;

layout(std430, binding = 5) buffer MoveFlagBuffer {
    int moveFlags[];
};

// Required uniforms
uniform int chunkSize;
uniform float voxelSize;
uniform float deltaTime;
uniform uint frame;
uniform vec3 gravityDir;
uniform vec3 currentChunkPosition;

#include "chunk_common.glsl"

bool tryMove(vec3 sourcePos, vec3 targetPos) {
    int sourceChunkIndex = findChunkIndex(sourcePos);
    int targetChunkIndex = findChunkIndex(targetPos);
    
    if (sourceChunkIndex == -1 || targetChunkIndex == -1) return false;
    
    ChunkData sourceChunk = chunks[sourceChunkIndex];
    ChunkData targetChunk = chunks[targetChunkIndex];
    
    int sourceLocalIndex = getChunkLocalIndex(sourcePos, sourceChunk.position);
    int targetLocalIndex = getChunkLocalIndex(targetPos, targetChunk.position);
    
    if (sourceLocalIndex == -1 || targetLocalIndex == -1) return false;
    
    int sourceOffset = sourceChunk.bufferOffset;
    int targetOffset = targetChunk.bufferOffset;
    
    if (materials[targetOffset + targetLocalIndex] != 0) return false;
    
    // Try to atomically claim the target position
    int original = atomicCompSwap(moveFlags[targetOffset + targetLocalIndex], 0, sourceLocalIndex + 1);
    
    if (original == 0) {
        // Successfully claimed the target position
        materials[targetOffset + targetLocalIndex] = materials[sourceOffset + sourceLocalIndex];
        colors[targetOffset + targetLocalIndex] = colors[sourceOffset + sourceLocalIndex];
        lightLevels[targetOffset + targetLocalIndex] = lightLevels[sourceOffset + sourceLocalIndex];
        
        materials[sourceOffset + sourceLocalIndex] = 0;
        colors[sourceOffset + sourceLocalIndex] = vec4(0);
        lightLevels[sourceOffset + sourceLocalIndex] = vec4(0);
        
        return true;
    }
    
    return false;
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= chunkSize * chunkSize * chunkSize) return;
    
    // Calculate position in current chunk
    vec3 localPos = vec3(
        float(index % chunkSize),
        float((index / chunkSize) % chunkSize),
        float(index / (chunkSize * chunkSize))
    ) * voxelSize;
    
    vec3 worldPos = localPos + currentChunkPosition;
    
    int currentChunkIndex = findChunkIndex(worldPos);
    if (currentChunkIndex == -1) return;
    
    ChunkData chunk = chunks[currentChunkIndex];
    int localIndex = getChunkLocalIndex(worldPos, chunk.position);
    if (localIndex == -1) return;
    
    int materialOffset = chunk.bufferOffset;
    
    // Only process non-empty voxels
    if (materials[materialOffset + localIndex] == 0) return;
    
    // Get movement directions based on gravity
    vec3 primaryDir = normalize(gravityDir);
    vec3 targetPos = worldPos + primaryDir * voxelSize;
    
    if (tryMove(worldPos, targetPos)) return;
    
    // Try diagonal movements if primary movement fails
    uint seed = uint(index) + frame * 1237;
    float spreadRandom = random(seed);
    
    vec3 right = normalize(cross(primaryDir, vec3(0, 1, 0)));
    vec3 up = normalize(cross(right, primaryDir));
    
    vec3 diagonals[4];
    diagonals[0] = primaryDir + right;
    diagonals[1] = primaryDir - right;
    diagonals[2] = primaryDir + up;
    diagonals[3] = primaryDir - up;
    
    int startDiagonal = int(spreadRandom * 4.0);
    
    for (int i = 0; i < 4; i++) {
        int diagIndex = (startDiagonal + i) % 4;
        vec3 diagPos = worldPos + normalize(diagonals[diagIndex]) * voxelSize;
        
        if (tryMove(worldPos, diagPos)) return;
    }
}