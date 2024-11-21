#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform int pass; // 0 = X, 1 = Y, 2 = Z
uniform int currentChunk;

#include "chunk_common.glsl"

void main() {
    ChunkData chunk = chunks[currentChunk];
    ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
    int index = getDataIndex(pos, currentChunk);
    
    // Early exit for solid blocks or invalid positions
    if (index == -1 || materials[index] != 0) {
        if (index != -1) {
            lightLevels[index] = vec4(0);
        }
        return;
    }
    
    vec4 sum = vec4(0);
    int count = 0;
    
    // Define the axis of blur based on the current pass
    ivec3 direction = (pass == 0) ? ivec3(1, 0, 0) : 
                     (pass == 1) ? ivec3(0, 1, 0) : 
                                  ivec3(0, 0, 1);
    
    // Sample neighboring voxels along the current axis
    for (int offset = -3; offset <= 3; offset++) {
        ivec3 samplePos = pos + direction * offset;
        int sampleIndex = getDataIndexSafe(samplePos, currentChunk);
        
        if (sampleIndex != -1 && materials[sampleIndex] == 0) {
            sum += lightLevels[sampleIndex];
            count++;
        }
    }
    
    // Write result directly back to light levels buffer
    lightLevels[index] = (count > 0) ? sum / float(count) : vec4(0);
}