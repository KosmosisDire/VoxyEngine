#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 1) readonly buffer MaterialsBuffer {
    int materials[];
};

layout(std430, binding = 2) buffer LightLevelsBuffer {
    vec4 lightLevels[];
};

layout(location = 0) uniform int chunkSize;
layout(location = 1) uniform float voxelSize;
layout(location = 2) uniform int pass; // 0 = X, 1 = Y, 2 = Z

int getIndex(ivec3 pos) {
    if (pos.x < 0 || pos.x >= chunkSize ||
        pos.y < 0 || pos.y >= chunkSize ||
        pos.z < 0 || pos.z >= chunkSize) {
        return -1;
    }
    return pos.x + pos.y * chunkSize + pos.z * chunkSize * chunkSize;
}

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
    int index = getIndex(pos);
    
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
    for (int offset = -2; offset <= 2; offset++) {
        ivec3 samplePos = pos + direction * offset;
        int sampleIndex = getIndex(samplePos);
        
        if (sampleIndex != -1 && materials[sampleIndex] == 0) {
            sum += lightLevels[sampleIndex];
            count++;
        }
    }
    
    // Write result directly back to light levels buffer
    lightLevels[index] = (count > 0) ? sum / float(count) : vec4(0);
}