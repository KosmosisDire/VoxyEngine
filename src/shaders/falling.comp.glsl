#version 460

layout(local_size_x = 512) in;

// Input buffers
layout(std430, binding = 1) buffer MaterialsBuffer {
    int materials[];
};

layout(std430, binding = 2) buffer LightLevelsBuffer {
    vec4 lightLevels[];
};

layout(std430, binding = 3) buffer ColorsBuffer {
    vec4 colors[];
};

layout(std430, binding = 4) buffer MoveFlagBuffer {
    int moveFlags[];
};

// Uniforms
layout(location = 0) uniform int chunkSize;
layout(location = 1) uniform float deltaTime;
layout(location = 2) uniform uint frame;
layout(location = 3) uniform vec3 gravityDir;

// Optimized position calculation using bit shifts
ivec3 getPosition(int index) {
    int shift = int(log2(float(chunkSize)));
    int mask = chunkSize - 1;
    return ivec3(
        index & mask,
        (index >> shift) & mask,
        index >> (shift * 2)
    );
}

// Fast index calculation using bit shifts when chunkSize is power of 2
int getIndex(ivec3 pos) {
    if (pos.x < 0 || pos.x >= chunkSize ||
        pos.y < 0 || pos.y >= chunkSize ||
        pos.z < 0 || pos.z >= chunkSize) {
        return -1;
    }
    return pos.x + (pos.y << int(log2(float(chunkSize)))) + 
           (pos.z << int(log2(float(chunkSize * chunkSize))));
}

// Random number generation
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

float random(uint seed) {
    return float(hash(seed)) / float(0xffffffffU);
}

// Atomic move attempt
bool tryAtomicMove(int sourceIndex, int targetIndex) {
    if (targetIndex < 0 || sourceIndex < 0) return false;
    if (materials[targetIndex] != 0) return false;
    
    // Try to atomically claim the target position
    int original = atomicCompSwap(moveFlags[targetIndex], 0, sourceIndex + 1);
    
    if (original == 0) {
        // Successfully claimed the target position
        // Now safely swap the materials
        int sourceMaterial = materials[sourceIndex];
        vec4 sourceColor = colors[sourceIndex];
        vec4 sourceLight = lightLevels[sourceIndex];
        
        materials[sourceIndex] = 0;
        colors[sourceIndex] = vec4(0);
        lightLevels[sourceIndex] = vec4(0);
        
        materials[targetIndex] = sourceMaterial;
        colors[targetIndex] = sourceColor;
        lightLevels[targetIndex] = sourceLight;
        
        return true;
    }
    
    return false;
}

// Get movement directions based on gravity
void getMovementDirections(out ivec3 primary, out ivec3[4] diagonals) {
    // Determine primary direction based on strongest gravity component
    vec3 absGrav = abs(gravityDir);
    float maxComp = max(max(absGrav.x, absGrav.y), absGrav.z);
    
    if (absGrav.x == maxComp) {
        primary = ivec3(sign(gravityDir.x), 0, 0);
        diagonals[0] = primary + ivec3(0, -1, 0);
        diagonals[1] = primary + ivec3(0, 1, 0);
        diagonals[2] = primary + ivec3(0, 0, -1);
        diagonals[3] = primary + ivec3(0, 0, 1);
    } else if (absGrav.y == maxComp) {
        primary = ivec3(0, sign(gravityDir.y), 0);
        diagonals[0] = primary + ivec3(-1, 0, 0);
        diagonals[1] = primary + ivec3(1, 0, 0);
        diagonals[2] = primary + ivec3(0, 0, -1);
        diagonals[3] = primary + ivec3(0, 0, 1);
    } else {
        primary = ivec3(0, 0, sign(gravityDir.z));
        diagonals[0] = primary + ivec3(-1, 0, 0);
        diagonals[1] = primary + ivec3(1, 0, 0);
        diagonals[2] = primary + ivec3(0, -1, 0);
        diagonals[3] = primary + ivec3(0, 1, 0);
    }
}


// Improved hash function for better distribution
uint wang_hash(uint seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// Decorrelate the processing order from the spatial position
uint getRandomizedIndex(uint index, uint maxSize) {
    // Use both frame number and index for variation
    uint seed = wang_hash(index ^ frame);
    
    // Generate a pseudo-random offset for this frame
    uint frameOffset = wang_hash(frame * 719393);
    
    // Combine index, frame offset, and a large prime for better distribution
    uint mixed = wang_hash(seed + frameOffset + 4294967291u);
    
    // Map back to valid range while maintaining good distribution
    return mixed % maxSize;
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= materials.length()) return;

    index = getRandomizedIndex(index, uint(materials.length()));
    
    // Only process sand particles
    if (materials[index] == 0) return;
    
    // Get current position
    ivec3 pos = getPosition(int(index));
    
    // Get movement directions
    ivec3 primaryDir;
    ivec3 diagonals[4];
    getMovementDirections(primaryDir, diagonals);
    
    // Compute checkerboard pattern based on position and frame
    // Use a 3D pattern that shifts each frame
    bool shouldProcess = ((pos.x + pos.y + pos.z + int(frame)) & 1) == 0;
    if (!shouldProcess) return;
    
    // Try primary direction first
    if (tryAtomicMove(int(index), getIndex(pos + primaryDir))) {
        return;
    }
    
    // If can't move in primary direction, try diagonals
    uint seed = uint(index) + frame * 1237;
    float spreadRandom = random(seed);
    
    // Randomize diagonal order
    int startDiagonal = int(spreadRandom * 4.0);
    
    // Try all diagonal directions in random order
    for (int i = 0; i < 4; i++) {
        int diagIndex = (startDiagonal + i) % 4;
        ivec3 targetPos = pos + diagonals[diagIndex];
        
        if (tryAtomicMove(int(index), getIndex(targetPos))) {
            return;
        }
    }
}