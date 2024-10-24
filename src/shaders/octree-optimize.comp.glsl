// #version 460

// layout(local_size_x = 512) in;

// layout(std430, binding = 0) buffer PositionsBuffer {
//     vec4 positions[];
// };

// layout(std430, binding = 1) buffer MaterialsBuffer {
//     int materials[];
// };

// layout(location = 0) uniform int maxDepth;
// layout(location = 1) uniform float voxelSize;
// layout(location = 2) uniform int currentLevel;
// layout(location = 3) uniform int chunkSize;

// layout(std430, binding = 4) buffer ChangesBuffer {
//     int changes[];
// };

// const int MATERIAL_AIR = 0;

// bool isLeaf(vec4 posSize) {
//     return posSize.w < 0.0;
// }

// int getNodeDepth(int index) {
//     if (index == 0) return 0;
//     return int(floor(log2(float(index * 7 + 1)) / log2(8.0))) + 1;
// }

// float min3(vec3 x)
// {
//     return min(min(x.x, x.y), x.z);
// }

// float max3(vec3 x)
// {
//     return max(max(x.x, x.y), x.z);
// }

// int getIndexFromPosition(vec3 position)
// {
//     if (min3(position) < 0 || max3(position) >= chunkSize) return -1;

//     int result =  int(position.x) + int(position.y) * int(chunkSize) + int(position.z) * int(chunkSize) * int(chunkSize);
//     return result;
// }

// void main() {
//     uint index = gl_GlobalInvocationID.x;
//     if (index >= positions.length()) return;
//     if (index == 0) return; // Skip root node
    
//     int depth = getNodeDepth(int(index)); 
//     if (depth != currentLevel) return;
    
//     vec4 posSize = positions[index];
//     if (posSize.w == 0.0 || posSize.w == voxelSize) return; // Skip uninitialized nodes
    
//     vec3 pos = posSize.xyz;
//     float size = abs(posSize.w);
    
//     if (size <= voxelSize) return;

//     int firstChild = int(index) * 8 + 1;
//     float childSize = size * 0.5;


//     // Check children state
//     bool hasNonAir = false;
//     bool allChildrenAreLeaves = true;
    
//     for (int i = 0; i < 8; i++) {
//         int childIndex = firstChild + i;
//         if (childIndex >= positions.length()) continue;
        
//         vec4 childPosSize = positions[childIndex];
//         if (childPosSize.w == 0.0) continue;
        
//         if (!isLeaf(childPosSize)) {
//             allChildrenAreLeaves = false;
//         } else
//         {
//             int childDataIndex = getIndexFromPosition(childPosSize.xyz / voxelSize);
//             int childMaterial = materials[childDataIndex];

//             // Check if this referenced voxel has non-air material
//             if (childMaterial != MATERIAL_AIR) {
//                 hasNonAir = true;
//             }
//         }
//     }
    
//     if (isLeaf(posSize) && (!allChildrenAreLeaves || hasNonAir))
//     {
//         // If it's marked as a leaf but has non-leaf children, make it a node
//         positions[index].w = size;
//         atomicAdd(changes[0], 1);
//     }
//     else if (allChildrenAreLeaves && !hasNonAir) {
//         // If all children are leaves and empty, make this a leaf
//         positions[index].w = -size;
//         atomicAdd(changes[0], 1);
//     }
    
// }

#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer PositionsBuffer {
    vec4 positions[];
};

layout(std430, binding = 1) buffer MaterialsBuffer {
    int materials[];
};

layout(location = 0) uniform int maxDepth;
layout(location = 1) uniform float voxelSize;
layout(location = 2) uniform int currentLevel;
layout(location = 3) uniform int chunkSize;


// Shared memory for caching node data within a workgroup
shared vec4 sharedPositions[64];
shared int sharedMaterials[512]; // Cache for materials data

const int MATERIAL_AIR = 0;
const uint LEAF_MASK = 0x80000000;
const float EPSILON = 1e-6;

// Optimized helper functions using bit operations
bool isLeaf(vec4 posSize) {
    return posSize.w < -EPSILON;
}

// Fast integer log2 using bit operations
int fastLog2(int value) {
    return findMSB(value);
}

int getNodeDepth(int index) {
    if (index == 0) return 0;
    return fastLog2((index * 7 + 1)) / 3 + 1;
}

// Optimized position to index calculation using bit shifts
int getIndexFromPosition(vec3 position) {
    ivec3 ipos = ivec3(position);
    if (any(lessThan(ipos, ivec3(0))) || any(greaterThanEqual(ipos, ivec3(chunkSize)))) 
        return -1;
    
    return ipos.x + (ipos.y << int(log2(float(chunkSize)))) + 
           (ipos.z << int(2.0 * log2(float(chunkSize))));
}

// Process a group of nodes in parallel
void processNodeGroup(uint localId, uint groupSize) {
    // Load node data into shared memory
    if (localId < groupSize) {
        uint globalIndex = gl_WorkGroupID.x * groupSize + localId;
        if (globalIndex < positions.length()) {
            sharedPositions[localId] = positions[globalIndex];
        }
    }
    
    barrier();
    memoryBarrierShared();
    
    // Process nodes in parallel within the workgroup
    if (localId < groupSize) {
        uint globalIndex = gl_WorkGroupID.x * groupSize + localId;
        if (globalIndex >= positions.length() || globalIndex == 0) return;
        
        int depth = getNodeDepth(int(globalIndex));
        if (depth != currentLevel) return;
        
        vec4 posSize = sharedPositions[localId];
        if (abs(posSize.w) < EPSILON || abs(posSize.w - voxelSize) < EPSILON) return;
        
        vec3 pos = posSize.xyz;
        float size = abs(posSize.w);
        
        if (size <= voxelSize) return;

        // Calculate child indices using bit operations
        int firstChild = int(globalIndex) * 8 + 1;
        float childSize = size * 0.5;
        
        // Use bit field to track child states
        uint childStates = 0;
        uint leafCount = 0;
        uint nonAirCount = 0;
        
        // Process children in parallel within the workgroup
        for (int i = 0; i < 8; i++) {
            int childIndex = firstChild + i;
            if (childIndex >= positions.length()) continue;
            
            vec4 childPosSize = positions[childIndex];
            if (abs(childPosSize.w) < EPSILON) continue;
            
            if (isLeaf(childPosSize)) {
                leafCount++;
                int childDataIndex = getIndexFromPosition(childPosSize.xyz / voxelSize);
                if (childDataIndex >= 0) {
                    int material = materials[childDataIndex];
                    nonAirCount += uint(material != MATERIAL_AIR);
                }
            }
            
            childStates |= (1u << i);
        }
        
        // Update node state based on accumulated results
        bool shouldBeLeaf = (leafCount == bitCount(childStates)) && (nonAirCount == 0);
        bool currentlyLeaf = isLeaf(posSize);
        
        if (currentlyLeaf != shouldBeLeaf) {
            vec4 newPosSize = posSize;
            newPosSize.w = shouldBeLeaf ? -size : size;
            positions[globalIndex] = newPosSize;
        }
    }
}

void main() {
    processNodeGroup(gl_LocalInvocationID.x, gl_WorkGroupSize.x);
}