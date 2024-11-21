#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Required uniforms
uniform int maxDepth;
uniform int currentLevel;
uniform int chunkIndex;

#include "chunk_common.glsl"

const float EPSILON = 1e-6;

// Precompute bit patterns for depth calculation
const int[16] DEPTH_LOOKUP = int[16](
    0, 0, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3
);

// Fast integer log2 for depth calculation
int fastLog2(int value) {
    int result = 0;
    if (value > 0xFFFF) { result += 16; value >>= 16; }
    if (value > 0xFF)   { result += 8;  value >>= 8; }
    if (value > 0xF)    { result += 4;  value >>= 4; }
    return result + DEPTH_LOOKUP[value];
}

int getNodeDepth(int index) {
    if (index == 0) return 0;
    return fastLog2((index * 7 + 1)) / 3;
}

void processOctreeNode() {
    // Get thread index and current chunk data
    uint globalIndex = gl_GlobalInvocationID.x;
    ChunkData currentChunk = chunks[chunkIndex];
    
    // Early exit conditions
    if (currentChunk.nodeCount == 0 || 
        globalIndex >= currentChunk.nodeCount || 
        globalIndex <= 0 || 
        getNodeDepth(int(globalIndex)) != currentLevel) {
        return;
    }
    
    // Calculate node's position in buffer
    int nodeOffset = currentChunk.chunkIndex * currentChunk.nodeCount;
    int nodeIndex = int(globalIndex) + nodeOffset;
    NodeData data = unpackNodeData(nodeIndex);
    ivec3 pos = data.position;
    float size = float(data.size);
    bool isLeaf = data.data == 1;
    
    if (size <= voxelSize + EPSILON) {
        return;
    }
    
    // Process children
    int firstChild = int(globalIndex) * 8 + 1 + nodeOffset;
    int lastValidChild = min(firstChild + 8, nodeOffset + currentChunk.nodeCount);
    
    // Use bit fields for state tracking
    uint childState = 0u;  // bit 0: has valid child, bit 1: has non-leaf, bit 2: has non-empty
    
    // Unrolled child check loop for better performance
    for (int childIndex = firstChild; childIndex < lastValidChild; childIndex++) {
        NodeData childData = unpackNodeData(childIndex);
        ivec3 childPos = childData.position;
        float childSize = float(childData.size);
        bool childIsLeaf = childData.data == 1;

        if (childSize < EPSILON) continue;
        
        childState |= 1u;  // Has valid child
        
        if (!childIsLeaf) {
            childState |= 2u;  // Has non-leaf child
            break;  // Early exit if we found a non-leaf
        }
        
        // Material check only for leaf nodes
        ivec3 localPos = ivec3(vec3(childPos) / voxelSize);
        int dataIndex = getDataIndex(localPos, chunkIndex);
        VoxelMaterial material = unpackMaterial(dataIndex);
        
        if (dataIndex >= 0 && material.isSolid) {
            childState |= 4u;  // Has non-empty child
            break;  // Early exit if we found a non-empty child
        }
    }
    
    // Determine node state using bit operations
    bool shouldBeLeaf = ((childState & 1u) != 0u) &&    // Has valid children
                       ((childState & 2u) == 0u) &&    // All children are leaves
                       ((childState & 4u) == 0u);      // All children are empty
    
    // Update node if state needs to change
    if (isLeaf != shouldBeLeaf) {
        positions[nodeIndex] = packNodeData(NodeData(pos, int(size), shouldBeLeaf ? 1 : 0));
    }
}

void main() {
    processOctreeNode();
}