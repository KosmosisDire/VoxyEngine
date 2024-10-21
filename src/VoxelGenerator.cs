using System.Numerics;
using Engine;
using ILGPU.Util;
namespace VoxelEngine;


public class OctreeGenerator
{
    private const int MIN_SIZE = 1;
    public int NodeCount { get; private set; }
    public int LeafCount { get; private set; }
    public int ChunkSize { get; private set; }

    public Float3[] positions; // xyz: position, w: size
    public float[] sizes;
    public int[] indices; // index into the position indexed array so we can find neighbors

    public int[] materials;
    public int[] isLeaf;
    public float[] lightLevels;
    public Float3[] colors;

    public OctreeGenerator()
    {
    }

    public void GenerateOctree(int chunkSize, Vector3 chunkPosition)
    {
        if ((chunkSize & (chunkSize - 1)) != 0)
        {
            throw new ArgumentException("Chunk size must be a power of 2");
        }

        ChunkSize = chunkSize;
        NodeCount = (int)((Math.Pow(8, Math.Log(chunkSize, 2) + 1) - 1) / 7);
        LeafCount = chunkSize * chunkSize * chunkSize;

        Console.WriteLine($"Node count: {NodeCount}");
        Console.WriteLine($"Leaf count: {LeafCount}");

        positions = new Float3[NodeCount];
        sizes = new float[NodeCount];
        indices = new int[NodeCount];

        materials = new int[LeafCount];
        isLeaf = new int[LeafCount];
        lightLevels = new float[LeafCount];
        colors = new Float3[LeafCount];
    }

    private static readonly Float3[] childOffsets = [
        new Float3(0, 0, 0),
        new Float3(1, 0, 0),
        new Float3(0, 1, 0),
        new Float3(1, 1, 0),
        new Float3(0, 0, 1),
        new Float3(1, 0, 1),
        new Float3(0, 1, 1),
        new Float3(1, 1, 1)
    ];

    public int GetIndexFromPosition(Float3 position)
    {
        return (int)(position.X + position.Y * ChunkSize + position.Z * ChunkSize * ChunkSize);
    }

    private void BuildOctreeRecursive(Float3 position, float size, int index = 0)
    {
        positions[index] = position;
        sizes[index] = size;

        int dataIndex = GetIndexFromPosition(position);
        indices[index] = dataIndex;

        float childSize = size / 2.0f;

        if (childSize < MIN_SIZE)
        {
            materials[dataIndex] = GetNoiseAt(position);
            isLeaf[dataIndex] = 1;
            return;
        }

        for (int i = 0; i < 8; i++)
        {
            Float3 offset = childOffsets[i] * childSize;
            Float3 childPosition = position + offset;

            var childIndex = index * 8 + 1 + i;
            BuildOctreeRecursive(childPosition, childSize, childIndex);
        }
    }

    public List<int> SetMaterial(Vector3 position, int material)
    {
        // Track modified nodes
        var modifiedNodes = new List<int>();

        // Early validation
        if (NodeCount == 0 || posAndSize == null || nodeData == null)
            throw new InvalidOperationException("Octree not initialized");

        // Stack to store the path for parent updates
        Span<int> nodePath = stackalloc int[32]; // Maximum depth for a reasonable octree
        int pathLength = 0;

        // Start from root
        int currentIndex = 0;
        Vector3 currentPosition = new(posAndSize[0].X, posAndSize[0].Y, posAndSize[0].Z);
        int currentSize = (int)posAndSize[0].W;

        // Traverse down to leaf while splitting leaf nodes if necessary
        while (currentSize > 1)
        {
            nodePath[pathLength++] = currentIndex;
            int childSize = currentSize / 2;

            // Check if we need to split this node
            if (nodeData[currentIndex].Y == 1)
            {
                int currentMaterial = (int)nodeData[currentIndex].X;
                // Split the node by creating 8 children with the same material
                for (int i = 0; i < 8; i++)
                {
                    int childIndex = currentIndex * 8 + 1 + i;
                    var oldData = nodeData[childIndex];
                    nodeData[childIndex] = new Vector4(currentMaterial, 1, 0, 0);
                    Vector3 childPos = currentPosition + childOffsets[i] * childSize;
                    posAndSize[childIndex] = new Vector4(childPos, childSize);

                    if (oldData != nodeData[childIndex])
                    {
                        modifiedNodes.Add(childIndex);
                    }
                }

                // Convert current node from leaf to internal
                var oldNodeData = nodeData[currentIndex];
                nodeData[currentIndex] = new Vector4(0, 0, 0, 0); // Internal node with no material

                if (oldNodeData != nodeData[currentIndex])
                {
                    modifiedNodes.Add(currentIndex);
                }
            }

            // Find appropriate child octant
            int childOctant = GetChildOctant(position, currentPosition, childSize);
            currentIndex = currentIndex * 8 + 1 + childOctant;

            // Update position and size for next iteration
            currentPosition += childOffsets[childOctant] * childSize;
            currentSize = childSize;
        }

        // Set the material at the leaf node
        var oldLeafData = nodeData[currentIndex];
        nodeData[currentIndex] = new Vector4(material, 1, 0, 0);

        if (oldLeafData != nodeData[currentIndex])
        {
            modifiedNodes.Add(currentIndex);
        }

        for (int i = pathLength - 1; i >= 0; i--)
        {
            int parentIndex = nodePath[i];
            int firstChildIndex = parentIndex * 8 + 1;

            var oldParentData = nodeData[parentIndex];

            // Check children state
            int firstChildMaterial = (int)nodeData[firstChildIndex].X;
            bool allSame = true;
            bool allLeaves = true;

            // Check all children
            for (int child = 0; child < 8; child++)
            {
                var childData = nodeData[firstChildIndex + child];
                if (childData.Y != 1)
                {
                    allLeaves = false;
                    break;
                }
                if ((int)childData.X != firstChildMaterial)
                {
                    allSame = false;
                    break;
                }
            }

            // Determine new parent state
            Vector4 newParentData;
            if (allLeaves && allSame)
            {
                // All children are leaves with same material - parent becomes a leaf
                newParentData = new Vector4(firstChildMaterial, 1, 0, 0);
            }
            else
            {
                // Children are different or not all leaves - parent must be internal
                newParentData = new Vector4(0, 0, 0, 0);
            }

            // Update parent if changed
            if (oldParentData != newParentData)
            {
                nodeData[parentIndex] = newParentData;
                modifiedNodes.Add(parentIndex);
            }
        }


        // Sort and remove duplicates
        modifiedNodes.Sort();
        return modifiedNodes.Distinct().ToList();
    }

    // Helper method to determine which child octant contains the position
    private int GetChildOctant(Vector3 position, Vector3 nodePosition, int childSize)
    {
        int x = position.X >= nodePosition.X + childSize ? 1 : 0;
        int y = position.Y >= nodePosition.Y + childSize ? 2 : 0;
        int z = position.Z >= nodePosition.Z + childSize ? 4 : 0;
        return x | y | z;
    }

    public int GetVoxelMaterial(Vector3 position)
    {
        // Early validation - return 0 (empty) for out of bounds
        if (NodeCount == 0 || posAndSize == null || nodeData == null)
        {
            return 0;
        }

        // Start at root
        int currentIndex = 0;
        var rootPos = posAndSize[0];

        // Early bounds check against root node
        float size = rootPos.W;
        if (position.X < rootPos.X || position.Y < rootPos.Y || position.Z < rootPos.Z ||
            position.X >= rootPos.X + size || position.Y >= rootPos.Y + size || position.Z >= rootPos.Z + size)
        {
            return 0;
        }

        // While not at a leaf node
        while (nodeData[currentIndex].Y != 1) // .Y is isLeaf
        {
            float childSize = size * 0.5f;

            // Calculate child octant using bit operations
            int childOffset = ((position.X >= rootPos.X + childSize ? 1 : 0) |
                             (position.Y >= rootPos.Y + childSize ? 2 : 0) |
                             (position.Z >= rootPos.Z + childSize ? 4 : 0));

            // Move to child
            currentIndex = currentIndex * 8 + 1 + childOffset;

            // Update position reference for next iteration
            if ((childOffset & 1) != 0) rootPos.X += childSize;
            if ((childOffset & 2) != 0) rootPos.Y += childSize;
            if ((childOffset & 4) != 0) rootPos.Z += childSize;

            size = childSize;

            // Bounds check - if we've gone beyond valid indices
            if (currentIndex >= NodeCount)
            {
                return 0;
            }
        }

        // Return material from leaf node
        return (int)nodeData[currentIndex].X;
    }

    public List<int> GetSolidNodes()
    {
        var solidNodes = new List<int>();
        for (int i = 0; i < NodeCount; i++)
        {
            if (nodeData[i].X != 0)
            {
                solidNodes.Add(i);
            }
        }
        return solidNodes;
    }

    public List<Vector3> GetSolidNodePositions()
    {
        var solidNodes = GetSolidNodes();
        var solidNodePositions = new List<Vector3>();
        foreach (var nodeIndex in solidNodes)
        {
            var node = posAndSize[nodeIndex];
            solidNodePositions.Add(new Vector3(node.X, node.Y, node.Z));
        }
        return solidNodePositions;
    }

    private int GetNoiseAt(Float3 position)
    {
        float nx = position.X * 0.1f;
        float ny = position.Y * 0.1f;
        float nz = position.Z * 0.1f;
        var noise = Perlin.Noise(nx, ny, nz);
        return noise > 0.001f ? 1 : 0;
        return 0;
    }
    
    public void CalculateLightLevels(Vector3 lightDirection, int numBounces)
    {
        using (var lightingCalculator = new GPULightingCalculator(this))
        lightingCalculator.CalculateLighting(lightDirection, numBounces);
    }


}