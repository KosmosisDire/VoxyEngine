using System.Collections.Concurrent;
using System.Numerics;
namespace VoxelEngine;

public class OctreeGenerator
{
    public readonly float voxelSize = 0.5f;
    public int NodeCount { get; private set; }
    public int LeafCount { get; private set; }
    public int ChunkSize { get; private set; }
    public int ChunkWidth { get; private set; }

    public Vector4[] positions;
    public int[] materials;
    public Vector4[] lightLevels;
    public Vector4[] colors;

    private readonly TerrainGenerator generator;
    
    public OctreeGenerator(int chunkSize)
    {
        ChunkWidth = (int)(chunkSize * voxelSize);
        ChunkSize = chunkSize;
        NodeCount = CalculateNodeCount(ChunkWidth);
        LeafCount = CalculateLeafCount(ChunkWidth);

        Console.WriteLine($"Node count: {NodeCount}");
        Console.WriteLine($"Leaf count: {LeafCount}");

        // Initialize arrays
        positions = new Vector4[NodeCount];
        materials = new int[LeafCount];
        lightLevels = new Vector4[LeafCount];
        colors = new Vector4[LeafCount];

        generator = new TerrainGenerator(seed: 12);
    }

    private int CalculateNodeCount(int chunkWidth)
    {
        if ((chunkWidth & (chunkWidth - 1)) != 0)
        {
            throw new ArgumentException("Chunk size must be a power of 2");
        }

        int chunkSize = (int)(chunkWidth / voxelSize);
        int nodeCount = (int)(((Math.Pow(8, Math.Log(chunkSize, 2) + 1) - 1) / 7));
        return nodeCount;
    }

    private int CalculateLeafCount(int chunkWidth)
    {
        int chunkSize = (int)(chunkWidth / voxelSize);
        return chunkSize * chunkSize * chunkSize;
    }

    public void GenerateOctree(Vector3 chunkPosition)
    {
        var startTime = DateTime.Now;
        int nodeCounter = 0;
        int leafCounter = 0;

        // Divide the chunk into subregions for parallel processing
        int subregionSize = ChunkWidth / 2; // Split into 8 major subregions
        var tasks = new Task[8];

        for (int i = 0; i < 8; i++)
        {
            Vector3 offset = childOffsets[i] * subregionSize;
            Vector3 subregionPos = chunkPosition + offset;
            int baseIndex = i + 1;  // Root is at 0, children start at 1

            tasks[i] = Task.Run(() =>
            {
                BuildOctreeSubregion(subregionPos, subregionSize, baseIndex, ref nodeCounter, ref leafCounter);
            });
        }

        // Set root node
        positions[0] = new Vector4(chunkPosition.X, chunkPosition.Y, chunkPosition.Z, ChunkWidth);
        Interlocked.Increment(ref nodeCounter);

        Task.WaitAll(tasks);

        var endTime = DateTime.Now;
        Console.WriteLine($"Parallel octree generation took {(endTime - startTime).TotalMilliseconds} ms");
        Console.WriteLine($"Octree nodes: {nodeCounter}, leaves: {leafCounter}");
    }

    private static readonly Vector3[] childOffsets = [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 0, 1),
        new Vector3(1, 0, 1),
        new Vector3(0, 1, 1),
        new Vector3(1, 1, 1)
    ];

    private void BuildOctreeSubregion(Vector3 position, float size, int index, ref int nodeCounter, ref int leafCounter)
    {
        positions[index] = new Vector4(position.X, position.Y, position.Z, size);
        Interlocked.Increment(ref nodeCounter);

        float childSize = size / 2.0f;
        if (childSize < voxelSize)
        {
            int dataIndex = GetIndexFromPosition(position);
            var (material, color) = generator.GetTerrainAt(position, ChunkWidth);
            
            materials[dataIndex] = material;
            colors[dataIndex] = color;
            lightLevels[dataIndex] = new Vector4(0);
            positions[index] = new Vector4(position.X, position.Y, position.Z, -size);
            
            Interlocked.Increment(ref leafCounter);
            return;
        }

        for (int i = 0; i < 8; i++)
        {
            Vector3 offset = childOffsets[i] * childSize;
            Vector3 childPosition = position + offset;
            int childIndex = index * 8 + 1 + i;

            BuildOctreeSubregion(childPosition, childSize, childIndex, ref nodeCounter, ref leafCounter);
        }
    }

    public int GetIndexFromPosition(Vector3 position)
    {
        var divPosition = new Vector3(position.X / voxelSize, position.Y / voxelSize, position.Z / voxelSize);
        uint xOff = (uint)divPosition.X;
        uint yOff = (uint)divPosition.Y * (uint)ChunkSize;
        uint zOff = (uint)divPosition.Z * (uint)ChunkSize * (uint)ChunkSize;
        return (int)(xOff + yOff + zOff);
    }
}