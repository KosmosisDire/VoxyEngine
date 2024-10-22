using System.Numerics;
using Engine;
using ILGPU.Util;
namespace VoxelEngine;


public class OctreeGenerator
{
    public readonly float voxelSize = 0.25f;
    public int NodeCount { get; private set; }
    public int LeafCount { get; private set; }
    public int ChunkSize { get; private set; }
    public int ChunkWidth { get; private set; }

    public Float4[] positions; // xyz: position, w: size

    public int[] materials;
    public float[] lightLevels;
    public Float4[] colors;

    public OctreeGenerator(int chunkSize)
    {
        ChunkWidth = (int)(chunkSize * voxelSize);
        ChunkSize = chunkSize;
        NodeCount = CalculateNodeCount(ChunkWidth);
        LeafCount = CalculateLeafCount(ChunkWidth);
        

        Console.WriteLine($"Node count: {NodeCount}");
        Console.WriteLine($"Leaf count: {LeafCount}");

        positions = new Float4[NodeCount + 1];

        materials = new int[LeafCount + 1];
        lightLevels = new float[LeafCount + 1];
        colors = new Float4[LeafCount + 1];
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
        BuildOctreeRecursive(new Vector3(chunkPosition.X, chunkPosition.Y, chunkPosition.Z), ChunkWidth);
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

    public int GetIndexFromPosition(Vector3 position)
    {
        position = position / voxelSize;
        return (int)(position.X + position.Y * ChunkSize + position.Z * ChunkSize * ChunkSize);
    }

    private void BuildOctreeRecursive(Vector3 position, float size, int index = 0)
    {
        positions[index] = new Float4(position.X, position.Y, position.Z, size);

        float childSize = size / 2.0f;
        if (childSize < voxelSize)
        {
            int dataIndex = GetIndexFromPosition(position);
            materials[dataIndex] = GetNoiseAt(position);
            lightLevels[dataIndex] = 0.0f;
            colors[dataIndex] = new Float4(1, 1, 1, 1);
            positions[index] = new Float4(position.X, position.Y, position.Z, -size);
            return;
        }

        bool allEmpty = true;
        for (int i = 0; i < 8; i++)
        {
            Vector3 offset = childOffsets[i] * childSize;
            Vector3 childPosition = position + offset;

            var childIndex = index * 8 + 1 + i;
            BuildOctreeRecursive(childPosition, childSize, childIndex);

            if (positions[childIndex].W > 0)
            {
                allEmpty = false;
            }
            else
            {
                int childDataIndex = GetIndexFromPosition(childPosition);
                if (materials[childDataIndex] > 0)
                {
                    allEmpty = false;
                }
            }

            // // also make it not all empty if one of the children is a leaf which is adjacent to a solid leaf
            // if (allEmpty)
            // {
            //     for (int j = 0; j < 8; j++)
            //     {
            //         Vector3 adjacentOffset = childOffsets[j] * childSize;
            //         Vector3 adjacentPosition = childPosition + adjacentOffset;
            //         int adjacentDataIndex = GetIndexFromPosition(adjacentPosition);
            //         if (adjacentDataIndex < 0 || adjacentDataIndex >= materials.Length)
            //         {
            //             continue;
            //         }
            //         if (materials[adjacentDataIndex] > 0)
            //         {
            //             allEmpty = false;
            //             break;
            //         }
            //     }
            // }
        }

        if (allEmpty)
        {
            positions[index] = new Float4(position.X, position.Y, position.Z, -size);
        }
    }

    private int GetNoiseAt(Vector3 position)
    {
        // var chunkCenter = new Vector3(ChunkWidth / 2, ChunkWidth / 2, ChunkWidth / 2);
        // var distance = Vector3.Distance(position, chunkCenter);
        // if (distance > ChunkWidth / 2)
        // {
        //     return 0;
        // }

        float nx = position.X * 0.1f;
        float ny = position.Y * 0.1f;
        float nz = position.Z * 0.1f;
        var noise = Perlin.Noise(nx, ny, nz);
        return noise > 0.001f ? 1 : 0;
    }
    

}