using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.OpenGL;
using Vector3i = OpenTK.Mathematics.Vector3i;

[StructLayout(LayoutKind.Sequential)]
public struct ChunkData
{
    public Vector3 position;       // Chunk world position
    public float padding1;      // Padding for alignment
    public int nodeCount;       // Number of nodes in the octree
    public int leafCount;       // Number of leaf nodes
    public int chunkIndex;    // Offset into the data buffers
    public int padding2;        // Padding for alignment
}

public struct NodeData
{
    public Vector3i position;  // 7 bits each (0-127)
    public int size;            // 7 bits (0-127)
    public int data;            // 4 bits (0-15)

    public NodeData(Vector3i position, int size, int data)
    {
        this.position = position;
        this.size = size;
        this.data = data;
    }

    public int PackNodeData()
    {
        return (position.X << 25) | (position.Y << 18) | (position.Z << 11) | (size << 4) | data;
    }
}

public struct VoxelMaterial
{
    Vector3 color; // 3 x 8
    bool isSolid; // 1
    int data; // 4

    public VoxelMaterial(Vector3 color, bool isSolid, int data)
    {
        this.color = color;
        this.isSolid = isSolid;
        this.data = data;
    }

    public int PackMaterial() 
    {
        return ((int)(color.X * 255) << 24) | ((int)(color.Y * 255) << 16) | ((int)(color.Z * 255) << 8) | ((isSolid ? 1 : 0) << 7) | data;
    }
}


public class Chunk
{
    public enum ChunkState
    {
        Uninitialized,
        Generating,
        AwaitFinalizedGeneration,
        AwaitFirstUpload,

        Dirty,
        Clean,
    }

    public ChunkGenerator Generator { get; set; }

    public Vector3 Position { get; private set; }
    public ChunkState State { get; set; }

    public int[] Positions { get; set; } // packed bytes (x, y, z, size)
    public int[] Materials { get; set; }
    public Vector4[] LightLevels { get; set; }

    public ConcurrentDictionary<Vector3, (int, Vector4)> chunkData;

    public int NodeCount { get; private set; }
    public int LeafCount { get; private set; }
    public int ChunkSize { get; private set; }
    public int ChunkWidth { get; private set; }
    public float VoxelSize { get; private set; }

    private int minChangedIndex = -1;
    private int maxChangedIndex = -1;

    public int ChunkIndex { get; private set; } = -1;

    public ChunkData Data => new ChunkData
    {
        position = new Vector3(Position.X, Position.Y, Position.Z),
        nodeCount = NodeCount,
        leafCount = LeafCount,
        chunkIndex = ChunkIndex
    };

    public Chunk(Vector3 position, int chunkSize, float voxelSize)
    {
        State = ChunkState.Uninitialized;
        Position = position;
        ChunkSize = chunkSize;
        VoxelSize = voxelSize;
        ChunkWidth = (int)(chunkSize * voxelSize);
        NodeCount = CalculateNodeCount(ChunkSize);
        LeafCount = CalculateLeafCount(ChunkSize);

        Positions = new int[NodeCount];
        Materials = new int[LeafCount];
        LightLevels = new Vector4[LeafCount];

        Generator = new ChunkGenerator(this);
    }

    public void Move(Vector3 position)
    {
        State = ChunkState.Uninitialized;
        Position = position;
    }

    public void BeginGenerate(int index)
    {
        ChunkIndex = index;
        State = ChunkState.Generating;
        chunkData = new ConcurrentDictionary<Vector3, (int, Vector4)>();
        TerrainGenerator.GenerateChunk(Position, ChunkSize, VoxelSize, chunkData);
        State = ChunkState.AwaitFinalizedGeneration;
    }

    public async Task FinalizeGeneration()
    {
        TerrainGenerator.ApplyFeatures(Position, chunkData);
        await Generator.GenerateOctree(chunkData);
        State = ChunkState.AwaitFirstUpload;
    }

    public unsafe void UploadAll(GL gl, int chunkIndex, uint positionsBuffer, uint materialsBuffer, uint lightLevelsBuffer, uint chunkDataBuffer)
    {
        ChunkIndex = chunkIndex;

        if (State == ChunkState.AwaitFirstUpload)
        {
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, positionsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(chunkIndex * NodeCount * sizeof(int)), (nuint)(sizeof(int) * NodeCount), (ReadOnlySpan<int>)Positions);
        }

        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialsBuffer);
        gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(chunkIndex * LeafCount * sizeof(int)), (nuint)(sizeof(int) * LeafCount), (ReadOnlySpan<int>)Materials);
        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, lightLevelsBuffer);
        gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(chunkIndex * LeafCount * sizeof(Vector4)), (nuint)(sizeof(Vector4) * LeafCount), (ReadOnlySpan<Vector4>)LightLevels);

        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, chunkDataBuffer);
        gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(chunkIndex  * sizeof(ChunkData)), (nuint)sizeof(ChunkData), (ReadOnlySpan<ChunkData>)new ChunkData[] { Data });

        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);

        State = ChunkState.Clean;

        Console.WriteLine($"Chunk at {Position} uploaded to index {ChunkIndex}");
    }

    public unsafe void Unload(GL gl, uint positionsBuffer, uint materialsBuffer, uint lightLevelsBuffer, uint chunkDataBuffer)
    {
        if (State != ChunkState.Clean)
        {
            Console.WriteLine("Chunk is not uploaded yet.");
            return;
        }

        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, positionsBuffer);
        gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(ChunkIndex * NodeCount * sizeof(Vector4)), (nuint)(sizeof(Vector4) * NodeCount), null);

        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialsBuffer);
        gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(ChunkIndex * LeafCount * sizeof(int)), (nuint)(sizeof(int) * LeafCount), null);
        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, lightLevelsBuffer);
        gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(ChunkIndex * LeafCount * sizeof(Vector4)), (nuint)(sizeof(Vector4) * LeafCount), null);

        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, chunkDataBuffer);
        gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(ChunkIndex * sizeof(ChunkData)), (nuint)sizeof(ChunkData), null);

        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);

        ChunkIndex = -1;

        State = ChunkState.AwaitFirstUpload;
    }

    public void Dispose()
    {
        if (State == ChunkState.Clean)
        {
            Console.WriteLine("Chunk is still uploaded. Unload it first.");
            return;
        }

        Positions = null;
        Materials = null;
        LightLevels = null;
        State = ChunkState.Uninitialized;
    }

    public static int CalculateNodeCount(int chunkSize)
    {
        if ((chunkSize & (chunkSize - 1)) != 0)
        {
            throw new ArgumentException("Chunk size must be a power of 2");
        }

        int nodeCount = (int)(((Math.Pow(8, Math.Log(chunkSize, 2) + 1) - 1) / 7));
        return nodeCount;
    }

    public static int CalculateLeafCount(int chunkSize)
    {
        return chunkSize * chunkSize * chunkSize;
    }
}

public class ChunkGenerator
{
    public Chunk chunk;
    public float VoxelSize => chunk.VoxelSize;
    public int NodeCount => chunk.NodeCount;
    public int LeafCount => chunk.LeafCount;
    public int ChunkSize => chunk.ChunkSize;
    public int ChunkWidth => chunk.ChunkWidth;
    
    public ChunkGenerator(Chunk chunk)
    {
        this.chunk = chunk;
    }

    public async Task GenerateOctree(ConcurrentDictionary<Vector3, (int, Vector4)> chunkData)
    {
        var startTime = DateTime.Now;

        // Set root node
        chunk.Positions[0] = new NodeData(Vector3i.Zero, ChunkWidth, 0).PackNodeData();

        // Divide the chunk into subregions for parallel processing
        int subregionSize = ChunkWidth / 2; // Split into 8 major subregions
        var tasks = new Task[8];

        for (int i = 0; i < 8; i++)
        {
            Vector3 offset = childOffsets[i] * subregionSize;
            Vector3 subregionPos = offset;
            int baseIndex = i + 1;  // Root is at 0, children start at 1

            tasks[i] = Task.Run(() =>
            {
                BuildOctreeSubregion(subregionPos, subregionSize, baseIndex, chunkData);
            });
        }

        await Task.WhenAll(tasks);

        var endTime = DateTime.Now;
        Console.WriteLine($"Parallel octree generation took {(endTime - startTime).TotalMilliseconds} ms");
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

    private bool BuildOctreeSubregion(Vector3 position, float size, int index, ConcurrentDictionary<Vector3, (int, Vector4)> chunkData)
    {
        chunk.Positions[index] = new NodeData(new Vector3i((int)position.X, (int)position.Y, (int)position.Z), (int)size, 0).PackNodeData();

        float childSize = size / 2.0f;
        if (childSize < VoxelSize)
        {
            // Convert world position to chunk-local position
            Vector3 chunkPosition = new Vector3(
                MathF.Floor(position.X / (ChunkSize * VoxelSize)) * ChunkSize * VoxelSize,
                MathF.Floor(position.Y / (ChunkSize * VoxelSize)) * ChunkSize * VoxelSize,
                MathF.Floor(position.Z / (ChunkSize * VoxelSize)) * ChunkSize * VoxelSize
            );
            Vector3 localPos = (position - chunkPosition) / VoxelSize;

            int dataIndex = GetIndexFromLocalPosition(localPos);
            if (dataIndex >= 0 && dataIndex < LeafCount)
            {
                var (material, color) = chunkData.TryGetValue(localPos, out var data) ? data : (0, Vector4.Zero);
                chunk.Materials[dataIndex] = new VoxelMaterial(new Vector3(color.X, color.Y, color.Z), material == 1, 0).PackMaterial();
                chunk.LightLevels[dataIndex] = new Vector4(0);

                chunk.Positions[index] = new NodeData(new Vector3i((int)position.X, (int)position.Y, (int)position.Z), (int)size, 1).PackNodeData();
            }
            return true;
        }

        bool allEmpty = false;
        for (int i = 0; i < 8; i++)
        {
            Vector3 offset = childOffsets[i] * childSize;
            Vector3 childPosition = position + offset;
            int childIndex = index * 8 + 1 + i;

            var isLeaf = BuildOctreeSubregion(childPosition, childSize, childIndex, chunkData);

            if (!isLeaf)
            {
                allEmpty = false;
            }
        }

        if (allEmpty)
        {
            
        }

        return allEmpty;
    }

    // Get index from local position (position already relative to chunk origin)
    public int GetIndexFromLocalPosition(Vector3 localPos)
    {
        // Ensure position is within chunk bounds
        if (localPos.X < 0 || localPos.X >= ChunkSize ||
            localPos.Y < 0 || localPos.Y >= ChunkSize ||
            localPos.Z < 0 || localPos.Z >= ChunkSize)
        {
            return -1;
        }

        return (int)localPos.X + 
               (int)localPos.Y * ChunkSize + 
               (int)localPos.Z * ChunkSize * ChunkSize;
    }

    // Get index from world position
    public int GetIndexFromPosition(Vector3 worldPos)
    {
        // Convert world position to chunk-local position
        Vector3 chunkPosition = new Vector3(
            MathF.Floor(worldPos.X / (ChunkSize * VoxelSize)) * ChunkSize * VoxelSize,
            MathF.Floor(worldPos.Y / (ChunkSize * VoxelSize)) * ChunkSize * VoxelSize,
            MathF.Floor(worldPos.Z / (ChunkSize * VoxelSize)) * ChunkSize * VoxelSize
        );
        Vector3 localPos = (worldPos - chunkPosition) / VoxelSize;
        
        return GetIndexFromLocalPosition(localPos);
    }

    

}