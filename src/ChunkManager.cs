using System.Collections;
using System.Numerics;
using System.Runtime.InteropServices;
using Engine;
using Silk.NET.OpenGL;

public class ChunkManager
{
    public Vector3 Origin;

    // GPU Buffer handles
    public uint chunkMasks;         // uvec4 per chunk for block occupancy
    public uint blockMasks;         // uvec4 per block for voxel occupancy
    public uint materialIndices;    // Block material indices for homogeneous blocks
    public uint materialData;       // Material struct array
    public uint uncompressedMaterials;       // Material struct array

    // Constants
    public const uint ChunkSize = 8;             // 8x8x8 blocks per chunk
    public const uint GridSize = 25;             // 50x50x50 chunks
    public const uint NumChunks = GridSize * GridSize * GridSize;
    public const uint SplitNodeCount = ChunkSize * ChunkSize * ChunkSize;
    public const uint MaskUintsNeeded = (SplitNodeCount + 31) / 32;
    public const uint BytesPerMask = MaskUintsNeeded * sizeof(uint);

    ComputeShader buildShader;

    public ChunkManager()
    {
        this.Origin = Vector3.Zero;
        Materials.Initialize();
    }

    public unsafe void CreateBuffers(GLContext ctx)
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            // Generate buffer objects
            gl.GenBuffers(1, out chunkMasks);
            gl.GenBuffers(1, out blockMasks);
            gl.GenBuffers(1, out materialIndices);
            gl.GenBuffers(1, out materialData);
            gl.GenBuffers(1, out uncompressedMaterials);

            // Create chunk occupancy buffer (uvec4 per chunk)
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, chunkMasks);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(NumChunks * BytesPerMask), null, BufferUsageARB.DynamicDraw);

            // Create block occupancy buffer (uvec4 per block)
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, blockMasks);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(NumChunks * SplitNodeCount * BytesPerMask), null, BufferUsageARB.DynamicDraw);

            // Create block material indices buffer (uint per block)
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialIndices);
            uint[] initialIndices = new uint[NumChunks * SplitNodeCount];
            Array.Fill(initialIndices, uint.MaxValue);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(NumChunks * SplitNodeCount * sizeof(uint)), [.. initialIndices], BufferUsageARB.DynamicDraw);

            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialData);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(256 * sizeof(Material)), [.. Materials.materials], BufferUsageARB.StaticDraw);
            gl.ObjectLabel(ObjectIdentifier.Buffer, materialData, (uint)Materials.materials.Length, "Material Data");

            // re-upload material data on reload
            Materials.OnMaterialsReloaded += () =>
            {
                ctx.ExecuteCmd((dt, gl) =>
                {
                    Console.WriteLine("Uploading material data");
                    gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialData);
                    gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, 0, (nuint)(256 * sizeof(Material)), [.. Materials.materials]);
                    gl.ObjectLabel(ObjectIdentifier.Buffer, materialData, (uint)Materials.materials.Length, "Material Data");
                });
            };

            // Uncompressed materials buffer. One byte per voxel representing the material index
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, uncompressedMaterials);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(NumChunks * SplitNodeCount * SplitNodeCount), null, BufferUsageARB.DynamicDraw);



            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });

        buildShader = new ComputeShader(ctx, "shaders/terrain-gen.comp.glsl");
    }

    public unsafe void BindBuffers(GL gl)
    {
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 0, chunkMasks);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 1, blockMasks);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 2, materialIndices);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 3, materialData);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 4, uncompressedMaterials);
    }

    public unsafe void GenerateChunkTerrain(GLContext ctx, int numChunks = 1)
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            BindBuffers(gl);

            buildShader.SetUniform("chunkIndex", chunkCount);
            buildShader.SetUniform("chunkCount", numChunks);

            buildShader.Dispatch(ChunkSize, ChunkSize, ChunkSize);

            chunkCount += numChunks;
        });
    }

    public void GenerateAllTerrainCPU(GLContext ctx)
    {
        var generator = new TerrainGenerator();
        ctx.ExecuteCmd((dt, gl) =>
        {
            generator.GenerateAllTerrain(gl, chunkMasks, blockMasks, uncompressedMaterials);
        });
        chunkCount = (int)(GridSize * GridSize * GridSize);
    }

    public unsafe void Dispose(GL gl)
    {
        gl.DeleteBuffers(1, ref chunkMasks);
        gl.DeleteBuffers(1, ref blockMasks);
        gl.DeleteBuffers(1, ref materialIndices);
        gl.DeleteBuffers(1, ref materialData);
        gl.DeleteBuffers(1, ref uncompressedMaterials);
        buildShader.Dispose();
    }

    private int chunkCount = 0;
}