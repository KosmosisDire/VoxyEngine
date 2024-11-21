using System.Numerics;
using Engine;
using Silk.NET.OpenGL;

public class ChunkManager2
{
    public Vector3 Origin;

    public uint outerBitmasksLow;
    public uint outerBitmasksHigh;
    public uint voxelBitmasksLow;
    public uint voxelBitmasksHigh;
    public uint voxelData;
    public uint blockIndices;    
    public uint freeBlockQueue;
    public uint atomicCounters;

    int chunkCount = 0;
    public const uint chunkGridWidth = 128;
    public const uint numChunks = chunkGridWidth * chunkGridWidth * chunkGridWidth;
    public const uint maxBlockCount = numChunks * 64; // Maximum number of 64-voxel blocks
    public uint maxVoxelCount = uint.MaxValue; // Maximum number of voxels

    ComputeShader buildShader;

    public ChunkManager2()
    {
        this.Origin = Vector3.Zero;
    }

    public unsafe void CreateBuffers(GLContext ctx)
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            // Generate all buffer objects
            gl.GenBuffers(1, out outerBitmasksLow);
            gl.GenBuffers(1, out outerBitmasksHigh);
            gl.GenBuffers(1, out voxelBitmasksLow);
            gl.GenBuffers(1, out voxelBitmasksHigh);
            gl.GenBuffers(1, out voxelData);
            gl.GenBuffers(1, out blockIndices);
            gl.GenBuffers(1, out freeBlockQueue);
            gl.GenBuffers(1, out atomicCounters);

            // Create and initialize outer bitmask buffers
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, outerBitmasksLow);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(numChunks * sizeof(uint)), null, BufferUsageARB.DynamicDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, outerBitmasksHigh);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(numChunks * sizeof(uint)), null, BufferUsageARB.DynamicDraw);

            // Create and initialize voxel bitmask buffers
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, voxelBitmasksLow);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(maxBlockCount * sizeof(uint)), null, BufferUsageARB.DynamicDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, voxelBitmasksHigh);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(maxBlockCount * sizeof(uint)), null, BufferUsageARB.DynamicDraw);

            // Initialize block indices mapping buffer
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, blockIndices);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(maxBlockCount * sizeof(uint)), null, BufferUsageARB.DynamicDraw);
            
            // fill with -1
            uint[] initialBlockIndices = new uint[maxBlockCount];
            Array.Fill(initialBlockIndices, uint.MaxValue);
            gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, 0, (nuint)(maxBlockCount * sizeof(uint)), [..initialBlockIndices]);

            // Initialize voxel data buffer
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, voxelData);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(maxVoxelCount * sizeof(uint)), null, BufferUsageARB.DynamicDraw);

            // Initialize free block queue
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, freeBlockQueue);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(maxBlockCount * sizeof(uint)), null, BufferUsageARB.DynamicDraw);

            // assign every number
            uint[] freeBlocks = new uint[maxBlockCount];
            for (uint i = 0; i < maxBlockCount; i++)
            {
                freeBlocks[i] = i * 64;
            }
            gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, 0, (nuint)(maxBlockCount * sizeof(uint)), [..freeBlocks]);

            // Initialize atomic counters (front and back pointers)
            gl.BindBuffer(BufferTargetARB.AtomicCounterBuffer, atomicCounters);
            gl.BufferData(BufferTargetARB.AtomicCounterBuffer, 2 * sizeof(uint), null, BufferUsageARB.DynamicDraw);
            gl.BufferSubData(BufferTargetARB.AtomicCounterBuffer, 0, 2 * sizeof(uint), [0u, maxBlockCount]);

            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
            gl.BindBuffer(BufferTargetARB.AtomicCounterBuffer, 0);
        });

        buildShader = new ComputeShader(ctx, "shaders/build-tree.comp.glsl");
    }

    public unsafe void BindBuffers(GL gl)
    {
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 2, outerBitmasksLow);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 3, outerBitmasksHigh);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 4, blockIndices);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 6, voxelBitmasksLow);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 7, voxelBitmasksHigh);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 8, voxelData);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 9, freeBlockQueue);
        gl.BindBufferBase(BufferTargetARB.AtomicCounterBuffer, 0, atomicCounters); // Binding point 0 for atomic counters
    }

    public unsafe void GenerateChunkTerrain(GLContext ctx, int numChunks = 1)
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            BindBuffers(gl);

            // Set uniforms
            buildShader.SetUniform("chunkIndex", chunkCount);
            buildShader.SetUniform("chunkCount", numChunks);
            buildShader.SetUniform("blockIdxCapacity", maxBlockCount);


            buildShader.Dispatch(4, 4, 4);

            chunkCount += numChunks;
        }); 
    }

    public unsafe void Dispose(GL gl)
    {
        gl.DeleteBuffers(1, ref outerBitmasksLow);
        gl.DeleteBuffers(1, ref outerBitmasksHigh);
        gl.DeleteBuffers(1, ref voxelBitmasksLow);
        gl.DeleteBuffers(1, ref voxelBitmasksHigh);
        gl.DeleteBuffers(1, ref voxelData);
        gl.DeleteBuffers(1, ref blockIndices);
        gl.DeleteBuffers(1, ref freeBlockQueue);
        gl.DeleteBuffers(1, ref atomicCounters);
        buildShader.Dispose();
    }
}