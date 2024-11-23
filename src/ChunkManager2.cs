using System.Numerics;
using System.Runtime.InteropServices;
using Engine;
using Silk.NET.OpenGL;

public class ChunkManager2
{
    public Vector3 Origin;

    // GPU Buffer handles
    public uint chunkMasks;         // uvec4 per chunk for 5x5x5 block occupancy
    public uint blockMasks;         // uvec4 per block for 5x5x5 voxel occupancy
    public uint materialIndices;    // Block material indices for homogeneous blocks
    public uint materialData;       // Material struct array

    public uint blockHashmap;       // Block hashmap for heterogeneous blocks
    public uint voxelMaterials;     // Voxel material indices for heterogeneous blocks
    public uint voxelMaterialsCounter; // Counter for voxel material slots

    public const uint ChunkSize = 5;             // 5x5x5 blocks per chunk
    public const uint GridSize = 128;            // 128x128x128 chunks
    public const uint NumChunks = GridSize * GridSize * GridSize;
    public const uint SplitNodeCount = ChunkSize * ChunkSize * ChunkSize;
    public const uint MaskUintsNeeded = 4;
    public const uint BytesPerMask = MaskUintsNeeded * sizeof(uint);

    ComputeShader buildShader;

    [StructLayout(LayoutKind.Sequential)]
    public struct Material
    {
        public Vector3 Color;
        public float Metallic;
        public float Roughness;
        public float Emission;
        public float NoiseSize;
        public float NoiseStrength;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct KeyValue
    {
        public uint Key;
        public uint Value;

        public KeyValue(uint key, uint value)
        {
            Key = key;
            Value = value;
        }
    }

    public ChunkManager2()
    {
        this.Origin = Vector3.Zero;
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
            gl.GenBuffers(1, out blockHashmap);
            gl.GenBuffers(1, out voxelMaterials);
            gl.GenBuffers(1, out voxelMaterialsCounter);

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
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(NumChunks * SplitNodeCount * sizeof(uint)), [..initialIndices], BufferUsageARB.DynamicDraw);

            // Create and initialize materials buffer
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialData);
            Material[] defaultMaterials = new Material[]
            {
                // invalid
                new Material {
                    Color = new Vector3(1f, 0, 1f),
                    Metallic = 0.5f,
                    Roughness = 0.5f,
                    Emission = 1f,
                    NoiseSize = 0f,
                    NoiseStrength = 0f
                },
                // Grass
                new Material { 
                    Color = new Vector3(0.21f, 0.31f, 0.27f), 
                    Metallic = 0, 
                    Roughness = 0.9f, 
                    Emission = 0,
                    NoiseSize = 0.3f,
                    NoiseStrength = 0.05f
                },
                // Dirt
                new Material { 
                    Color = new Vector3(0.21f, 0.13f, 0.09f), 
                    Metallic = 0, 
                    Roughness = 0.95f, 
                    Emission = 0,
                    NoiseSize = 0.5f,
                    NoiseStrength = 0.2f
                },
                // Stone
                new Material { 
                    Color = new Vector3(0.7f, 0.7f, 0.71f), 
                    Metallic = 0.1f, 
                    Roughness = 0.8f, 
                    Emission = 0,
                    NoiseSize = 0.1f,
                    NoiseStrength = 0.05f
                },
                // Ore
                new Material { 
                    Color = new Vector3(0.7f, 0.3f, 0.9f), 
                    Metallic = 0.8f, 
                    Roughness = 0.6f, 
                    Emission = 0.2f,
                    NoiseSize = 0.2f,
                    NoiseStrength = 0.1f
                }
            };
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(256 * sizeof(Material)), [..defaultMaterials], BufferUsageARB.StaticDraw);

            // Create block hashmap buffer
            // size must be power of 2 (1u << 24)
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, blockHashmap);
            KeyValue[] initialHashmap = new KeyValue[536870912];
            Array.Fill(initialHashmap, new KeyValue(uint.MaxValue, uint.MaxValue));
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(uint.MaxValue), [..initialHashmap], BufferUsageARB.DynamicDraw);

            // Create voxel material indices buffer
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, voxelMaterials);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(uint.MaxValue), null, BufferUsageARB.DynamicDraw);

            // Create voxel material counter buffer
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, voxelMaterialsCounter);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(1 * sizeof(uint)), null, BufferUsageARB.DynamicDraw);

            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });

        buildShader = new ComputeShader(ctx, "shaders/build-tree.comp.glsl");
    }

    public unsafe void BindBuffers(GL gl)
    {
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 0, chunkMasks);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 1, blockMasks);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 2, materialIndices);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 3, materialData);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 4, blockHashmap);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 5, voxelMaterials);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 6, voxelMaterialsCounter);
    }

    public unsafe void GenerateChunkTerrain(GLContext ctx, int numChunks = 1)
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            BindBuffers(gl);

            buildShader.SetUniform("chunkIndex", chunkCount);
            buildShader.SetUniform("chunkCount", numChunks);

            // Work group size matches block dimensions (5x5x5)
            buildShader.Dispatch(5, 5, 5);

            chunkCount += numChunks;
        }); 
    }

    public unsafe void Dispose(GL gl)
    {
        gl.DeleteBuffers(1, ref chunkMasks);
        gl.DeleteBuffers(1, ref blockMasks);
        gl.DeleteBuffers(1, ref materialIndices);
        gl.DeleteBuffers(1, ref materialData);
        gl.DeleteBuffers(1, ref blockHashmap);
        gl.DeleteBuffers(1, ref voxelMaterials);
        gl.DeleteBuffers(1, ref voxelMaterialsCounter);
        buildShader.Dispose();
    }

    private int chunkCount = 0;
}