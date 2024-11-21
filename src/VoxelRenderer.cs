using System.Numerics;
using Engine;
using Silk.NET.OpenGL;
using Shader = Engine.Shader;

namespace VoxelEngine;

public class VoxelRenderer : IDisposable
{
    // private ComputeShader directLightingShader;
    // private ComputeShader blurShader;
    // private ComputeShader sandPhysicsShader;
    // private ComputeShader octreeOptimizeShader;
    // private uint changesBuffer;
    // private uint moveFlags;
    // private uint currentFrame = 0;

    private GLContext ctx;
    private Window window;
    private Camera camera;

    readonly Mesh screenQuad;
    readonly Shader voxelShader;

    Vector3 sunDirection = new Vector3(0.7f, 0.5f, 0.5f);
    Vector3 sunColor = new Vector3(1, 0.9f, 1);

    public unsafe VoxelRenderer(Window window, Camera camera, int chunkSize, float voxelSize)
    {
        this.ctx = window.context;
        this.window = window;
        this.camera = camera;

        screenQuad = MeshGen.Quad(ctx);
        screenQuad.CreateFlattenedBuffers();
        screenQuad.UploadBuffers();

        voxelShader = new Shader(window.context, "shaders/vert.glsl", "shaders/vox-frag2.glsl");

        // ctx.ExecuteCmd((dt, gl) =>
        // {
        //     // Initialize compute shaders
        //     directLightingShader = new ComputeShader(ctx, "shaders/lighting.comp.glsl");
        //     blurShader = new ComputeShader(ctx, "shaders/lighting-blur.comp.glsl");
        //     sandPhysicsShader = new ComputeShader(ctx, "shaders/falling.comp.glsl");
        //     octreeOptimizeShader = new ComputeShader(ctx, "shaders/octree-optimize.comp.glsl");


        //     // Initialize buffers
        //     gl.GenBuffers(1, out changesBuffer);
        //     gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, changesBuffer);
        //     gl.BufferData(BufferTargetARB.ShaderStorageBuffer, sizeof(int), null, BufferUsageARB.DynamicCopy);

        //     gl.GenBuffers(1, out moveFlags);
        //     gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, moveFlags);
        //     gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(chunkSize * chunkSize * chunkSize * sizeof(int)), null, BufferUsageARB.DynamicCopy);
        // });

        window.SilkWindow.Update += (dt) =>
        {
            sunDirection = Vector3.Transform(sunDirection, Matrix4x4.CreateRotationY(0.0001f));
        };
    }


    // public async Task UpdateChunks()
    // {
    //     await chunkManager.GenerateChunkTerrain(camera.Position);
    // }

    // int currentLevel = 0;
    // int currentChunk = 0;
    // public unsafe void OptimizeOctrees()
    // {
    //     ctx.RenderCmd((dt, gl) =>
    //     {
    //         var depth = MathF.Log2(chunkManager.ChunkSize);
    //         octreeOptimizeShader.SetUniform("maxDepth", depth);
    //         octreeOptimizeShader.SetUniform("currentLevel", currentLevel);
    //         octreeOptimizeShader.SetUniform("chunkIndex", currentChunk);
    //         octreeOptimizeShader.SetUniform("chunkSize", chunkManager.ChunkSize);
    //         octreeOptimizeShader.SetUniform("voxelSize", chunkManager.VoxelSize);

    //         int startSearch = currentChunk;
    //         do
    //         {
    //             currentChunk = (currentChunk + 1) % ChunkManager.MAX_CHUNKS;
    //         }
    //         while ((chunkManager.ChunksArray[currentChunk] == null || chunkManager.ChunksArray[currentChunk].State != Chunk.ChunkState.Clean) && currentChunk != startSearch);
                

    //         if (currentChunk == 0)
    //             currentLevel = (currentLevel + 1) % (int)depth;

    //         const uint GROUP_SIZE = 64;
    //         uint numWorkGroups = (uint)(chunkManager.NodeCount / GROUP_SIZE); 
    //         octreeOptimizeShader.Dispatch(numWorkGroups, 1, 1);
    //     });
    // }

    // public unsafe void UpdatePhysics(double deltaTime, Vector3 gravity)
    // {
    //     foreach (var (chunk, posBuffer, matBuffer, lightBuffer, colorBuffer) in chunkManager.GetVisibleChunks())
    //     {
    //         ctx.ExecuteCmd((dt, gl) =>
    //         {
    //             // Clear move flags buffer
    //             gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, moveFlags);
    //             gl.ClearBufferData(GLEnum.ShaderStorageBuffer, GLEnum.R8i, GLEnum.RedInteger, GLEnum.Int, null);

    //             gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 1, matBuffer);
    //             gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 2, lightBuffer);
    //             gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 3, colorBuffer);
    //             gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 4, moveFlags);
    //         });

    //         sandPhysicsShader.SetUniform("chunkSize", chunk.NodeCount);
    //         sandPhysicsShader.SetUniform("deltaTime", (float)deltaTime);
    //         sandPhysicsShader.SetUniform("frame", currentFrame++);
    //         sandPhysicsShader.SetUniform("gravityDir", Vector3.Normalize(gravity));
    //         sandPhysicsShader.SetUniform("currentChunkPosition", chunk.Position);

    //         const int COMPUTE_GROUP_SIZE = 512;
    //         uint numWorkGroups = (uint)(chunk.LeafCount + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
    //         sandPhysicsShader.Dispatch(numWorkGroups);
    //     }
    // }

    // int lightingChunk;
    // public void CalculateLighting()
    // {
    //     // Only calculate lighting for chunks that are actually loaded and visible
    //     ctx.ExecuteCmd((dt, gl) =>
    //     {
    //         var chunkPos = chunkManager.GetChunkPosition(camera.Position);
    //         if (!chunkManager.ChunksLookup.TryGetValue(chunkPos, out var chunk))
    //             return;
    //         var chunkIndex = chunk.ChunkIndex;

    //         int[] chunks = [lightingChunk, chunkIndex];

    //         foreach (var c in chunks)
    //         {
    //             // Direct lighting pass
    //             directLightingShader.SetUniform("chunkSize", chunkManager.ChunkSize);
    //             directLightingShader.SetUniform("voxelSize", chunkManager.VoxelSize);
    //             directLightingShader.SetUniform("lightDirection", sunDirection);
    //             directLightingShader.SetUniform("lightColor", new Vector4(sunColor, 1.0f));
    //             directLightingShader.SetUniform("currentTime", (uint)DateTime.UtcNow.Ticks);
    //             directLightingShader.SetUniform("cameraPosition", camera.Position);
    //             directLightingShader.SetUniform("currentChunk", c);

    //             // Calculate dispatch size based on chunk dimensions
    //             const uint GROUP_SIZE = 8;
    //             uint numGroups = (uint)Math.Ceiling(chunkManager.ChunkSize / (float)GROUP_SIZE);
    //             directLightingShader.Dispatch(numGroups, numGroups, numGroups);

    //             // Memory barrier between lighting and blur passes
    //             gl.MemoryBarrier(MemoryBarrierMask.ShaderStorageBarrierBit);

    //             // Blur passes - one for each axis
    //             blurShader.SetUniform("chunkSize", chunkManager.ChunkSize);
    //             blurShader.SetUniform("voxelSize", chunkManager.VoxelSize);
    //             blurShader.SetUniform("currentChunk", c);

    //             // Perform three blur passes (X, Y, Z axes)
    //             for (int pass = 0; pass < 3; pass++)
    //             {
    //                 blurShader.SetUniform("pass", pass);
    //                 blurShader.Dispatch(numGroups, numGroups, numGroups);
    //             }
    //         }

    //         lightingChunk = (lightingChunk + 1) % ChunkManager.MAX_CHUNKS;
    //     });
    // }

    public async void Draw(double dt)
    {
        // await UpdateChunks();

        ctx.RenderCmd((dt, gl) =>
        {
            // CalculateLighting();
            // OptimizeOctrees();

            // voxelShader.Use();
            // voxelShader.SetUniform("time", (float)window.context.Time);
            // voxelShader.SetUniform("resolution", window.Size);
            // voxelShader.SetUniform("mouse", window.context.Input.Mice[0].Position);
            // voxelShader.SetUniform("view", camera.ViewMatrix);
            // voxelShader.SetUniform("projection", camera.PerspectiveMatrix);
            // voxelShader.SetUniform("near", camera.Near);
            // voxelShader.SetUniform("far", camera.Far);
            // voxelShader.SetUniform("sunDir", sunDirection);
            // voxelShader.SetUniform("sunColor", sunColor);
            // voxelShader.SetUniform("chunkSize", chunkManager.ChunkSize);
            // voxelShader.SetUniform("voxelSize", chunkManager.VoxelSize);
            // voxelShader.SetUniform("maxChunkCount", ChunkManager.MAX_CHUNKS);

            // chunkManager.BindBuffers(0, 1, 2, 3, 4, 5);

            screenQuad.Draw();
        });
    }

    public void Dispose()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            // chunkManager.Dispose();

            // directLightingShader.Dispose();
            // blurShader.Dispose();
            // sandPhysicsShader.Dispose();
            // octreeOptimizeShader.Dispose();

            // if (moveFlags != 0) gl.DeleteBuffer(moveFlags);
            // if (changesBuffer != 0) gl.DeleteBuffer(changesBuffer);

            // voxelShader?.Dispose();
            // screenQuad?.Dispose();
            // ssao?.Dispose();
        });
    }

}