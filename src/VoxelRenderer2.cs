using System.Diagnostics;
using System.Numerics;
using Engine;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Shader = Engine.Shader;

public class VoxelRenderer2 : IDisposable
{
    private readonly GLContext ctx;
    private readonly Window window;
    private readonly Camera camera;
    private readonly ChunkManager2 chunkManager;
    
    private ComputeShader renderShader;
    private uint outputTexture;
    private Vector3 sunDirection = new(0.7f, 0.5f, 0.5f);
    private Vector3 sunColor = new(1f, 0.9f, 1f);
    private Vector2D<int> currentSize;

    public long frameTime = 0;

    Mesh screenQuad;
    Shader blitShader;

    public unsafe VoxelRenderer2(Window window, Camera camera)
    {
        this.ctx = window.context;
        this.window = window;
        this.camera = camera;
        this.chunkManager = new ChunkManager2();
        this.currentSize = window.Size;

        screenQuad = MeshGen.Quad(ctx);
        screenQuad.CreateFlattenedBuffers();
        screenQuad.UploadBuffers();

        // Create compute shader
        renderShader = new ComputeShader(ctx, "shaders/raymarch.comp.glsl");
        blitShader = new Shader(ctx, "shaders/vert.glsl", "shaders/tex-frag.glsl");
        
        // Create initial output texture
        CreateOutputTexture();

        // Initialize chunk manager buffers
        chunkManager.CreateBuffers(ctx);

        // for (int x = 0; x < 63; x++)
        // for (int z = 0; z < 63; z++)
        // for (int y = 0; y < 63; y++)
        // {
        //     var pos = new Vector3(x, y, z) * 16;
        //     chunkManager.GenerateChunkTerrain(ctx, pos);
        // }
        
        // Subscribe to window resize events
        window.SilkWindow.Resize += HandleResize;
    }

    private void HandleResize(Vector2D<int> newSize)
    {
        if (newSize == currentSize) return;
        
        currentSize = newSize;
        ctx.ExecuteCmd((dt, gl) =>
        {
            // Delete old texture
            if (outputTexture != 0)
            {
                gl.DeleteTexture(outputTexture);
                outputTexture = 0;
            }

            // Create new texture with updated size
            CreateOutputTexture();
        });
    }

    private unsafe void CreateOutputTexture()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.GenTextures(1, out outputTexture);
            gl.BindTexture(TextureTarget.Texture2D, outputTexture);
            gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba32f, 
                (uint)currentSize.X, (uint)currentSize.Y, 0, 
                PixelFormat.Rgba, PixelType.Float, null);
            gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
            gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
            gl.BindImageTexture(0, outputTexture, 0, false, 0, BufferAccessARB.WriteOnly, InternalFormat.Rgba32f);
        });
    }
 
    ulong chunkIndex = 0;
    int chunksPerFrame = 256;

    public void Draw(double dt)
    {
        ctx.RenderCmd((dt, gl) =>
        {
            var timer = Stopwatch.StartNew();
            timer.Start();

            if (chunkIndex < ChunkManager2.numChunks / 2.0)
            {
                chunkManager.GenerateChunkTerrain(ctx, chunksPerFrame);
                chunkIndex += (ulong)chunksPerFrame;
            }
            

            chunkManager.BindBuffers(gl);

            // Bind output texture
            gl.BindImageTexture(0, outputTexture, 0, false, 0, BufferAccessARB.WriteOnly, InternalFormat.Rgba32f);

            // Set uniforms
            renderShader.SetUniform("viewMatrix", camera.ViewMatrix);
            renderShader.SetUniform("projMatrix", camera.PerspectiveMatrix);
            renderShader.SetUniform("cameraPos", camera.Position);
            renderShader.SetUniform("worldOrigin", chunkManager.Origin);
            renderShader.SetUniform("sunDir", sunDirection);
            renderShader.SetUniform("sunColor", sunColor);
            renderShader.SetUniform("blockIdxCapacity", ChunkManager2.maxBlockCount);


            sunDirection = Vector3.Transform(sunDirection, Matrix4x4.CreateRotationY(0.0001f));

            // Dispatch compute shader
            uint groupSize = 8;
            uint workGroupsX = (uint)((currentSize.X + (groupSize-1)) / groupSize);
            uint workGroupsY = (uint)((currentSize.Y + (groupSize-1)) / groupSize);
            renderShader.Dispatch(workGroupsX, workGroupsY, 1);

            // Draw screen quad
            gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            blitShader.Use();
            gl.BindTexture(TextureTarget.Texture2D, outputTexture);
            blitShader.SetUniform("tex", 0);
            screenQuad.Draw();

            frameTime = timer.ElapsedMilliseconds;
        });
    }

    public void Dispose()
    {
        // Unsubscribe from window resize events
        window.SilkWindow.Resize -= HandleResize;
        
        ctx.ExecuteCmd((dt, gl) =>
        {
            renderShader?.Dispose();
            if (outputTexture != 0) gl.DeleteTexture(outputTexture);
            chunkManager.Dispose(gl);
        });
    }
}