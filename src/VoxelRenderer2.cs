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
    private Vector2D<int> currentSize;
    
    // Lighting parameters
    private Vector3 sunDirection = new(0.7f, 0.5f, 0.5f);
    private Vector3 sunColor = new(1f, 0.95f, 0.8f);
    
    // Screen quad for output
    private Mesh screenQuad;
    private Shader blitShader;
    
    // Generation parameters
    private const int ChunksPerFrame = 1024; // Number of chunks to generate per frame
    private bool isGenerating = true;
    private ulong generatedChunks = 0;

    public unsafe VoxelRenderer2(Window window, Camera camera)
    {
        this.ctx = window.context;
        this.window = window;
        this.camera = camera;
        this.chunkManager = new ChunkManager2();
        this.currentSize = window.Size;

        // Create screen quad for output display
        screenQuad = MeshGen.Quad(ctx);
        screenQuad.CreateFlattenedBuffers();
        screenQuad.UploadBuffers();

        // Initialize shaders
        renderShader = new ComputeShader(ctx, "shaders/raymarch.comp.glsl");
        blitShader = new Shader(ctx, "shaders/vert.glsl", "shaders/tex-frag.glsl");
        
        // Create initial output texture
        CreateOutputTexture();

        // Initialize chunk manager
        chunkManager.CreateBuffers(ctx);

        // Subscribe to window resize
        window.SilkWindow.Resize += HandleResize;
    }

    private void HandleResize(Vector2D<int> newSize)
    {
        if (newSize == currentSize) return;
        
        currentSize = newSize;
        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.DeleteTexture(outputTexture);
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

    public void Draw(double dt)
    {
        ctx.RenderCmd((dt, gl) =>
        {
            // Generate chunks if needed
            if (isGenerating && generatedChunks < ChunkManager2.NumChunks)
            {
                chunkManager.GenerateChunkTerrain(ctx, ChunksPerFrame);
                generatedChunks += (ulong)ChunksPerFrame;
                
                if (generatedChunks >= ChunkManager2.NumChunks)
                {
                    isGenerating = false;
                }
            }

            // Bind buffers and output texture
            chunkManager.BindBuffers(gl);
            gl.BindImageTexture(0, outputTexture, 0, false, 0, BufferAccessARB.WriteOnly, InternalFormat.Rgba32f);

            // Update sun direction
            sunDirection = Vector3.Transform(sunDirection, Matrix4x4.CreateRotationY(0.0001f));

            // Set raymarching uniforms
            renderShader.SetUniform("viewMatrix", camera.ViewMatrix);
            renderShader.SetUniform("projMatrix", camera.PerspectiveMatrix);
            renderShader.SetUniform("cameraPos", camera.Position);
            renderShader.SetUniform("sunDir", sunDirection);
            renderShader.SetUniform("sunColor", sunColor);
            renderShader.SetUniform("time", (float)window.SilkWindow.Time);

            // Dispatch compute shader
            uint groupSizeX = 4;
            uint groupSizeY = 4;
            uint workGroupsX = (uint)((currentSize.X + (groupSizeX - 1)) / groupSizeX);
            uint workGroupsY = (uint)((currentSize.Y + (groupSizeY - 1)) / groupSizeY);
            renderShader.Dispatch(workGroupsX, workGroupsY, 1);

            // Draw result to screen
            gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            blitShader.Use();
            gl.BindTexture(TextureTarget.Texture2D, outputTexture);
            blitShader.SetUniform("tex", 0);
            screenQuad.Draw();
        });
    }

    public void Dispose()
    {
        window.SilkWindow.Resize -= HandleResize;
        
        ctx.ExecuteCmd((dt, gl) =>
        {
            renderShader?.Dispose();
            blitShader?.Dispose();
            if (outputTexture != 0) gl.DeleteTexture(outputTexture);
            chunkManager.Dispose(gl);
            screenQuad?.Dispose();
        });
    }
}