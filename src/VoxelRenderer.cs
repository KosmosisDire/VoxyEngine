using System;
using System.Numerics;
using Engine;
using ILGPU.Util;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Shader = Engine.Shader;
namespace VoxelEngine;

public class VoxelRenderer : IDisposable
{
    private OctreeGenerator octreeGenerator;
    private GPULightingCalculator lightingCalculator;

    private uint positionsBuffer;
    private uint materialsBuffer;
    private uint lightLevelsBuffer;
    private uint colorsBuffer;

    private GLContext ctx;
    private Window window;
    private Camera camera;
    private int nodeCount;

    readonly Mesh screenQuad;
    readonly Shader voxelShader;
    readonly Shader texShader;
    readonly SSAORenderer ssao;
    uint ssaoTex = 0;

    public int chunksize;



    public unsafe VoxelRenderer(Window window, Camera camera, int chunkSize)
    {
        this.ctx = window.context;
        this.window = window;
        this.camera = camera;
        this.chunksize = chunkSize;
        octreeGenerator = new OctreeGenerator(chunkSize);
        lightingCalculator = new GPULightingCalculator(octreeGenerator);


        screenQuad = MeshGen.Quad(ctx);
        screenQuad.CreateFlattenedBuffers();
        screenQuad.UploadBuffers();

        voxelShader = new Shader(window.context, "shaders/vert.glsl", "shaders/vox-frag.glsl");
        texShader = new Shader(window.context, "shaders/vert.glsl", "shaders/tex-frag.glsl");

        ssao = new SSAORenderer(window, screenQuad);

        nodeCount = octreeGenerator.NodeCount;
        var leafCount = octreeGenerator.LeafCount;

        ctx.ExecuteCmd((dt, gl) =>
        {
            var buffers = new uint[4];
            gl.GenBuffers(4, buffers);
            positionsBuffer = buffers[0];
            materialsBuffer = buffers[1];
            lightLevelsBuffer = buffers[2];
            colorsBuffer = buffers[3];

            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, positionsBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(nodeCount * sizeof(float) * 4), null, BufferUsageARB.StaticDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialsBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(leafCount * sizeof(int)), null, BufferUsageARB.DynamicDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, lightLevelsBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(leafCount * sizeof(float)), null, BufferUsageARB.DynamicDraw);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, colorsBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(leafCount * sizeof(float) * 4), null, BufferUsageARB.DynamicDraw);
            
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);

            CreateFramebuffer(gl);
        });

        window.SilkWindow.FramebufferResize += (size) =>
        {
            ctx.ExecuteCmd((dt, gl) =>
            {
                CreateFramebuffer(gl);
            });
        };

        window.SilkWindow.Update += (dt) =>
        {
            // move sun in a circle
            // sunDirection = Vector3.Transform(sunDirection, Matrix4x4.CreateRotationY(0.01f));
        };

        // new Thread(LightingThread).Start();
    }

    void LightingThread()
    {
        while (true)
        {
            lightingCalculator.CalculateLighting(sunDirection);
            UploadLighting();
        }
    }

    private uint framebuffer;   
    private uint colorTexture, normalTexture, depthTexture, positionTexture;

    private unsafe void CreateFramebuffer(GL gl)
    {
        DisposeFrameBuffer(gl);
        // Generate the framebuffer
        gl.GenFramebuffers(1, out framebuffer);
        gl.BindFramebuffer(FramebufferTarget.Framebuffer, framebuffer);

        // Create the color texture
        gl.GenTextures(1, out colorTexture);
        gl.BindTexture(TextureTarget.Texture2D, colorTexture);
        gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba16f, (uint)window.Size.X, (uint)window.Size.Y, 0, PixelFormat.Rgba, PixelType.Float, null);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, colorTexture, 0);

        // Create the normal texture
        gl.GenTextures(1, out normalTexture);
        gl.BindTexture(TextureTarget.Texture2D, normalTexture);
        gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba16f, (uint)window.Size.X, (uint)window.Size.Y, 0, PixelFormat.Rgba, PixelType.Float, null);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment1, TextureTarget.Texture2D, normalTexture, 0);

        // Create the position texture
        gl.GenTextures(1, out positionTexture);
        gl.BindTexture(TextureTarget.Texture2D, positionTexture);
        gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba16f, (uint)window.Size.X, (uint)window.Size.Y, 0, PixelFormat.Rgba, PixelType.Float, null);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment2, TextureTarget.Texture2D, positionTexture, 0);

        // Create the depth texture
        gl.GenTextures(1, out depthTexture);
        gl.BindTexture(TextureTarget.Texture2D, depthTexture);
        gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.DepthComponent, (uint)window.Size.X, (uint)window.Size.Y, 0, PixelFormat.DepthComponent, PixelType.Float, null);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, TextureTarget.Texture2D, depthTexture, 0);

        // Specify the list of draw buffers
        GLEnum[] drawBuffers = { GLEnum.ColorAttachment0, GLEnum.ColorAttachment1, GLEnum.ColorAttachment2 };
        fixed (GLEnum* drawBuffersPtr = drawBuffers)
        {
            gl.DrawBuffers(3, drawBuffersPtr);  // Tell OpenGL we are drawing to these color attachments
        }

        // Check framebuffer completeness
        if (gl.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != GLEnum.FramebufferComplete)
        {
            Console.WriteLine("Framebuffer not complete!");
        }

        gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);  // Unbind framebuffer
    }

    Vector3 sunDirection = new Vector3(0.7f, 0.5f, 0.5f);
 
    public unsafe void UpdateTerrain(Vector3 centerPosition)
    {
        Console.WriteLine($"Updating terrain centered at {centerPosition}");
        octreeGenerator.GenerateOctree(centerPosition);
        lightingCalculator.UploadOctreeData();
        UploadTerrain();        
    }

    private unsafe void UploadLighting()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, lightLevelsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(octreeGenerator.LeafCount * sizeof(float)), (ReadOnlySpan<float>)octreeGenerator.lightLevels);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    private unsafe void UploadTerrain()
    {
        nodeCount = octreeGenerator.NodeCount;

        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, positionsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(nodeCount * sizeof(float) * 4), (ReadOnlySpan<Float4>)octreeGenerator.positions);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(octreeGenerator.LeafCount * sizeof(int)), (ReadOnlySpan<int>)octreeGenerator.materials);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, lightLevelsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(octreeGenerator.LeafCount * sizeof(float)), (ReadOnlySpan<float>)octreeGenerator.lightLevels);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, colorsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(octreeGenerator.LeafCount * sizeof(float) * 4), (ReadOnlySpan<Float4>)octreeGenerator.colors);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    public void Draw()
    {
        ctx.RenderCmd((dt, gl) =>
        {
            // Step 1: Bind the framebuffer to render off-screen
            // gl.BindFramebuffer(FramebufferTarget.Framebuffer, framebuffer);
            // gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            voxelShader.Use();
            voxelShader.SetUniform("time", (float)window.context.Time);
            voxelShader.SetUniform("resolution", window.Size);
            voxelShader.SetUniform("mouse", window.context.Input.Mice[0].Position);
            voxelShader.SetUniform("view", camera.ViewMatrix);
            voxelShader.SetUniform("projection", camera.PerspectiveMatrix);
            voxelShader.SetUniform("near", camera.Near);
            voxelShader.SetUniform("far", camera.Far);
            voxelShader.SetUniform("sun_dir", sunDirection);
            voxelShader.SetUniform("voxelSize", octreeGenerator.voxelSize);
            voxelShader.SetUniform("chunkSize", octreeGenerator.ChunkSize);

            // Bind the octree data buffers to the shader storage buffer binding points
            gl.BindBufferBase(GLEnum.ShaderStorageBuffer, 0, positionsBuffer);
            gl.BindBufferBase(GLEnum.ShaderStorageBuffer, 1, materialsBuffer);
            gl.BindBufferBase(GLEnum.ShaderStorageBuffer, 2, lightLevelsBuffer);
            gl.BindBufferBase(GLEnum.ShaderStorageBuffer, 3, colorsBuffer);
            

            // Set shader uniforms
            voxelShader.SetUniform("octreeNodeCount", nodeCount);
            voxelShader.SetUniform("maxTreeDepth", 10);  // Adjust this based on your octree depth

            //Bind the SSAO texture in texture unit 0
            // voxelShader.SetUniform("ssaoTexture", 0);
            // gl.ActiveTexture(TextureUnit.Texture0);
            // gl.BindTexture(TextureTarget.Texture2D, ssaoTex);

            // Render the scene to the framebuffer
            screenQuad.Draw();

            // Unbind the framebuffer
            // gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

            // Step 2: Generate SSAO from depth texture
            // ssaoTex = ssao.GenerateSSAO(positionTexture, normalTexture, camera.ViewMatrix, camera.PerspectiveMatrix);

            // Step 3: Render the final scene to the screen
            // texShader.Use();

            // texShader.SetUniform("tex", 0);
            // gl.ActiveTexture(TextureUnit.Texture0);
            // gl.BindTexture(TextureTarget.Texture2D, colorTexture);

            // Draw the final screen quad to display on screen
            // screenQuad.Draw();
        });
    }


    public void Dispose()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            // Clean up OpenGL resources
            gl.DeleteBuffers(4, new uint[] { positionsBuffer, materialsBuffer, lightLevelsBuffer, colorsBuffer });
            DisposeFrameBuffer(gl);
        });
    }

    private void DisposeFrameBuffer(GL gl)
    {
        if (depthTexture != 0)
        {
            gl.DeleteTextures(1, ref depthTexture);
        }

        if (colorTexture != 0)
        {
            gl.DeleteTextures(1, ref colorTexture);
        }

        if (normalTexture != 0)
        {
            gl.DeleteTextures(1, ref normalTexture);
        }

        if (positionTexture != 0)
        {
            gl.DeleteTextures(1, ref positionTexture);
        }

        if (framebuffer != 0)
        {
            gl.DeleteFramebuffers(1, ref framebuffer);
        }
    }

}
