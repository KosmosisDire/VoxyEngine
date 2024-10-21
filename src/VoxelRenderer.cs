using System;
using System.Numerics;
using Engine;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Shader = Engine.Shader;
namespace VoxelEngine;

public class VoxelRenderer : IDisposable
{
    private const int MAX_OCTREE_NODES = 3000000;

    private OctreeGenerator octreeGenerator;

    private uint nodeDataBuffer;
    private uint nodeInfoBuffer;

    private GLContext ctx;
    private Window window;
    private Camera camera;
    private int nodeCount;

    readonly Mesh screenQuad;
    readonly Shader voxelShader;
    readonly Shader texShader;
    readonly SSAORenderer ssao;
    uint ssaoTex = 0;

    public unsafe VoxelRenderer(Window window, Camera camera)
    {
        this.ctx = window.context;
        this.window = window;
        this.camera = camera;
        octreeGenerator = new OctreeGenerator();

        screenQuad = MeshGen.Quad(ctx);
        screenQuad.CreateFlattenedBuffers();
        screenQuad.UploadBuffers();

        voxelShader = new Shader(window.context, "shaders/vert.glsl", "shaders/vox-frag.glsl");
        texShader = new Shader(window.context, "shaders/vert.glsl", "shaders/tex-frag.glsl");

        ssao = new SSAORenderer(window, screenQuad);

        ctx.ExecuteCmd((dt, gl) =>
        {
            var buffers = new uint[2];
            gl.GenBuffers(2, buffers);
            nodeDataBuffer = buffers[0];
            nodeInfoBuffer = buffers[1];

            // Generate buffers for shader storage
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, nodeDataBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(MAX_OCTREE_NODES * sizeof(float) * 4), null, BufferUsageARB.DynamicDraw);

            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, nodeInfoBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(MAX_OCTREE_NODES * sizeof(float) * 4), null, BufferUsageARB.DynamicDraw);
            
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
 
    public unsafe void UpdateTerrain(Vector3 centerPosition)
    {
        Console.WriteLine($"Updating terrain centered at {centerPosition}");
        octreeGenerator.GenerateOctree(128, centerPosition);
        UploadTerrain();
    }

    private unsafe void UploadTerrain(int start = 0, int count = -1)
    {
        nodeCount = octreeGenerator.NodeCount;
        if (count == -1)
        {
            count = nodeCount;
        }

        ctx.ExecuteCmd((dt, gl) =>
        {
            int startByte = start * sizeof(float) * 4;
            nuint countByte = (nuint)(count * sizeof(float) * 4); 

            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, nodeDataBuffer);
            gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, startByte, countByte, (ReadOnlySpan<Vector4>)octreeGenerator.posAndSize);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, nodeInfoBuffer);
            gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, startByte, countByte, (ReadOnlySpan<Vector4>)octreeGenerator.nodeData);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    public void Draw()
    {
        ctx.RenderCmd((dt, gl) =>
        {
            // Step 1: Bind the framebuffer to render off-screen
            gl.BindFramebuffer(FramebufferTarget.Framebuffer, framebuffer);
            gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            voxelShader.Use();
            voxelShader.SetUniform("time", (float)window.context.Time);
            voxelShader.SetUniform("resolution", window.Size);
            voxelShader.SetUniform("mouse", window.context.Input.Mice[0].Position);
            voxelShader.SetUniform("view", camera.ViewMatrix);
            voxelShader.SetUniform("projection", camera.PerspectiveMatrix);
            voxelShader.SetUniform("near", camera.Near);
            voxelShader.SetUniform("far", camera.Far);

            // Bind the octree data buffers to the shader storage buffer binding points
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 0, nodeDataBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 1, nodeInfoBuffer);
            

            // Set shader uniforms
            voxelShader.SetUniform("octreeNodeCount", nodeCount);
            voxelShader.SetUniform("maxTreeDepth", 10);  // Adjust this based on your octree depth

            //Bind the SSAO texture in texture unit 0
            voxelShader.SetUniform("ssaoTexture", 0);
            gl.ActiveTexture(TextureUnit.Texture0);
            gl.BindTexture(TextureTarget.Texture2D, ssaoTex);

            // Render the scene to the framebuffer
            screenQuad.Draw();

            // Unbind the framebuffer
            gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

            // Step 2: Generate SSAO from depth texture
            ssaoTex = ssao.GenerateSSAO(positionTexture, normalTexture, camera.ViewMatrix, camera.PerspectiveMatrix);

            // Step 3: Render the final scene to the screen
            texShader.Use();

            texShader.SetUniform("tex", 0);
            gl.ActiveTexture(TextureUnit.Texture0);
            gl.BindTexture(TextureTarget.Texture2D, colorTexture); // render depth for debugging

            // Draw the final screen quad to display on screen
            screenQuad.Draw();
        });
    }


    public void Dispose()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            // Clean up OpenGL resources
            gl.DeleteBuffers(2, new uint[] { nodeDataBuffer, nodeInfoBuffer });
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
