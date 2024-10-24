using System.Numerics;
using Engine;
using Silk.NET.OpenGL;
using Shader = Engine.Shader;

namespace VoxelEngine;

public class VoxelRenderer : IDisposable
{
    private uint directLightingProgram;
    private uint blurProgram;


    private uint sandPhysicsProgram;
    private uint moveFlags;
    private uint currentFrame = 0;


    private uint octreeOptimizeProgram;
    private uint changesBuffer;
    private int[] changesArray = new int[1];

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

    Vector3 sunDirection = new Vector3(0.7f, 0.5f, 0.5f);
    Vector3 sunColor = new Vector3(1, 0.9f, 1);

    private uint framebuffer;
    private uint colorTexture, normalTexture, depthTexture, positionTexture;

    private OctreeGenerator octreeGenerator;

    public unsafe VoxelRenderer(Window window, Camera camera, int chunkSize)
    {
        this.ctx = window.context;
        this.window = window;
        this.camera = camera;
        this.chunksize = chunkSize;

        octreeGenerator = new OctreeGenerator(chunkSize);

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
            // Initialize compute shaders
            directLightingProgram = CreateComputeShader(gl, "shaders/lighting.comp.glsl");
            blurProgram = CreateComputeShader(gl, "shaders/lighting-blur.comp.glsl");
            sandPhysicsProgram = CreateComputeShader(gl, "shaders/falling.comp.glsl");
            octreeOptimizeProgram = CreateComputeShader(gl, "shaders/octree-optimize.comp.glsl");

            // changes buffer for octree optimization
            gl.GenBuffers(1, out changesBuffer);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, changesBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, sizeof(int), null, BufferUsageARB.DynamicCopy);

            // move flags for sand physics
            gl.GenBuffers(1, out moveFlags);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, moveFlags);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(leafCount * sizeof(int)), null, BufferUsageARB.DynamicCopy);


            // Create buffers
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
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)(leafCount * sizeof(float) * 4), null, BufferUsageARB.DynamicDraw);
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
            sunDirection = Vector3.Transform(sunDirection, Matrix4x4.CreateRotationY(0.001f));
        };
    }

    private unsafe uint CreateComputeShader(GL gl, string path)
    {
        uint shader = gl.CreateShader(ShaderType.ComputeShader);
        string source = File.ReadAllText(path);

        // Add a #line directive to help debugger find the source file
        // source = $"#line 1 \"{Path.GetFileName(path)}\"\n" + source;

        gl.ShaderSource(shader, source);
        gl.CompileShader(shader);


        gl.GetShader(shader, ShaderParameterName.CompileStatus, out int status);
        if (status != (int)GLEnum.True)
        {
            throw new Exception($"Failed to compile compute shader: {gl.GetShaderInfoLog(shader)}");
        }

        uint program = gl.CreateProgram();

        gl.AttachShader(program, shader);
        gl.LinkProgram(program);

        // Enable debug output
        gl.Enable(EnableCap.DebugOutput);
        gl.Enable(EnableCap.DebugOutputSynchronous);

        gl.GetProgram(program, ProgramPropertyARB.LinkStatus, out status);
        if (status != (int)GLEnum.True)
        {
            throw new Exception($"Failed to link compute shader: {gl.GetProgramInfoLog(program)}");
        }

        gl.DeleteShader(shader);
        return program;
    }

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
            gl.DrawBuffers(3, drawBuffersPtr);
        }

        // Check framebuffer completeness
        if (gl.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != GLEnum.FramebufferComplete)
        {
            Console.WriteLine("Framebuffer not complete!");
        }

        gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    int octreeOptimizeLevel = 0;
    public unsafe void OptimizeOctree()
    {
        int maxDepth = (int)Math.Log2(octreeGenerator.ChunkSize);

        ctx.ExecuteCmd((dt, gl) =>
        {
            var start = DateTime.Now;

            gl.UseProgram(octreeOptimizeProgram);

            // Set uniforms
            gl.Uniform1(0, maxDepth);
            gl.Uniform1(1, octreeGenerator.voxelSize);
            gl.Uniform1(2, octreeOptimizeLevel); // Current level being processed
            gl.Uniform1(3, octreeGenerator.ChunkSize);

            // Bind buffers
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 0, positionsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 1, materialsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 2, lightLevelsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 3, colorsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 4, changesBuffer);

            // Dispatch compute shader
            const int COMPUTE_GROUP_SIZE = 512;
            int numWorkGroups = (nodeCount + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
            gl.DispatchCompute((uint)numWorkGroups, 1, 1);

            // Memory barrier
            gl.MemoryBarrier(MemoryBarrierMask.ShaderStorageBarrierBit);

            octreeOptimizeLevel = (octreeOptimizeLevel + 1) % (maxDepth + 1);

            var end = DateTime.Now;
            Console.WriteLine($"Octree optimization took {(end - start).TotalMilliseconds}ms");
        });
    }


    // Update the physics simulation method
    public unsafe void UpdatePhysics(double deltaTime, Vector3 gravity)
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            var start = DateTime.Now;
            // Clear move flags buffer at the start of each frame
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, moveFlags);
            gl.ClearBufferData(GLEnum.ShaderStorageBuffer, GLEnum.R8i, GLEnum.RedInteger, GLEnum.Int, null);

            gl.UseProgram(sandPhysicsProgram);

            // Set uniforms
            gl.Uniform1(0, octreeGenerator.ChunkSize);
            gl.Uniform1(1, (float)deltaTime);
            gl.Uniform1(2, currentFrame++);
            gl.Uniform3(3, Vector3.Normalize(gravity)); // Ensure gravity is normalized

            // Bind buffers - note the new moveFlags buffer at binding point 4
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 1, materialsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 2, lightLevelsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 3, colorsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 4, moveFlags);

            // Dispatch compute shader
            const int COMPUTE_GROUP_SIZE = 512;
            int numWorkGroups = (octreeGenerator.LeafCount + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
            gl.DispatchCompute((uint)numWorkGroups, 1, 1);

            // Memory barrier to ensure physics completes before rendering
            gl.MemoryBarrier(MemoryBarrierMask.ShaderStorageBarrierBit);
            var end = DateTime.Now;
            Console.WriteLine($"Physics update took {(end - start).TotalMilliseconds}ms");
        });
    }

    public void CalculateLighting()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            var start = DateTime.Now;
            // Direct lighting pass
            gl.UseProgram(directLightingProgram);

            gl.Uniform3(0, sunDirection);
            gl.Uniform4(1, new float[] { sunColor.X, sunColor.Y, sunColor.Z, 1.0f });
            gl.Uniform1(2, octreeGenerator.ChunkSize);
            gl.Uniform1(3, octreeGenerator.voxelSize);
            gl.Uniform1(4, (uint)DateTime.UtcNow.Ticks);
            gl.Uniform3(5, camera.Position);

            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 0, positionsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 1, materialsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 2, lightLevelsBuffer);
            gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 3, colorsBuffer);

            const uint GROUP_SIZE = 8;
            uint numWorkGroups = (uint)octreeGenerator.ChunkSize / GROUP_SIZE;
            gl.DispatchCompute(numWorkGroups, numWorkGroups, numWorkGroups);

            // Blur pass
            gl.UseProgram(blurProgram);

            gl.Uniform1(0, octreeGenerator.ChunkSize);
            gl.Uniform1(1, octreeGenerator.voxelSize);

            gl.Uniform1(2, 0);
            gl.DispatchCompute(numWorkGroups, numWorkGroups, numWorkGroups);
            gl.MemoryBarrier(MemoryBarrierMask.ShaderStorageBarrierBit);

            gl.Uniform1(2, 1);
            gl.DispatchCompute(numWorkGroups, numWorkGroups, numWorkGroups);
            gl.MemoryBarrier(MemoryBarrierMask.ShaderStorageBarrierBit);

            gl.Uniform1(2, 2);
            gl.DispatchCompute(numWorkGroups, numWorkGroups, numWorkGroups);
            gl.MemoryBarrier(MemoryBarrierMask.ShaderStorageBarrierBit);

            var end = DateTime.Now;
            Console.WriteLine($"Lighting calculation took {(end - start).TotalMilliseconds}ms");
        });
    }

    public void UpdateTerrain(Vector3 centerPosition)
    {
        Console.WriteLine($"Updating terrain centered at {centerPosition}");
        octreeGenerator.GenerateOctree(centerPosition);
        UploadTerrain();
    }

    private unsafe void UploadMaterials(int start, int count)
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(start * sizeof(int)), (nuint)(count * sizeof(int)), new ReadOnlySpan<int>(octreeGenerator.materials, start, count));
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    private unsafe void UploadColors(int start, int count)
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, colorsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, (nint)(start * sizeof(float) * 4), (nuint)(count * sizeof(float) * 4), new ReadOnlySpan<Vector4>(octreeGenerator.colors, start, count));
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    private unsafe void UploadTerrain()
    {
        Console.WriteLine($"Uploading terrain data to GPU, node count: {nodeCount}");
        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, positionsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(nodeCount * sizeof(float) * 4), (ReadOnlySpan<Vector4>)octreeGenerator.positions);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, materialsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(octreeGenerator.LeafCount * sizeof(int)), (ReadOnlySpan<int>)octreeGenerator.materials);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, lightLevelsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(octreeGenerator.LeafCount * sizeof(float) * 4), (ReadOnlySpan<Vector4>)octreeGenerator.lightLevels);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, colorsBuffer);
            gl.BufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(octreeGenerator.LeafCount * sizeof(float) * 4), (ReadOnlySpan<Vector4>)octreeGenerator.colors);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    int counter = 0;
    public void Draw(double dt)
    {
        OptimizeOctree();
        CalculateLighting();
        // if (counter++ < 500)
            // UpdatePhysics(dt, new Vector3(0, -1, 0));

        ctx.RenderCmd((dt, gl) =>
        {
            voxelShader.Use();
            voxelShader.SetUniform("time", (float)window.context.Time);
            voxelShader.SetUniform("resolution", window.Size);
            voxelShader.SetUniform("mouse", window.context.Input.Mice[0].Position);
            voxelShader.SetUniform("view", camera.ViewMatrix);
            voxelShader.SetUniform("projection", camera.PerspectiveMatrix);
            voxelShader.SetUniform("near", camera.Near);
            voxelShader.SetUniform("far", camera.Far);
            voxelShader.SetUniform("sunDir", sunDirection);
            voxelShader.SetUniform("sunColor", sunColor);
            voxelShader.SetUniform("voxelSize", octreeGenerator.voxelSize);
            voxelShader.SetUniform("chunkSize", octreeGenerator.ChunkSize);

            // Bind the octree data buffers
            gl.BindBufferBase(GLEnum.ShaderStorageBuffer, 0, positionsBuffer);
            gl.BindBufferBase(GLEnum.ShaderStorageBuffer, 1, materialsBuffer);
            gl.BindBufferBase(GLEnum.ShaderStorageBuffer, 2, lightLevelsBuffer);
            gl.BindBufferBase(GLEnum.ShaderStorageBuffer, 3, colorsBuffer);

            // Set shader uniforms
            voxelShader.SetUniform("octreeNodeCount", nodeCount);
            voxelShader.SetUniform("maxTreeDepth", 10);

            // Render the scene to the framebuffer
            // gl.BindFramebuffer(FramebufferTarget.Framebuffer, framebuffer);
            // gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            screenQuad.Draw();
        });
    }

    private void DisposeFrameBuffer(GL gl)
    {
        if (depthTexture != 0)
        {
            gl.DeleteTextures(1, ref depthTexture);
            depthTexture = 0;
        }

        if (colorTexture != 0)
        {
            gl.DeleteTextures(1, ref colorTexture);
            colorTexture = 0;
        }

        if (normalTexture != 0)
        {
            gl.DeleteTextures(1, ref normalTexture);
            normalTexture = 0;
        }

        if (positionTexture != 0)
        {
            gl.DeleteTextures(1, ref positionTexture);
            positionTexture = 0;
        }

        if (framebuffer != 0)
        {
            gl.DeleteFramebuffers(1, ref framebuffer);
            framebuffer = 0;
        }
    }

    public void Dispose()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            // Clean up compute shaders
            if (directLightingProgram != 0)
            {
                gl.DeleteProgram(directLightingProgram);
                directLightingProgram = 0;
            }
            if (blurProgram != 0)
            {
                gl.DeleteProgram(blurProgram);
                blurProgram = 0;
            }
            if (sandPhysicsProgram != 0)
            {
                gl.DeleteProgram(sandPhysicsProgram);
                sandPhysicsProgram = 0;
            }
            if (moveFlags != 0)
            {
                gl.DeleteBuffer(moveFlags);
                moveFlags = 0;
            }
            // Add cleanup for octree optimization
            if (octreeOptimizeProgram != 0)
            {
                gl.DeleteProgram(octreeOptimizeProgram);
            }
            if (changesBuffer != 0)
            {
                gl.DeleteBuffer(changesBuffer);
            }

            // Clean up buffers
            if (positionsBuffer != 0)
            {
                gl.DeleteBuffer(positionsBuffer);
                positionsBuffer = 0;
            }
            if (materialsBuffer != 0)
            {
                gl.DeleteBuffer(materialsBuffer);
                materialsBuffer = 0;
            }
            if (lightLevelsBuffer != 0)
            {
                gl.DeleteBuffer(lightLevelsBuffer);
                lightLevelsBuffer = 0;
            }
            if (colorsBuffer != 0)
            {
                gl.DeleteBuffer(colorsBuffer);
                colorsBuffer = 0;
            }

            // Clean up framebuffer and textures
            DisposeFrameBuffer(gl);

            // Dispose of other resources
            voxelShader?.Dispose();
            texShader?.Dispose();
            screenQuad?.Dispose();
            ssao?.Dispose();
        });
    }
}