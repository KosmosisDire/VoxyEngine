using System.Numerics;
using Engine;
using Silk.NET.OpenGL;
using Shader = Engine.Shader;


public class SSAORenderer : IDisposable
{
    private const int SSAO_KERNEL_SIZE = 64;
    private const int NOISE_DIM = 4;  // Noise texture dimension (4x4)

    private uint ssaoFBO, blurFBO;
    private uint ssaoColorBuffer, blurBuffer, noiseTexture;

    private GL gl;
    private GLContext ctx;
    private Shader ssaoShader;
    private Shader blurShader;
    private Vector3[] ssaoKernel;

    private Mesh screenQuad;
    private Vector2 resolution;

    public SSAORenderer(Window window, Mesh screenQuad)
    {
        this.ctx = window.context;
        this.gl = ctx.gl;
        this.resolution = (Vector2)window.Size;
        this.screenQuad = screenQuad;

        window.SilkWindow.FramebufferResize += (size) =>
        {
            resolution = (Vector2)size;
            GenerateSSAOFramebuffer();
        };

        ctx.RenderCmd((dt, gl) =>
        {
            GenerateSSAOKernel();
            GenerateSSAOFramebuffer();
            LoadShaders();
        });

    }

    /// <summary>
    /// Generates SSAO kernel and noise texture.
    /// </summary>
    private void GenerateSSAOKernel()
    {
        // SSAO kernel
        Random rand = new Random();
        ssaoKernel = new Vector3[SSAO_KERNEL_SIZE];

        for (int i = 0; i < SSAO_KERNEL_SIZE; i++)
        {
            Vector3 sample = new Vector3(
                (float)(rand.NextDouble() * 2.0 - 1.0),
                (float)(rand.NextDouble() * 2.0 - 1.0),
                (float)(rand.NextDouble())
            );
            sample = Vector3.Normalize(sample);
            sample *= (float)rand.NextDouble();  // Scale between 0 and 1
            float scale = (float)i / SSAO_KERNEL_SIZE;
            scale = EngineMath.Lerp(0.01f, 1.0f, scale * scale);  // Larger samples towards the center
            sample *= scale;
            ssaoKernel[i] = sample;
        }

        // SSAO noise
        Vector3[] ssaoNoise = new Vector3[NOISE_DIM * NOISE_DIM];
        for (int i = 0; i < NOISE_DIM * NOISE_DIM; ++i)
        {
            ssaoNoise[i] = new Vector3(
                (float)(rand.NextDouble() * 2.0 - 1.0),
                (float)(rand.NextDouble() * 2.0 - 1.0),
                0.0f);  // Rotate only in the XY plane
        }

        // Generate noise texture
        gl.GenTextures(1, out noiseTexture);
        gl.BindTexture(TextureTarget.Texture2D, noiseTexture);
        unsafe
        {
            fixed (Vector3* noisePtr = ssaoNoise)
            {
                gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgb32f, NOISE_DIM, NOISE_DIM, 0, PixelFormat.Rgb, PixelType.Float, noisePtr);
            }
        }
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
    }

    private unsafe void GenerateSSAOFramebuffer()
    {
        DisposeFrameBuffer();

        // SSAO Framebuffer
        gl.GenFramebuffers(1, out ssaoFBO);
        gl.BindFramebuffer(FramebufferTarget.Framebuffer, ssaoFBO);

        // SSAO Color Buffer (result texture)
        gl.GenTextures(1, out ssaoColorBuffer);
        gl.BindTexture(TextureTarget.Texture2D, ssaoColorBuffer);
        gl.TexImage2D(GLEnum.Texture2D, 0, (int)InternalFormat.R32f, (uint)resolution.X, (uint)resolution.Y, 0, PixelFormat.Red, PixelType.Float, null);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, ssaoColorBuffer, 0);

        if (gl.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != GLEnum.FramebufferComplete)
        {
            throw new Exception("SSAO Framebuffer is not complete.");
        }

        gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);  // Unbind framebuffer

        // Blur Framebuffer
        gl.GenFramebuffers(1, out blurFBO);
        gl.BindFramebuffer(FramebufferTarget.Framebuffer, blurFBO);

        // Blur Color Buffer (result texture)
        gl.GenTextures(1, out blurBuffer);
        gl.BindTexture(TextureTarget.Texture2D, blurBuffer);
        gl.TexImage2D(TextureTarget.Texture2D, 0, (int)InternalFormat.R32f, (uint)resolution.X, (uint)resolution.Y, 0, PixelFormat.Red, PixelType.Float, null);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, blurBuffer, 0);

        if (gl.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != GLEnum.FramebufferComplete)
        {
            throw new Exception("Blur Framebuffer is not complete.");
        }

        gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);  // Unbind framebuffer
    }

    /// <summary>
    /// Loads SSAO and blur shaders.
    /// </summary>
    private void LoadShaders()
    {
        ssaoShader = new Shader(ctx, "shaders/vert.glsl", "shaders/ssao-frag.glsl");
        blurShader = new Shader(ctx, "shaders/vert.glsl", "shaders/blur-frag.glsl");
    }

    /// <summary>
    /// Generates the SSAO texture from a given depth buffer.
    /// </summary>
    public uint GenerateSSAO(uint positionTexture, uint normalTexture, Matrix4x4 viewMatrix, Matrix4x4 projectionMatrix)
    {
        // Step 1: SSAO Pass
        gl.BindFramebuffer(FramebufferTarget.Framebuffer, ssaoFBO);
        gl.Clear(ClearBufferMask.ColorBufferBit);

        ssaoShader.Use();
        ssaoShader.SetUniform("projection", projectionMatrix);
        ssaoShader.SetUniform("view", viewMatrix);
        ssaoShader.SetUniform("resolution", resolution);

        // Set SSAO samples
        for (int i = 0; i < SSAO_KERNEL_SIZE; ++i)
        {
            ssaoShader.SetUniform($"samples[{i}]", ssaoKernel[i]);
        }

        // Bind textures
        ssaoShader.SetUniform("positionTexture", 0);
        gl.ActiveTexture(TextureUnit.Texture0);
        gl.BindTexture(TextureTarget.Texture2D, positionTexture);

        ssaoShader.SetUniform("normalTexture", 1);
        gl.ActiveTexture(TextureUnit.Texture1);
        gl.BindTexture(TextureTarget.Texture2D, normalTexture);

        ssaoShader.SetUniform("noiseTexture", 2);
        gl.ActiveTexture(TextureUnit.Texture2);
        gl.BindTexture(TextureTarget.Texture2D, noiseTexture);

        // Render SSAO to ssaoColorBuffer (render quad)
        screenQuad.Draw();

        // Step 2: Blur SSAO texture
        gl.BindFramebuffer(FramebufferTarget.Framebuffer, blurFBO);
        gl.Clear(ClearBufferMask.ColorBufferBit);

        blurShader.Use();
        blurShader.SetUniform("ssaoInput", 0);
        gl.ActiveTexture(TextureUnit.Texture0);
        gl.BindTexture(TextureTarget.Texture2D, ssaoColorBuffer);

        // Render blurred SSAO to blurBuffer (render quad)
        screenQuad.Draw();

        // Unbind framebuffer and return the blurred SSAO buffer handle
        gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

        return blurBuffer;
    }


    /// <summary>
    /// Disposes the resources used by SSAORenderer.
    /// </summary>
    public void Dispose()
    {
        ssaoShader.Dispose();
        gl.DeleteTextures(1, ref noiseTexture);
        DisposeFrameBuffer();
    }

    private void DisposeFrameBuffer()
    {
        if (ssaoFBO != 0)
            gl.DeleteFramebuffers(1, ref ssaoFBO);

        if (ssaoColorBuffer != 0)
            gl.DeleteTextures(1, ref ssaoColorBuffer);

        if (blurBuffer != 0)
            gl.DeleteTextures(1, ref blurBuffer);
    }
}
