using Silk.NET.OpenGL;

namespace Engine;

public class MeshShader : BaseShader
{
    public static MeshShader CurrentShader { get; private set; }
    public static MeshShader DefaultShader { get; private set; }

    public MeshShader(GLContext ctx, string vertexShaderPath, string fragmentShaderPath) : base(ctx)
    {
        ctx.ExecuteCmd((dt, gl) => 
        {
            InitializeShader(new ShaderLoader(ctx, vertexShaderPath, ShaderType.VertexShader));
            InitializeShader(new ShaderLoader(ctx, fragmentShaderPath, ShaderType.FragmentShader));
            LinkProgram();
        });
    }

    public new void Use()
    {
        base.Use();
        CurrentShader = this;
    }

    public override void Dispose()
    {
        if (CurrentShader == this)
        {
            CurrentShader = null;
        }
        base.Dispose();
    }
}