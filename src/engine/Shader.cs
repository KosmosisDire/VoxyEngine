using System.Numerics;
using Silk.NET.Maths;
using Silk.NET.OpenGL;

namespace Engine;

public class Shader : IDisposable
{
    public static Shader CurrentShader { get; private set; }
    public static Shader DefaultShader { get; private set; }

    public class ShaderLoader : IDisposable
    {
        private string source;
        public uint handle { get; private set; }
        private GL gl;

        public ShaderLoader(GL gl, string shaderPath, GLEnum shaderType)
        {
            this.gl = gl;
            if (!File.Exists(shaderPath))
            {
                throw new FileNotFoundException($"Shader file not found: {shaderPath}");
            }

            source = File.ReadAllText(shaderPath);
            handle = gl.CreateShader(shaderType);
            gl.ShaderSource(handle, source);
            gl.CompileShader(handle);
        }

        public void Dispose()
        {
            gl.DeleteShader(handle);
            GC.SuppressFinalize(this);
        }
    }

    private List<ShaderLoader> shaders = [];
    public uint handle { get; private set; }

    private Dictionary<string, object> uniforms = new();
    private GLContext ctx;

    public Shader(GLContext ctx, string vertexShaderPath, string fragmentShaderPath)
    {
        this.ctx = ctx;

        ctx.ExecuteCmd((dt, gl) => 
        {
            handle = gl.CreateProgram();
            AttachShader(new ShaderLoader(ctx, vertexShaderPath, GLEnum.VertexShader));
            AttachShader(new ShaderLoader(ctx, fragmentShaderPath, GLEnum.FragmentShader));
            Link();
        });
    }

    private void AttachShader(ShaderLoader shader)
    {
        ctx.ExecuteCmd((dt, gl) => 
        {
            shaders.Add(shader);
            gl.AttachShader(handle, shader.handle);
        });
    }

    private void Link()
    {
        ctx.ExecuteCmd((dt, gl) => 
        {
            gl.LinkProgram(handle);

            // Check for linking errors
            gl.GetProgram(handle, GLEnum.LinkStatus, out int status);
            if (status == 0)
            {
                string infoLog = gl.GetProgramInfoLog(handle);
                throw new Exception($"Error linking program: {infoLog}");
            }
        });
    }

    public void Use()
    {
        ctx.RenderCmd((dt, gl) => 
        {
            gl.UseProgram(handle);

            if (CurrentShader == this) return;
            
            foreach (var uniform in uniforms)
            {
                SetUniform(uniform.Key, uniform.Value);
            }

            CurrentShader = this;
        });
    }

    public void SetUniform<T>(string name, T value)
    {
        ctx.RenderCmd((dt, gl) => 
        {
            int location = gl.GetUniformLocation(handle, name);
            if (value is int i)
            {
                gl.Uniform1(location, i);
            }
            else if (value is float f)
            {
                gl.Uniform1(location, f);
            }
            else if (value is bool b)
            {
                gl.Uniform1(location, b ? 1 : 0);
            }
            else if (value is Vector2 vec2)
            {
                gl.Uniform2(location, vec2.X, vec2.Y);
            }
            else if (value is Vector2D<float> vec2df)
            {
                gl.Uniform2(location, vec2df.X, vec2df.Y);
            }
            else if (value is Vector2D<int> vec2di)
            {
                var vec = vec2di.As<float>();
                gl.Uniform2(location, vec.X, vec.Y);
            }
            else if (value is Vector3 vec3)
            {
                gl.Uniform3(location, vec3.X, vec3.Y, vec3.Z);
            }
            else if (value is Vector3D<float> vec3df)
            {
                gl.Uniform3(location, vec3df.X, vec3df.Y, vec3df.Z);
            }
            else if (value is Vector3D<int> vec3di)
            {
                var vec = vec3di.As<float>();
                gl.Uniform3(location, vec.X, vec.Y, vec.Z);
            }
            else if (value is Vector4 vec4)
            {
                gl.Uniform4(location, vec4.X, vec4.Y, vec4.Z, vec4.W);
            }
            else if (value is Vector4D<float> vec4df)
            {
                gl.Uniform4(location, vec4df.X, vec4df.Y, vec4df.Z, vec4df.W);
            }
            else if (value is Vector4D<int> vec4di)
            {
                var vec = vec4di.As<float>();
                gl.Uniform4(location, vec.X, vec.Y, vec.Z, vec.W);
            }
            else if (value is Silk.NET.SDL.Color color)
            {
                gl.Uniform4(location, color.R / 255f, color.G / 255f, color.B / 255f, color.A / 255f);
            }
            else if (value is System.Drawing.Color colorsys)
            {
                gl.Uniform4(location, colorsys.R / 255f, colorsys.G / 255f, colorsys.B / 255f, colorsys.A / 255f);
            }
            else if (value is Matrix4x4 mat4)
            {
                float[] mat = [
                    mat4.M11, mat4.M12, mat4.M13, mat4.M14,
                    mat4.M21, mat4.M22, mat4.M23, mat4.M24,
                    mat4.M31, mat4.M32, mat4.M33, mat4.M34,
                    mat4.M41, mat4.M42, mat4.M43, mat4.M44
                ];

                gl.UniformMatrix4(location, false, mat.AsSpan());
            }
            else
            {
                throw new ArgumentException($"Unsupported uniform type: {value.GetType()}");
            }

            uniforms[name] = value;
        });
    }

    public void Dispose()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            foreach (var shader in shaders)
            {
                gl.DetachShader(handle, shader.handle);
                shader.Dispose();
            }

            gl.DeleteProgram(handle);
            GC.SuppressFinalize(this);
        });
    }
}