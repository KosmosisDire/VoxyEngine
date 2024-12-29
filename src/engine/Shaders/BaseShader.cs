using System.Numerics;
using Silk.NET.Maths;
using Silk.NET.OpenGL;

namespace Engine;

public abstract class BaseShader : IDisposable
{
    protected readonly GLContext ctx;
    protected Dictionary<string, object> uniforms = new();
    protected List<ShaderLoader> shaders = [];
    public uint handle { get; protected set; }
    private uint pendingHandle = 0;

    protected BaseShader(GLContext ctx)
    {
        this.ctx = ctx;
        CreateProgram();
    }

    private void CreateProgram()
    {
        ctx.ExecuteCmd((dt, gl) => 
        {
            // Create new program
            pendingHandle = gl.CreateProgram();
            var error = gl.GetError();
            
            if (pendingHandle == 0 || error != GLEnum.NoError)
            {
                throw new InvalidOperationException($"Failed to create program. Error: {error}");
            }
        });
    }

    protected void InitializeShader(ShaderLoader shader)
    {
        if (shader.handle == 0)
        {
            throw new InvalidOperationException($"Cannot attach invalid shader handle 0");
        }

        if (pendingHandle == 0)
        {
            throw new InvalidOperationException("Cannot attach shader to invalid program");
        }

        shader.OnRecompiled += () => HandleShaderRecompiled(shader);
        shader.OnCompilationError += HandleShaderError;
        
        shaders.Add(shader);
        
        ctx.ExecuteCmd((dt, gl) => 
        {
            gl.AttachShader(pendingHandle, shader.handle);
            var error = gl.GetError();
            if (error != GLEnum.NoError)
            {
                throw new InvalidOperationException($"Failed to attach shader {shader.handle} to program {pendingHandle}. Error: {error}");
            }
        });
    }

    protected void LinkProgram()
    {
        if (pendingHandle == 0)
        {
            throw new InvalidOperationException("Cannot link invalid program handle 0");
        }

        ctx.ExecuteCmd((dt, gl) => 
        {
            // First verify the program exists
            if (!gl.IsProgram(pendingHandle))
            {
                throw new InvalidOperationException($"Program {pendingHandle} is not a valid program object");
            }

            // Check that shaders are attached
            gl.GetProgram(pendingHandle, GLEnum.AttachedShaders, out int numAttached);
            if (numAttached == 0)
            {
                throw new InvalidOperationException("No shaders attached to program");
            }

            gl.LinkProgram(pendingHandle);

            // Check for linking errors
            gl.GetProgram(pendingHandle, GLEnum.LinkStatus, out int status);
            
            if (status == 0)
            {
                string infoLog = gl.GetProgramInfoLog(pendingHandle);
                gl.DeleteProgram(pendingHandle);
                pendingHandle = 0;
                throw new Exception($"Error linking program: {infoLog}");
            }
            
            // Validate the program
            gl.ValidateProgram(pendingHandle);
            gl.GetProgram(pendingHandle, GLEnum.ValidateStatus, out int validateStatus);
            
            if (validateStatus == 0)
            {
                string validateLog = gl.GetProgramInfoLog(pendingHandle);
                Console.WriteLine($"Program validation failed: {validateLog}");
            }

            // If we get here, linking succeeded - clean up old program and switch to new one
            if (handle != 0)
            {
                gl.DeleteProgram(handle);
            }

            handle = pendingHandle;
            pendingHandle = 0;
        });
    }

    private void HandleShaderRecompiled(ShaderLoader shader)
    {
        // Store the old uniforms and program handle
        var oldUniforms = new Dictionary<string, object>(uniforms);
        var oldHandle = handle;

        try
        {
            // Create and setup new program
            CreateProgram();
            
            ctx.ExecuteCmd((dt, gl) =>
            {
                // Attach all shaders
                foreach (var s in shaders)
                {
                    gl.AttachShader(pendingHandle, s.handle);
                }
            });

            // Try to link
            LinkProgram();

            // Restore uniforms
            foreach (var uniform in oldUniforms)
            {
                SetUniform(uniform.Key, uniform.Value);
            }

            // Clean up old program at the end
            ctx.ExecuteCmd((dt, gl) =>
            {
                if (oldHandle != 0)
                {
                    gl.DeleteProgram(oldHandle);
                }
            });
        }
        catch (Exception e)
        {
            Console.WriteLine($"Failed to recreate program: {e.Message}");
            // If recreation failed, clean up pending program
            if (pendingHandle != 0)
            {
                ctx.ExecuteCmd((dt, gl) =>
                {
                    gl.DeleteProgram(pendingHandle);
                });
                pendingHandle = 0;
            }
            throw;
        }
    }

    private void HandleShaderError(List<ShaderError> errors)
    {
        foreach (var error in errors)
        {
            Console.WriteLine(error.ToString());
        }
    }

    public void Update()
    {
        // Check all shaders for recompilation
        foreach (var shader in shaders)
        {
            shader.Update();
        }
    }

    public void Use()
    {
        ctx.RenderCmd((dt, gl) => 
        {
            if (handle == 0)
            {
                throw new InvalidOperationException("Cannot use invalid program handle 0");
            }

            Update();

            gl.UseProgram(handle);
            
            foreach (var uniform in uniforms)
            {
                SetUniform(uniform.Key, uniform.Value);
            }
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
            else if (value is uint u)
            {
                gl.Uniform1(location, u);
            }
            else if (value is long l)
            {
                gl.Uniform1(location, l);
            }
            else if (value is ulong ul)
            {
                gl.Uniform1(location, ul);
            }
            else if (value is float f)
            {
                gl.Uniform1(location, f);
            }
            else if (value is double d)
            {
                gl.Uniform1(location, (float)d);
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
                gl.Uniform2(location, vec2di.X, vec2di.Y);
            }
            else if (value is Vector3 vec3)
            {
                gl.Uniform3(location, vec3.X, vec3.Y, vec3.Z);
            }
            else if (value is Vector3I vec3i)
            {
                gl.Uniform3(location, vec3i.X, vec3i.Y, vec3i.Z);
            }
            else if (value is Vector3D<float> vec3df)
            {
                gl.Uniform3(location, vec3df.X, vec3df.Y, vec3df.Z);
            }
            else if (value is Vector3D<int> vec3di)
            {
                gl.Uniform3(location, vec3di.X, vec3di.Y, vec3di.Z);
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

    public virtual void Dispose()
    {
        Console.WriteLine($"Disposing shader {handle}");
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